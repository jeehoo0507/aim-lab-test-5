#!/usr/bin/env bash
# Waterbirds WGA collapse, end to end in one command (resumable: rerun = only what is missing):
#   1) preflight  2) data  3) Stage 0 (lr sweeps, teacher, CE, KD x3, corruption cache, eval)
#   4) MaskedKD + rollout x keeps x seeds (+ eval)  5) summary -> results/wga_collapse.{md,csv,png}
#   6) optional backup of run dirs incl. checkpoints (BACKUP_DIR)
#
#   export DATA_ROOT=$HOME/stage0/data OUTPUT_ROOT=$HOME/stage0/outputs HF_HOME=$HOME/stage0/hf GPUS=0
#   setsid nohup ./scripts/run_wga_collapse.sh > $HOME/wga.log 2>&1 < /dev/null &
#   tail -f $HOME/wga.log        ;  cat results/wga_collapse.md
#
# Options (env): KEEPS="0.5 0.3 0.25 0.2 0.15"  SEEDS="0 1 2"  CRITERIA="maskedkd rollout"  RUNS_PER_GPU=3
#                BACKUP_DIR=/shared/users/<me>/wga   (rsync after each stage; checkpoints included)
# Smoke-only (env): TRAIN_ARGS EVAL_ARGS DIAG_ARGS CORRUPTION_ARGS PREP_ARGS PREFLIGHT=0  (see smoke_test_wga.sh)
set -euo pipefail
cd "$(dirname "$0")/.."
: "${GPUS:?set GPUS (e.g. GPUS=0)}"; : "${DATA_ROOT:?set DATA_ROOT}"; : "${OUTPUT_ROOT:?set OUTPUT_ROOT}"
export STAGE0_NO_TMUX_HINT=1
DS=waterbirds
KEEPS=${KEEPS:-"0.5 0.3 0.25 0.2 0.15"}; SEEDS=${SEEDS:-"0 1 2"}; CRITERIA=${CRITERIA:-"maskedkd rollout"}
TRAIN_ARGS=${TRAIN_ARGS:-}; EVAL_ARGS=${EVAL_ARGS:-}; DIAG_ARGS=${DIAG_ARGS:-}; CORRUPTION_ARGS=${CORRUPTION_ARGS:-}
SCHED=(--gpus "$GPUS" --datasets $DS --train-args "$TRAIN_ARGS" --eval-args "$EVAL_ARGS")
[[ -n "${RUNS_PER_GPU:-}" ]] && SCHED+=(--runs-per-gpu "$RUNS_PER_GPU")
[[ -n "${POLL:-}" ]] && SCHED+=(--poll "$POLL")
step() { echo; echo "======== [$(date '+%F %T')] $* ========"; }
backup() {
  [[ -z "${BACKUP_DIR:-}" ]] && return 0
  mkdir -p "$BACKUP_DIR"
  local res="${RESULTS_DIR:-results}"
  mkdir -p "$BACKUP_DIR/results"
  if command -v rsync >/dev/null; then
    rsync -a "$OUTPUT_ROOT/$DS" "$BACKUP_DIR/" || { echo "[error] backup failed"; exit 1; }
    rsync -a "$res/" "$BACKUP_DIR/results/" || { echo "[error] backup failed"; exit 1; }
  else                                   # no rsync on the box: plain copy (overwrites, never deletes)
    cp -a "$OUTPUT_ROOT/$DS" "$BACKUP_DIR/" || { echo "[error] backup failed"; exit 1; }
    cp -a "$res/." "$BACKUP_DIR/results/" || { echo "[error] backup failed"; exit 1; }
  fi
  echo "backup -> $BACKUP_DIR ($(du -sh "$BACKUP_DIR" | cut -f1))"
}
mkdir -p "$OUTPUT_ROOT"

if [[ "${PREFLIGHT:-1}" == 1 && ! -f $OUTPUT_ROOT/preflight.json ]]; then
  step "1/5 preflight"; uv run python -m stage0.preflight
fi
step "2/5 data ($DS)"
uv run python -m stage0.prepare_data --datasets $DS ${PREP_ARGS:-}

step "3/5 Stage 0: teacher / CE / KD (+ corruption cache, eval)"
uv run python -m stage0.scheduler "${SCHED[@]}" --diag-args "$DIAG_ARGS" --corruption-args "$CORRUPTION_ARGS"
backup

step "4/5 MaskedKD collapse sweep: [$CRITERIA] x keep [$KEEPS] x seed [$SEEDS]"
uv run python -m stage0.scheduler --stage 2 "${SCHED[@]}" --corruption-args "$CORRUPTION_ARGS" \
  --criteria $CRITERIA --keeps $KEEPS --seeds $SEEDS
backup

step "5/5 summary"
uv run python -m stage0.summarize_wga --dataset $DS --criteria $CRITERIA --keeps $KEEPS --seeds $SEEDS
backup
echo; echo "done: ${RESULTS_DIR:-results}/wga_collapse.md  (.csv, .png)"
