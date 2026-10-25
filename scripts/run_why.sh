#!/usr/bin/env bash
# "Why does MaskedKD's WGA collapse?" in one command, after scripts/run_wga_collapse.sh finished:
#   1) analysis (inference only, ~30-60 min): teacher target per group, bird coverage of kept patches, training
#      trajectory, same-student control, layer-wise CKA, figure  -> results/why/why.md, patches_g2.png
#   2) surgical fine-tuning (training, ~6 h on one A5000): head / last2 / first4 x {kd, maskedkd, rollout} @ 0.15
#      x 3 seeds                                                  -> results/why/surgical.md
#   export DATA_ROOT=$HOME/stage0/data OUTPUT_ROOT=$HOME/stage0/outputs HF_HOME=$HOME/stage0/hf GPUS=0
#   setsid nohup ./scripts/run_why.sh > $HOME/why.log 2>&1 < /dev/null &
# Options (env): SURGICAL=0 (analysis only)  WHY_ARGS="--cka" (extra why_collapse args)  P=4  BACKUP_DIR=...
set -euo pipefail
cd "$(dirname "$0")/.."
: "${OUTPUT_ROOT:?set OUTPUT_ROOT}"; : "${DATA_ROOT:?set DATA_ROOT}"
[[ -n "${GPUS:-}" && "$GPUS" != cpu ]] && export CUDA_VISIBLE_DEVICES=${GPUS%%,*}
RES=${RESULTS_DIR:-results}
step() { echo; echo "======== [$(date '+%F %T')] $* ========"; }
backup() {
  [[ -z "${BACKUP_DIR:-}" ]] && return 0
  mkdir -p "$BACKUP_DIR/results/why" "$BACKUP_DIR/waterbirds"
  cp -a "$RES/why/." "$BACKUP_DIR/results/why/" || { echo "[error] backup failed"; exit 1; }
  for d in "$OUTPUT_ROOT"/waterbirds/*_tb-*; do [[ -d $d ]] && { cp -a "$d" "$BACKUP_DIR/waterbirds/" || exit 1; }; done
  echo "backup -> $BACKUP_DIR"
}
step "1/2 analysis (inference only)"
uv run python -m stage0.why_collapse ${WHY_ARGS:---cka}
backup
if [[ "${SURGICAL:-1}" == 1 ]]; then
  step "2/2 surgical fine-tuning"
  ./scripts/run_surgical.sh
  backup
fi
echo; echo "done: $RES/why/why.md, $RES/why/surgical.md, $RES/why/patches_g2.png"
