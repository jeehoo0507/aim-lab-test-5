#!/usr/bin/env bash
# Follow-up sweep in one command (after run_wga_collapse.sh; resumable: rerun = only what is missing):
#   1) caches: teacher attribution (TrustMask) + CUB bird segmentation (oracle)          ~ 2-3 min
#   2) analysis on existing checkpoints (inference): signal decomposition + background-driven errors   ~10-20 min
#   3) training, in priority order, P at a time on one GPU, each run followed by its eval:
#        random 0.15 | oracle_bird 0.15 | oracle_bg 0.15 | mk_filt 0.15 | TrustMask (tam_r g=0) 0.15 |
#        random 0.25 | rollout 0.1 | rollout 0.05            (x seeds 0 1 2 = 24 runs)
#   4) analysis of the new runs (inference) + summary with pre-registered verdicts + efficiency table
#   -> results/more.md, results/why_more/why_more.md
#
#   export DATA_ROOT=$HOME/stage0/data OUTPUT_ROOT=$HOME/stage0/outputs HF_HOME=$HOME/stage0/hf GPUS=0
#   setsid nohup ./scripts/run_more.sh > $HOME/more.log 2>&1 < /dev/null &      # terminal can be closed
#   tail -f $HOME/more.log
# Options (env): SKIP_AB=1 (skip step 2)  C_SUFFIX=_2 (name of the part-C report)  P=3 (runs at a time)  SEEDS="0 1 2"  JOBS="random:0.15 ..." (override the list)  BACKUP_DIR=...
# Smoke-only (env): TRAIN_ARGS EVAL_ARGS WHY_ARGS SEG_ARGS ATTR_ARGS SUMMARY_ARGS
set -euo pipefail
cd "$(dirname "$0")/.."
: "${OUTPUT_ROOT:?set OUTPUT_ROOT}"; : "${DATA_ROOT:?set DATA_ROOT}"
[[ -n "${GPUS:-}" && "$GPUS" != cpu ]] && export CUDA_VISIBLE_DEVICES=${GPUS%%,*}
DS=waterbirds; RES=${RESULTS_DIR:-results}; P=${P:-3}; SEEDS=${SEEDS:-"0 1 2"}
JOBS=${JOBS:-"random:0.15 oracle_bird:0.15 oracle_bg:0.15 mk_filt:0.15 tam_r_g0:0.15 random:0.25 rollout:0.1 rollout:0.05"}
export STAGE0_CONCURRENT_RUNS=$P STAGE0_NO_TMUX_HINT=1
step() { echo; echo "======== [$(date '+%F %T')] $* ========"; }
backup() {
  [[ -z "${BACKUP_DIR:-}" ]] && return 0
  mkdir -p "$BACKUP_DIR/results" "$BACKUP_DIR/$DS"
  cp -a "$RES/." "$BACKUP_DIR/results/" || { echo "[error] backup failed"; exit 1; }
  for j in $JOBS; do d=$OUTPUT_ROOT/$DS/${j%%:*}_k${j##*:}; [[ -d $d ]] && { cp -a "$d" "$BACKUP_DIR/$DS/" || exit 1; }; done
  echo "backup -> $BACKUP_DIR"
}

exec 9>"$OUTPUT_ROOT/_run_more.lock"
flock -n 9 || { echo "[error] another run_more.sh is running (lock $OUTPUT_ROOT/_run_more.lock)"; exit 1; }

step "0/4 checks"
for d in teacher/seed0 kd/seed0 maskedkd_k0.15/seed0; do
  [[ -f $OUTPUT_ROOT/$DS/$d/DONE ]] || { echo "[error] $OUTPUT_ROOT/$DS/$d not finished: run scripts/run_wga_collapse.sh first"; exit 1; }
done
LR=$(uv run python -c "from stage0 import common as C; import os; print(C.load_json(os.path.join(os.environ['OUTPUT_ROOT'], '$DS', 'lr_selection.json'))['student']['lr'])")
N=0; for j in $JOBS; do for s in $SEEDS; do N=$((N + 1)); done; done
uv run python - "$OUTPUT_ROOT/$DS" "$N" "$P" <<'PY'
import csv, glob, statistics, sys
root, n, p = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
import json, os
fs = glob.glob(f"{root}/maskedkd_k0.15/seed*/log.csv")
t = [float(r["epoch_time_s"]) for f in fs for r in csv.DictReader(open(f))]
if t:
    ep = json.load(open(os.path.join(os.path.dirname(fs[0]), "config.json")))["epochs"]
    per = statistics.mean(t) * ep / 3600
    print(f"estimate: {n} runs, {statistics.mean(t):.0f} s/epoch in the last sweep -> ~{per:.2f} h per run there; "
          f"with {p} at a time ~ {n * per / p:.1f} h + eval/analysis (assumes the same concurrency as that sweep)")
PY
echo "lr $LR, $P runs at a time, jobs: $JOBS x seeds [$SEEDS]"

step "1/4 caches (TrustMask teacher map, bird segmentation)"
uv run python -m stage0.make_attribution_cache --dataset $DS ${ATTR_ARGS:-}
uv run python -m stage0.seg_cache --dataset $DS ${SEG_ARGS:-}

if [[ "${SKIP_AB:-0}" != 1 ]]; then
  step "2/4 analysis on existing checkpoints (signal decomposition, background-driven errors)"
  uv run python -m stage0.why_more --parts A B ${WHY_ARGS:-}
  cp "$RES/why_more/why_more.md" "$RES/why_more/why_more_AB.md"
  backup
fi

step "3/4 training ($N runs, priority order)"
export DS LR EXTRA=${TRAIN_ARGS:-} EVX=${EVAL_ARGS:-} LOGDIR=$OUTPUT_ROOT/_more_logs; mkdir -p "$LOGDIR"
one() {
  local crit=$1 keep=$2 s=$3 sub="${1}_k${2}" args rd
  if [[ $crit == tam_r_g0 ]]; then args=(--mask-criterion tam_r --tam-gap 0 0)
  else args=(--mask-criterion "$crit"); fi
  rd=$OUTPUT_ROOT/$DS/$sub/seed$s
  [[ -f $rd/eval.json ]] && { echo "skip  $sub/seed$s"; return 0; }
  echo "start $sub/seed$s  [$(date '+%T')]"
  if [[ ! -f $rd/DONE ]]; then
    uv run python -m stage0.train --mode maskedkd "${args[@]}" --keep "$keep" --subdir "$sub" --dataset $DS \
      --seed "$s" --lr "$LR" $EXTRA > "$LOGDIR/${sub}_s$s.train.log" 2>&1 || { echo "FAIL  $sub/seed$s train ($LOGDIR)"; return 0; }
  fi
  uv run python -m stage0.evaluate --dataset $DS --mode "$sub" --seed "$s" $EVX > "$LOGDIR/${sub}_s$s.eval.log" 2>&1 \
    || { echo "FAIL  $sub/seed$s eval ($LOGDIR)"; return 0; }
  echo "done  $sub/seed$s  [$(date '+%T')]"
}
export -f one
for j in $JOBS; do for s in $SEEDS; do echo "${j%%:*} ${j##*:} $s"; done; done \
  | xargs -P "$P" -L 1 bash -c 'one "$0" "$1" "$2"'
backup

step "4/4 analysis of the new runs + summary"
RUNS=$(for j in maskedkd:0.15 rollout:0.15 $JOBS; do echo -n "$j "; done)
uv run python -m stage0.why_more --parts C --runs $RUNS ${WHY_ARGS:-}
cp "$RES/why_more/why_more.md" "$RES/why_more/why_more_C${C_SUFFIX:-}.md"
uv run python -m stage0.summarize_more ${SUMMARY_ARGS:-}
backup
NF=$(grep -l "Traceback\|\[error\]" "$LOGDIR"/*.log 2>/dev/null | wc -l || true)
echo; echo "done: $RES/more.md, $RES/why_more/why_more_AB.md, $RES/why_more/why_more_C.md  (logs with errors: $NF)"
