#!/usr/bin/env bash
# Ablation: where does tam_r lose to rollout? tam_r with a FIXED gap ratio g (no curriculum):
#   g=0   -> teacher candidate pool only (rollout ranking inside the cached teacher top-2k)
#   g=0.5 -> pool + 50% gap tokens from the first epoch
# Output dirs: {dataset}/{CRIT}_g{g}_k{keep}/seed{s}/  (CRIT default tam_r)  (next to the scheduler's runs; nothing existing is touched)
#
#   DATA_ROOT=... OUTPUT_ROOT=... HF_HOME=... ./scripts/ablate_gap.sh            # waterbirds, keep 0.15, seeds 0-2
#   DS=cub KEEP=0.15 GAPS="0 0.5" SEEDS="0 1" P=3 ./scripts/ablate_gap.sh
#   CRIT=tam GAPS=0 ./scripts/ablate_gap.sh      # tam (S = last-block attention) instead of tam_r -> tam_g0_k0.15
# Then: uv run python scripts/compare_runs.py --dataset waterbirds --keep 0.15
set -euo pipefail
cd "$(dirname "$0")/.."
: "${OUTPUT_ROOT:?set OUTPUT_ROOT}"; : "${DATA_ROOT:?set DATA_ROOT}"
export DS=${DS:-waterbirds} KEEP=${KEEP:-0.15} CRIT=${CRIT:-tam_r}
GAPS=${GAPS:-"0 0.5"}; SEEDS=${SEEDS:-"0 1 2"}; P=${P:-3}
export LR=$(uv run python -c "from stage0 import common as C; import os; print(C.load_json(os.path.join(os.environ['OUTPUT_ROOT'], '$DS', 'lr_selection.json'))['student']['lr'])")
export LOGDIR=$OUTPUT_ROOT/_ablation_logs; mkdir -p "$LOGDIR"
echo "criterion $CRIT dataset $DS keep $KEEP lr $LR gaps [$GAPS] seeds [$SEEDS], $P at a time; logs in $LOGDIR"

one() {
  local g=$1 s=$2 sub="${CRIT}_g${1}_k${KEEP}" rd
  rd=$OUTPUT_ROOT/$DS/$sub/seed$s
  [[ -f $rd/eval.json ]] && { echo "skip  $sub/seed$s (eval.json)"; return 0; }
  echo "start $sub/seed$s"
  if [[ ! -f $rd/DONE ]]; then
    uv run python -m stage0.train --mode maskedkd --mask-criterion "$CRIT" --keep "$KEEP" --tam-gap "$g" "$g" \
      --dataset "$DS" --seed "$s" --lr "$LR" --subdir "$sub" > "$LOGDIR/${DS}_${sub}_s$s.train.log" 2>&1 \
      || { echo "FAIL  $sub/seed$s train (see $LOGDIR)"; return 1; }
  fi
  uv run python -m stage0.evaluate --dataset "$DS" --mode "$sub" --seed "$s" > "$LOGDIR/${DS}_${sub}_s$s.eval.log" 2>&1 \
    || { echo "FAIL  $sub/seed$s eval (see $LOGDIR)"; return 1; }
  echo "done  $sub/seed$s"
}
export -f one
for g in $GAPS; do for s in $SEEDS; do echo "$g $s"; done; done | xargs -P "$P" -L 1 bash -c 'one "$0" "$1"'
echo "all done: uv run python scripts/compare_runs.py --dataset $DS --keep $KEEP"
