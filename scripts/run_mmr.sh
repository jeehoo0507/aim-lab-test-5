#!/usr/bin/env bash
# MaskedKD + sink filter / MMR diversity (mk_filt, mk_mmr, mk_fmmr) vs maskedkd, one dataset, P runs at a time.
#   DS=waterbirds ./scripts/run_mmr.sh                    # pretrained student, lr from lr_selection.json, seeds 0-2
#   DS=imagenet100 SEEDS="0 1" ./scripts/run_mmr.sh       # from scratch: --random-init, 50 epochs, lr 1.25e-4
# Jobs: CRITS (default "mk_filt mk_mmr mk_fmmr maskedkd"); "kd" also allowed. Finished runs (eval.json) are skipped,
# so maskedkd / kd that already exist are not rerun.
# Then: uv run python scripts/compare_runs.py --dataset $DS --keep $KEEP --ref maskedkd --seeds ... \
#         --runs maskedkd rollout tam_r_g0 mk_filt mk_mmr mk_fmmr
set -euo pipefail
cd "$(dirname "$0")/.."
: "${OUTPUT_ROOT:?set OUTPUT_ROOT}"; : "${DATA_ROOT:?set DATA_ROOT}"
export DS=${DS:-waterbirds} KEEP=${KEEP:-0.15}
CRITS=${CRITS:-"mk_filt mk_mmr mk_fmmr maskedkd"}; P=${P:-3}
if [[ $DS == imagenet100 ]]; then
  SEEDS=${SEEDS:-"0 1"}; export LR=${LR:-1.25e-4} EXTRA="--epochs ${EPOCHS:-50} --random-init"
else
  SEEDS=${SEEDS:-"0 1 2"}; export EXTRA=""
  export LR=$(uv run python -c "from stage0 import common as C; import os; print(C.load_json(os.path.join(os.environ['OUTPUT_ROOT'], '$DS', 'lr_selection.json'))['student']['lr'])")
fi
export LOGDIR=$OUTPUT_ROOT/_mmr_logs; mkdir -p "$LOGDIR"
echo "dataset $DS keep $KEEP lr $LR jobs [$CRITS] seeds [$SEEDS] extra [$EXTRA], $P at a time; logs $LOGDIR"

one() {
  local c=$1 s=$2 sub args rd
  if [[ $c == kd ]]; then sub=kd; args=(--mode kd); else sub="${c}_k${KEEP}"; args=(--mode maskedkd --mask-criterion "$c" --keep "$KEEP"); fi
  rd=$OUTPUT_ROOT/$DS/$sub/seed$s
  [[ -f $rd/eval.json ]] && { echo "skip  $sub/seed$s (eval.json)"; return 0; }
  echo "start $sub/seed$s"
  if [[ ! -f $rd/DONE ]]; then
    uv run python -m stage0.train "${args[@]}" --dataset "$DS" --seed "$s" --lr "$LR" $EXTRA \
      > "$LOGDIR/${DS}_${sub}_s$s.train.log" 2>&1 || { echo "FAIL  $sub/seed$s train (see $LOGDIR)"; return 1; }
  fi
  uv run python -m stage0.evaluate --dataset "$DS" --mode "$sub" --seed "$s" > "$LOGDIR/${DS}_${sub}_s$s.eval.log" 2>&1 \
    || { echo "FAIL  $sub/seed$s eval (see $LOGDIR)"; return 1; }
  echo "done  $sub/seed$s  $(grep -o 'clean [0-9.]*' "$LOGDIR/${DS}_${sub}_s$s.eval.log" | head -1)"
}
export -f one
for s in $SEEDS; do for c in $CRITS; do echo "$c $s"; done; done | xargs -P "$P" -L 1 bash -c 'one "$0" "$1"'
echo "all done"
