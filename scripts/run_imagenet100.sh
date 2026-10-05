#!/usr/bin/env bash
# ImageNet-100 from-scratch check: DeiT-B (ImageNet, frozen) -> DeiT-Ti from random init, P jobs at a time.
#   JOBS="kd ce maskedkd:0.15 rollout:0.15 tam:0.15 tam_r_g0:0.15" EPOCHS=50 SEED=0 P=3 ./scripts/run_imagenet100.sh
# Job names: kd | ce | <criterion>:<keep> | tam_r_g0:<keep> / tam_g0:<keep> (TAM with gap fixed at 0).
# Prereq: scripts/prepare_imagenet100.py (data + teacher). This script also builds the corruption cache (CPU, in the
# background) and the attribution cache (if a tam* job is listed) before training.
# Then: uv run python scripts/compare_runs.py --dataset imagenet100 --keep 0.15 --ref maskedkd --seeds 0 \
#         --runs maskedkd rollout tam tam_r_g0
set -euo pipefail
cd "$(dirname "$0")/.."
: "${OUTPUT_ROOT:?set OUTPUT_ROOT}"; : "${DATA_ROOT:?set DATA_ROOT}"
export DS=imagenet100 EPOCHS=${EPOCHS:-50} SEED=${SEED:-0} LR=${LR:-1.25e-4} EXTRA=${EXTRA:-}
JOBS=${JOBS:-"kd ce maskedkd:0.15 rollout:0.15 tam:0.15 tam_r_g0:0.15"}; P=${P:-3}
export LOGDIR=$OUTPUT_ROOT/_imagenet100_logs; mkdir -p "$LOGDIR"
echo "imagenet100 from scratch: jobs [$JOBS] epochs $EPOCHS seed $SEED lr $LR, $P at a time; logs $LOGDIR"

CROOT=${CORRUPTION_ROOT:-$DATA_ROOT/corruptions}
if [[ ! -f $CROOT/$DS/DONE ]]; then
  echo "corruption cache: building in the background (CPU)"
  nohup uv run python -m stage0.make_corruptions --datasets $DS ${CORRUPTION_ARGS:-} > "$LOGDIR/corruptions.log" 2>&1 &
fi
if [[ " $JOBS " == *" tam"* ]] && [[ ! -f $OUTPUT_ROOT/$DS/attribution_cache/DONE ]]; then
  echo "attribution cache: building"
  uv run python -m stage0.make_attribution_cache --dataset $DS ${ATTR_ARGS:-} > "$LOGDIR/attribution_cache.log" 2>&1 \
    || { echo "FAIL attribution cache (see $LOGDIR)"; exit 1; }
fi

one() {
  local job=$1 mode args sub rd
  case $job in
    kd|ce) mode=$job; sub=$job; args=(--mode "$job") ;;
    tam_r_g0:*|tam_g0:*) local c=${job%%_g0:*} k=${job#*:}; sub="${c}_g0_k${k}"
         args=(--mode maskedkd --mask-criterion "$c" --keep "$k" --tam-gap 0 0 --subdir "$sub") ;;
    *:*) local c=${job%%:*} k=${job#*:}; sub="${c}_k${k}"; args=(--mode maskedkd --mask-criterion "$c" --keep "$k") ;;
    *) echo "unknown job $job"; return 1 ;;
  esac
  rd=$OUTPUT_ROOT/$DS/$sub/seed$SEED
  [[ -f $rd/eval.json ]] && { echo "skip  $sub (eval.json)"; return 0; }
  echo "start $sub"
  if [[ ! -f $rd/DONE ]]; then
    uv run python -m stage0.train "${args[@]}" --dataset $DS --seed "$SEED" --lr "$LR" --epochs "$EPOCHS" \
      --random-init $EXTRA > "$LOGDIR/${sub}_s$SEED.train.log" 2>&1 || { echo "FAIL  $sub train (see $LOGDIR)"; return 1; }
  fi
  while [[ ! -f ${CORRUPTION_ROOT:-$DATA_ROOT/corruptions}/$DS/DONE ]]; do sleep 60; done
  uv run python -m stage0.evaluate --dataset $DS --mode "$sub" --seed "$SEED" ${EVAL_ARGS:-} \
    > "$LOGDIR/${sub}_s$SEED.eval.log" 2>&1 || { echo "FAIL  $sub eval (see $LOGDIR)"; return 1; }
  echo "done  $sub  $(grep -o 'clean [0-9.]*' "$LOGDIR/${sub}_s$SEED.eval.log" | head -1)"
}
export -f one
for j in $JOBS; do echo "$j"; done | xargs -P "$P" -I{} bash -c 'one "$@"' _ {}
echo "all done"
