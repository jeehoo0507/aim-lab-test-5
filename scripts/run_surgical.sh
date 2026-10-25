#!/usr/bin/env bash
# Surgical fine-tuning check (Lee et al. 2022, arXiv 2210.11466): does MaskedKD's WGA still collapse when only
# part of the student trains? KD vs MaskedKD (vs rollout) at one keep, train_blocks in {head, last2, first4}.
#   head  = linear probe on the ImageNet-pretrained DeiT-Ti features -> if WGA still collapses, the cause is the
#           teacher TARGET alone (features cannot change)
#   last2 / first4 = late vs early layers adapt
# Output dirs: waterbirds/{kd,maskedkd_k0.15,rollout_k0.15}_tb-{blocks}/seed{s}; same lr as the full runs.
#   OUTPUT_ROOT=... DATA_ROOT=... ./scripts/run_surgical.sh          # P=4 runs at a time
#   BLOCKS="head" SEEDS="0 1 2" CRITS="maskedkd rollout" KEEP=0.15 ./scripts/run_surgical.sh
# Then: uv run python -m stage0.summarize_surgical
set -euo pipefail
cd "$(dirname "$0")/.."
: "${OUTPUT_ROOT:?set OUTPUT_ROOT}"; : "${DATA_ROOT:?set DATA_ROOT}"
export DS=waterbirds KEEP=${KEEP:-0.15} EXTRA=${TRAIN_ARGS:-} EVX=${EVAL_ARGS:-}
BLOCKS=${BLOCKS:-"head last2 first4"}; SEEDS=${SEEDS:-"0 1 2"}; CRITS=${CRITS:-"maskedkd rollout"}; P=${P:-4}
export LR=$(uv run python -c "from stage0 import common as C; import os; print(C.load_json(os.path.join(os.environ['OUTPUT_ROOT'], '$DS', 'lr_selection.json'))['student']['lr'])")
export LOGDIR=$OUTPUT_ROOT/_surgical_logs; mkdir -p "$LOGDIR"
echo "surgical: blocks [$BLOCKS] x [kd $CRITS] keep $KEEP x seeds [$SEEDS], lr $LR, $P at a time; logs $LOGDIR"
one() {
  local tb=$1 c=$2 s=$3 sub args rd
  if [[ $c == kd ]]; then sub="kd_tb-$tb"; args=(--mode kd)
  else sub="${c}_k${KEEP}_tb-$tb"; args=(--mode maskedkd --mask-criterion "$c" --keep "$KEEP"); fi
  rd=$OUTPUT_ROOT/$DS/$sub/seed$s
  [[ -f $rd/eval.json ]] && { echo "skip  $sub/seed$s"; return 0; }
  echo "start $sub/seed$s"
  if [[ ! -f $rd/DONE ]]; then
    uv run python -m stage0.train "${args[@]}" --train-blocks "$tb" --subdir "$sub" --dataset $DS --seed "$s" \
      --lr "$LR" $EXTRA > "$LOGDIR/${sub}_s$s.train.log" 2>&1 || { echo "FAIL  $sub/seed$s train"; return 1; }
  fi
  uv run python -m stage0.evaluate --dataset $DS --mode "$sub" --seed "$s" $EVX > "$LOGDIR/${sub}_s$s.eval.log" 2>&1 \
    || { echo "FAIL  $sub/seed$s eval"; return 1; }
  echo "done  $sub/seed$s"
}
export -f one
for tb in $BLOCKS; do for c in kd $CRITS; do for s in $SEEDS; do echo "$tb $c $s"; done; done; done \
  | xargs -P "$P" -L 1 bash -c 'one "$0" "$1" "$2"'
uv run python -m stage0.summarize_surgical --keep "$KEEP" --blocks $BLOCKS --crits $CRITS --seeds $SEEDS
