#!/usr/bin/env bash
# Smoke test for scripts/run_layer.sh on top of smoke_test_more.sh's toy outputs (run that first). CPU ok, ~10 min.
set -euo pipefail
cd "$(dirname "$0")"
export GPUS=${GPUS:-cpu}
SM="$(realpath -m smoke/more)"; test -d "$SM/outputs/waterbirds/kd" || { echo "run ./smoke_test_more.sh first"; exit 1; }
export OUTPUT_ROOT="$SM/outputs" CORRUPTION_ROOT="$SM/corruptions" RESULTS_DIR="$SM/results" BACKUP_DIR="$SM/backup" DATA_ROOT="$SM/data"
[[ "$GPUS" == cpu ]] && export CUDA_VISIBLE_DEVICES=
export TRAIN_ARGS="--epochs 1 --batch-size 16 --micro-batch 8 --limit 64 --num-workers 0 --random-init --ckpt-epochs 1" \
       EVAL_ARGS="--limit 64" ATTR_ARGS="--limit 64" SEG_ARGS="--limit 64" SUMMARY_ARGS="--seeds 0 1 --no-flops" \
       WHY_ARGS="--per-group 8 --seeds 0 1 --batch-size 16" SELECT_ARGS="--per-group 8 --seeds 0 1 --batch-size 16" \
       SEEDS="0 1" P=2 OPTIONAL=1
T0=$(date +%s)
./scripts/run_layer.sh > "$SM/layer.log" 2>&1 || { tail -40 "$SM/layer.log"; echo "FAIL run_layer"; exit 1; }
L=$(uv run python -c "import json; print(json.load(open('$RESULTS_DIR/why_more/layer_select.json'))['layer'])")
O=$OUTPUT_ROOT/waterbirds
for j in mk_L${L}_k0.15 mk_L${L}_k0.1 mk_L${L}_k0.05 oracle_bird_k0.1 oracle_bird_k0.05 maskedkd_k0.1 mk_dh2_k0.15; do
  for s in 0 1; do test -f "$O/$j/seed$s/eval.json" || { echo "FAIL missing $j/seed$s"; exit 1; }; done
done
grep -q "L\\\\\* = $L" "$RESULTS_DIR/why_more/layer_select.md" && grep -q "| L0 |" "$RESULTS_DIR/why_more/layer_select.md"
for s in "R6 MaskedKD @ L$L" "R7 token 수 한계" "MaskedKD − head 2" "val로 고름"; do
  grep -qF "$s" "$RESULTS_DIR/more.md" || { echo "FAIL more.md lacks: $s"; exit 1; }
done
grep -q "| mk_L$L | 0.15 |" "$RESULTS_DIR/why_more/why_more_C_2.md" && grep -q "| mk_dh2 | 0.15 |" "$RESULTS_DIR/why_more/why_more_C_2.md"
! grep -q "## B." "$RESULTS_DIR/why_more/why_more_C_2.md"
echo "SMOKE OK in $(( $(date +%s) - T0 )) s (L* = $L)"
