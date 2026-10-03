#!/usr/bin/env bash
# Stage 0 smoke test: the whole pipeline at toy scale, plus guard, parallel and resume tests.
#
#   GPUS="0" DATA_ROOT=/real/data HF_HOME=... ./smoke_test.sh          # on the server (real data + weights)
#   GPUS=cpu SMOKE_FAKE_DATA=1 SMOKE_RANDOM_INIT=1 ./smoke_test.sh     # offline / no GPU (synthetic data)
#
#  0. unit checks (teacher_forward, KD loss = MaskedKD, recipe asserts, augmentation determinism)
#  A. scheduler end to end, both datasets x {teacher, ce, kd}: 64 train/val images, 1 epoch, batch 16,
#     lr sweeps -> selection -> diagnosis -> eval (64 test images, 8-image corruption cache x 75) -> summary,
#     with RUNS_PER_GPU=3 so student/eval jobs overlap on one GPU (parallel test).
#  B. guards: changed config in an existing run dir, KD without teacher, KD with the other dataset's
#     teacher, wrong split sizes, rerun of a finished run.
#  C. resume: kill -9 a 3-epoch run after epoch 1, rerun the same command, compare with an uninterrupted run.
# Everything is written under SMOKE_ROOT (default ./smoke/run); real outputs/caches are not touched.
set -euo pipefail
cd "$(dirname "$0")"
: "${GPUS:?set GPUS (e.g. GPUS=0, or GPUS=cpu for a CPU-only test)}"
SMOKE_ROOT="$(realpath -m "${SMOKE_ROOT:-smoke/run}")"
RUNS_PER_GPU="${RUNS_PER_GPU:-3}"
rm -rf "$SMOKE_ROOT"
mkdir -p "$SMOKE_ROOT"
export OUTPUT_ROOT="$SMOKE_ROOT/outputs" CORRUPTION_ROOT="$SMOKE_ROOT/corruptions" RESULTS_DIR="$SMOKE_ROOT/results"
export STAGE0_NO_TMUX_HINT=1
if [[ "${SMOKE_FAKE_DATA:-0}" == 1 ]]; then
  export DATA_ROOT="$(realpath -m "${SMOKE_DATA_ROOT:-smoke/data}")"
  uv run python tests/make_fake_data.py
  uv run python -m stage0.prepare_data --skip-download
else
  : "${DATA_ROOT:?set DATA_ROOT (prepared real data) or SMOKE_FAKE_DATA=1}"
  uv run python -m stage0.prepare_data
fi
uv run python tests/test_units.py
INIT=""; [[ "${SMOKE_RANDOM_INIT:-0}" == 1 ]] && INIT="--random-init"
TRAIN_ARGS="--epochs 1 --batch-size 16 --micro-batch 8 --limit 64 $INIT"
DEV_ENV=(); [[ "$GPUS" == cpu ]] && DEV_ENV=(env CUDA_VISIBLE_DEVICES=) || DEV_ENV=(env CUDA_VISIBLE_DEVICES="${GPUS%%,*}")
STRICT=info; [[ "$GPUS" == cpu ]] && STRICT=strict
T0=$(date +%s)

echo; echo "################ A. pipeline + parallel (RUNS_PER_GPU=$RUNS_PER_GPU) ################"
uv run python -m stage0.scheduler --gpus "$GPUS" --runs-per-gpu "$RUNS_PER_GPU" --poll 1 \
  --corruption-workers 4 --train-args "$TRAIN_ARGS" --eval-args "--limit 64" --diag-args "--limit 64" \
  --corruption-args "--n 8"
./status.sh --all | tail -n 5
uv run python tests/check_smoke.py pipeline "$OUTPUT_ROOT"
uv run python tests/check_smoke.py parallel "$OUTPUT_ROOT" 2
test -s "$RESULTS_DIR/stage0_summary.md" && test -s "$RESULTS_DIR/lr_selection.md"
echo "  ok   results/stage0_summary.md and lr_selection.md written"

echo; echo "################ B. guards ################"
expect_fail() {  # expect_fail "<description>" "<pattern in output>" cmd...
  local what="$1" pat="$2"; shift 2
  if out=$("$@" 2>&1); then echo "  FAIL $what: command succeeded"; echo "$out" | tail -5; exit 1; fi
  echo "$out" | grep -q -- "$pat" || { echo "  FAIL $what: unexpected error"; echo "$out" | tail -5; exit 1; }
  echo "  ok   $what -> refused: $(echo "$out" | grep -m1 -- "$pat" | cut -c1-110)"
}
CE_LR=$(uv run python -c "import json;print(json.load(open('$OUTPUT_ROOT/cub/lr_selection.json'))['student']['lr'])")
out=$("${DEV_ENV[@]}" uv run python -m stage0.train --mode ce --dataset cub --seed 1 --lr "$CE_LR" $TRAIN_ARGS 2>&1)
echo "$out" | grep -q "already finished" && echo "  ok   rerun of a finished run is a no-op" || { echo "$out"; exit 1; }
expect_fail "same run dir, different lr" "different settings" \
  "${DEV_ENV[@]}" uv run python -m stage0.train --mode ce --dataset cub --seed 1 --lr 0.123 $TRAIN_ARGS
G="$SMOKE_ROOT/guard_outputs"; mkdir -p "$G"
expect_fail "KD without a teacher" "does not exist" \
  "${DEV_ENV[@]}" uv run python -m stage0.train --mode kd --dataset cub --seed 0 --lr "$CE_LR" $TRAIN_ARGS --output-root "$G"
mkdir -p "$G/cub/teacher"; cp -r "$OUTPUT_ROOT/waterbirds/teacher/seed0" "$G/cub/teacher/seed0"
expect_fail "KD with the other dataset's teacher" "trained for dataset=waterbirds" \
  "${DEV_ENV[@]}" uv run python -m stage0.train --mode kd --dataset cub --seed 0 --lr "$CE_LR" $TRAIN_ARGS --output-root "$G"
BAD="$SMOKE_ROOT/bad_data/waterbirds/waterbird_complete95_forest2water2"; mkdir -p "$BAD"
head -n -1 "$DATA_ROOT/waterbirds/waterbird_complete95_forest2water2/metadata.csv" > "$BAD/metadata.csv"
expect_fail "wrong split sizes" "split sizes" \
  uv run python -m stage0.prepare_data --datasets waterbirds --skip-download --data-root "$SMOKE_ROOT/bad_data"

echo; echo "################ C. kill -9 and resume ################"
R="$SMOKE_ROOT/resume_outputs"
RESUME_CMD=("${DEV_ENV[@]}" uv run python -m stage0.train --mode ce --dataset waterbirds --seed 0 --lr 1e-4
            --epochs 3 --batch-size 16 --micro-batch 8 --limit 64 $INIT --num-workers 1 --output-root "$R")
"${RESUME_CMD[@]}" --subdir ce_reference > /dev/null
"${RESUME_CMD[@]}" > /dev/null 2>&1 &
PID=$!
STATUS="$R/waterbirds/ce/seed0/status.json"
for _ in $(seq 600); do
  if [[ -f "$STATUS" ]] && grep -q '"epoch": 1' "$STATUS"; then break; fi; sleep 0.5
done
pkill -9 -P "$PID" -f stage0.train 2>/dev/null || true; kill -9 "$PID" 2>/dev/null || true; wait "$PID" 2>/dev/null || true
pkill -9 -f "output-root $R" 2>/dev/null || true
sleep 1
test ! -f "$R/waterbirds/ce/seed0/DONE" && echo "  ok   killed after epoch 1 (no DONE yet)"
"${RESUME_CMD[@]}" > /dev/null
uv run python tests/check_smoke.py resume "$R/waterbirds/ce/seed0" "$R/waterbirds/ce_reference/seed0" "$STRICT"

echo; echo "SMOKE TEST PASSED in $(( $(date +%s) - T0 ))s. Outputs: $SMOKE_ROOT"
echo "Summary: $RESULTS_DIR/stage0_summary.md"
