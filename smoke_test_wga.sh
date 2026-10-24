#!/usr/bin/env bash
# Smoke test for scripts/run_wga_collapse.sh: the whole pipeline at toy scale (fake Waterbirds, random init, CPU ok).
#   ./smoke_test_wga.sh                      # GPUS=cpu, fake data, ~10-20 min on CPU
#   GPUS=0 SMOKE_REAL_DATA=1 DATA_ROOT=... ./smoke_test_wga.sh     # real data + real weights, 64 images
# Writes only under smoke/wga/ (outputs, results, backup). Checks: Stage 0 + 2 runs finish, checkpoints exist,
# eval.json has clean / WGA / group / corruption, summary md/csv/png are written, verdict lines present, backup copied.
set -euo pipefail
cd "$(dirname "$0")"
export GPUS=${GPUS:-cpu}
SM="$(realpath -m smoke/wga)"; rm -rf "$SM"; mkdir -p "$SM"
export OUTPUT_ROOT="$SM/outputs" CORRUPTION_ROOT="$SM/corruptions" RESULTS_DIR="$SM/results" BACKUP_DIR="$SM/backup"
INIT="--random-init"
if [[ "${SMOKE_REAL_DATA:-0}" == 1 ]]; then
  : "${DATA_ROOT:?set DATA_ROOT}"; INIT=""; export PREP_ARGS=""
else
  export DATA_ROOT="$SM/data" PREP_ARGS="--skip-download"
  FAKE_DATASETS=waterbirds uv run python tests/make_fake_data.py
fi
[[ "$GPUS" == cpu ]] && export CUDA_VISIBLE_DEVICES=
export TRAIN_ARGS="--epochs 1 --batch-size 16 --micro-batch 8 --limit 64 --num-workers 0 $INIT" \
       EVAL_ARGS="--limit 64" DIAG_ARGS="--limit 64" CORRUPTION_ARGS="--n 8" \
       KEEPS="0.3 0.15" SEEDS="0 1" RUNS_PER_GPU=${RUNS_PER_GPU:-$([[ $GPUS == cpu ]] && echo 1 || echo 3)} POLL=1 PREFLIGHT=0
T0=$(date +%s)
./scripts/run_wga_collapse.sh

echo; echo "################ checks ################"
O=$OUTPUT_ROOT/waterbirds
for d in teacher/seed0 ce/seed0 kd/seed0 kd/seed1 maskedkd_k0.3/seed0 maskedkd_k0.15/seed1 rollout_k0.3/seed1 rollout_k0.15/seed0; do
  test -f "$O/$d/DONE" && test -f "$O/$d/last.pt" && test -f "$O/$d/eval.json" || { echo "FAIL missing run files: $d"; exit 1; }
done
uv run python - <<PY
import json
e = json.load(open("$O/maskedkd_k0.15/seed0/eval.json"))
assert {"clean_acc", "wga", "group_acc", "corruption_acc"} <= set(e) and len(e["group_acc"]) == 4, e.keys()
c = json.load(open("$O/rollout_k0.15/seed0/config.json"))
assert c["mask_criterion"] == "rollout" and c["keep_k"] == 29, c
PY
S=$RESULTS_DIR/wga_collapse.md
grep -q "^## 1. 재현 확인" "$S" && grep -q "| maskedkd | 0.15 | 2 |" "$S" && grep -q "| rollout | 0.3 | 2 |" "$S"
grep -q "MaskedKD 붕괴 지점" "$S" && grep -q "rollout − MaskedKD WGA" "$S" && grep -q "^| Full KD | – |" "$S"
test -s "$RESULTS_DIR/wga_collapse.csv" && test -s "$RESULTS_DIR/wga_collapse.png"
test -f "$BACKUP_DIR/waterbirds/maskedkd_k0.15/seed1/last.pt" && test -f "$BACKUP_DIR/results/wga_collapse.md"
echo "  ok   runs (Stage 0 + 8 Stage 2), eval fields, summary md/csv/png, verdicts, backup with checkpoints"
echo "  rerun = no-op check:"; ./scripts/run_wga_collapse.sh > "$SM/rerun.log" 2>&1 && ! grep "start " "$SM/rerun.log" | grep -qv summarize \
  && echo "  ok   second run trains / evaluates nothing (only the summaries refresh)" \
  || { echo "FAIL rerun started jobs (see $SM/rerun.log)"; exit 1; }
echo; echo "SMOKE OK in $(( $(date +%s) - T0 )) s. Summary: $S"
