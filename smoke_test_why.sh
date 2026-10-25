#!/usr/bin/env bash
# Smoke test for scripts/run_why.sh: toy pipeline (fake Waterbirds + fake masks, random init, CPU ok, ~20 min):
# Stage 0 + 4 Stage 2 runs (with an epoch-1 checkpoint) -> analysis (all sections + figure + CKA) -> surgical runs.
#   ./smoke_test_why.sh            # writes only under smoke/why/
set -euo pipefail
cd "$(dirname "$0")"
export GPUS=${GPUS:-cpu}
SM="$(realpath -m smoke/why)"; rm -rf "$SM"; mkdir -p "$SM"
export OUTPUT_ROOT="$SM/outputs" CORRUPTION_ROOT="$SM/corruptions" RESULTS_DIR="$SM/results" BACKUP_DIR="$SM/backup"
export DATA_ROOT="$SM/data" PREP_ARGS="--skip-download"
FAKE_DATASETS=waterbirds FAKE_SEG=1 uv run python tests/make_fake_data.py
[[ "$GPUS" == cpu ]] && export CUDA_VISIBLE_DEVICES=
export TRAIN_ARGS="--epochs 1 --batch-size 16 --micro-batch 8 --limit 64 --num-workers 0 --random-init --ckpt-epochs 1" \
       EVAL_ARGS="--limit 64" DIAG_ARGS="--limit 64" CORRUPTION_ARGS="--n 8" \
       KEEPS="0.3 0.15" SEEDS="0 1" RUNS_PER_GPU=${RUNS_PER_GPU:-$([[ $GPUS == cpu ]] && echo 1 || echo 3)} POLL=1 PREFLIGHT=0
T0=$(date +%s)
./scripts/run_wga_collapse.sh > "$SM/wga.log" 2>&1 || { tail -30 "$SM/wga.log"; echo "FAIL toy pipeline"; exit 1; }
echo "  ok   toy Stage 0 + Stage 2"
export WHY_ARGS="--per-group 8 --seeds 0 1 --ckpts e1 last --mkd-keeps 0.3 0.15 --rollout-keeps 0.3 0.15 --control-keeps 0.15 --cka --cka-n 24 --fig-n 3 --batch-size 16"
export BLOCKS="head last2" P=$([[ $GPUS == cpu ]] && echo 1 || echo 3)
SEEDS="0 1" ./scripts/run_why.sh

echo; echo "################ checks ################"
W=$RESULTS_DIR/why
for s in "## 1. teacher target (train" "## 1. teacher target (test" "## 2. 새 커버리지" "## 3. 새를 덜" "## 4. 학습 중" \
         "## 5. 같은 student" "## 6. 판정" "H1 teacher target" "H2 새 증거" "원인 분리" "## 7. 층별"; do
  grep -qF "$s" "$W/why.md" || { echo "FAIL why.md lacks: $s"; exit 1; }
done
grep -q "| maskedkd | 0.15 |" "$W/why.md" && grep -q "| e1 |" "$W/why.md"
test -s "$W/patches_g2.png" && test -s "$W/records.csv.gz"
uv run python - <<PY
import csv, gzip, math
rows = list(csv.DictReader(gzip.open("$W/records.csv.gz", "rt")))
labs = {r["label"] for r in rows}
assert {"maskedkd", "rollout", "KD student→maskedkd", "KD student→rollout"} <= labs, labs
bs = [float(r["bird_sel"]) for r in rows]
assert all(0 <= b <= 1 for b in bs), "bird_sel out of range (masks not read?)"
# fake bird = lower 2/3 of the image -> image-level bird share ~ 0.6-0.7 after the eval crop
bi = sum(float(r["bird_img"]) for r in rows) / len(rows)
assert 0.5 < bi < 0.8, bi
print(f"  ok   records: {len(rows)} rows, labels {sorted(labs)}, mean image bird share {bi:.2f}")
PY
grep -q "| head |" "$W/surgical.md" && grep -q "| last2 |" "$W/surgical.md" && grep -q "head만 학습" "$W/surgical.md"
grep -q "train_blocks" "$OUTPUT_ROOT/waterbirds/maskedkd_k0.15_tb-head/seed0/config.json"
grep -q "surgical: train_blocks=head, trainable params 386" "$OUTPUT_ROOT/_surgical_logs/maskedkd_k0.15_tb-head_s0.train.log"
test -f "$BACKUP_DIR/results/why/why.md" && test -f "$BACKUP_DIR/waterbirds/kd_tb-head/seed0/last.pt"
echo "  ok   why.md sections + verdicts, figure, records, surgical runs/summary, head-only trainable = head, backup"
echo; echo "SMOKE OK in $(( $(date +%s) - T0 )) s. Reports: $W/why.md, $W/surgical.md"
