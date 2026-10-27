#!/usr/bin/env bash
# Smoke test for scripts/run_more.sh: toy Stage 0 + Stage 2 (fake Waterbirds + fake masks, random init, CPU ok),
# then the follow-up sweep (all 8 jobs x 2 seeds, 1 epoch), both analyses and the summary.
#   ./smoke_test_more.sh            # writes only under smoke/more/
set -euo pipefail
cd "$(dirname "$0")"
export GPUS=${GPUS:-cpu}
SM="$(realpath -m smoke/more)"; rm -rf "$SM"; mkdir -p "$SM"
export OUTPUT_ROOT="$SM/outputs" CORRUPTION_ROOT="$SM/corruptions" RESULTS_DIR="$SM/results" BACKUP_DIR="$SM/backup"
export DATA_ROOT="$SM/data" PREP_ARGS="--skip-download"
FAKE_DATASETS=waterbirds FAKE_SEG=1 uv run python tests/make_fake_data.py
[[ "$GPUS" == cpu ]] && export CUDA_VISIBLE_DEVICES=
export TRAIN_ARGS="--epochs 1 --batch-size 16 --micro-batch 8 --limit 64 --num-workers 0 --random-init --ckpt-epochs 1" \
       EVAL_ARGS="--limit 64" DIAG_ARGS="--limit 64" CORRUPTION_ARGS="--n 8" \
       KEEPS="0.25 0.15" SEEDS="0 1" RUNS_PER_GPU=${RUNS_PER_GPU:-$([[ $GPUS == cpu ]] && echo 1 || echo 3)} POLL=1 PREFLIGHT=0
T0=$(date +%s)
./scripts/run_wga_collapse.sh > "$SM/wga.log" 2>&1 || { tail -30 "$SM/wga.log"; echo "FAIL toy pipeline"; exit 1; }
echo "  ok   toy Stage 0 + Stage 2"
export ATTR_ARGS="--limit 64" SEG_ARGS="--limit 64" SUMMARY_ARGS="--seeds 0 1 --no-flops" \
       WHY_ARGS="--per-group 8 --seeds 0 1 --batch-size 16" P=$([[ $GPUS == cpu ]] && echo 2 || echo 3)
./scripts/run_more.sh > "$SM/more.log" 2>&1 || { tail -40 "$SM/more.log"; echo "FAIL run_more"; exit 1; }
tail -5 "$SM/more.log"

echo; echo "################ checks ################"
O=$OUTPUT_ROOT/waterbirds
for j in random_k0.15 oracle_bird_k0.15 oracle_bg_k0.15 mk_filt_k0.15 tam_r_g0_k0.15 random_k0.25 rollout_k0.1 rollout_k0.05; do
  for s in 0 1; do test -f "$O/$j/seed$s/eval.json" || { echo "FAIL missing $j/seed$s/eval.json"; exit 1; }; done
done
echo "  ok   16 runs trained + evaluated"
grep -q '"mask_criterion": "tam_r"' "$O/tam_r_g0_k0.15/seed0/config.json" && grep -q '"tam_gap": \[' "$O/tam_r_g0_k0.15/seed0/config.json"
grep -q "oracle_signal" "$O/oracle_bird_k0.15/seed0/config.json"
uv run python - <<PY
import csv
def last(p): return list(csv.DictReader(open(p)))[-1]
b = last("$O/oracle_bird_k0.15/seed0/log.csv"); g = last("$O/oracle_bg_k0.15/seed0/log.csv")
sb, sg, vb = float(b["sel_bird_share"]), float(g["sel_bird_share"]), float(b["view_bird_share"])
# fake bird = lower 2/3 of the image: oracle_bird keeps (almost) only bird, oracle_bg (almost) none
assert sb > vb > sg, (sb, vb, sg)
assert sb - sg > 0.4, (sb, sg)
print(f"  ok   oracle selection: bird-only keeps {sb:.2f} bird, bg-only {sg:.2f} (view {vb:.2f})")
PY
W=$RESULTS_DIR/why_more
for s in "## A. 같은 student" "### sink 토큰" "## B. teacher 오류는 배경" "### 판정"; do
  grep -qF "$s" "$W/why_more_AB.md" || { echo "FAIL why_more_AB.md lacks: $s"; exit 1; }
done
grep -qF "## C. 학습된 각 run" "$W/why_more_C.md" && grep -q "| oracle_bird | 0.15 |" "$W/why_more_C.md" \
  && grep -q "| tam_r_g0 | 0.15 |" "$W/why_more_C.md" && grep -q "| attn_v |" "$W/why_more_AB.md" && grep -q "| L0 |" "$W/why_more_AB.md"
uv run python - <<PY
import csv, gzip
rows = [r for r in csv.DictReader(gzip.open("$W/records.csv.gz", "rt"))]
ob = [float(r["bird_sel"]) for r in rows if r["score"] == "oracle_bird" and r["bird_sel"] != "nan"]
og = [float(r["bird_sel"]) for r in rows if r["score"] == "oracle_bg" and r["bird_sel"] != "nan"]
assert sum(ob) / len(ob) > 0.9 and sum(og) / len(og) < 0.1, (sum(ob) / len(ob), sum(og) / len(og))
print(f"  ok   re-selection: oracle_bird {sum(ob)/len(ob):.2f}, oracle_bg {sum(og)/len(og):.2f}")
PY
for s in "## 1. 결과" "## 2. 판정" "R1 random 대조" "R2 oracle" "R3 sink" "R4 TrustMask" "R5 rollout 한계"; do
  grep -qF "$s" "$RESULTS_DIR/more.md" || { echo "FAIL more.md lacks: $s"; exit 1; }
done
test -f "$BACKUP_DIR/results/more.md" && test -f "$BACKUP_DIR/waterbirds/oracle_bg_k0.15/seed1/eval.json"
echo "  ok   why_more (A/B/C) + more.md verdicts + backup"
echo; echo "SMOKE OK in $(( $(date +%s) - T0 )) s. Reports: $RESULTS_DIR/more.md, $W/why_more_AB.md, $W/why_more_C.md"
