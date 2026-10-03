#!/usr/bin/env bash
# Stage 1 + 2 smoke test at toy scale (CUB only).
#
#   GPUS=cpu SMOKE_FAKE_DATA=1 SMOKE_RANDOM_INIT=1 [SMOKE_SHM_MB=64] ./smoke_test_stage12.sh   # offline / no GPU
#   GPUS=0 DATA_ROOT=/real/data HF_HOME=... ./smoke_test_stage12.sh                              # server
#
#  0. unit checks: Stage 0 (tests/test_units.py) and Stage 1/2 (tests/test_stage12.py)
#  1. Stage 0 prerequisites for CUB at toy scale (teacher, lr selection, CE / KD seeds, evals) via the Stage 0
#     scheduler, then a fingerprint of those Stage 0 run dirs (must be unchanged at the end: read only)
#  2. attribution cache -> Stage 1 fidelity (64 images per view, all criteria / keeps)
#  3. Stage 2 scheduler: maskedkd and random at keep 0.5, seed 0, 1 epoch -> eval -> stage2 summary
#  4. pairing: kd vs maskedkd vs random (same seed, 2 epochs, micro-batch 8) see identical student input
#     batches and mixup targets (STAGE0_BATCH_HASH_LOG); ckpt_e1.pt and mask_agree are written
# SMOKE_SHM_MB=64 remounts /dev/shm to that size for the test (root only) and restores it afterwards.
# Everything is written under SMOKE_ROOT (default ./smoke/run12).
set -euo pipefail
cd "$(dirname "$0")"
: "${GPUS:?set GPUS (e.g. GPUS=0, or GPUS=cpu for a CPU-only test)}"
SMOKE_ROOT="$(realpath -m "${SMOKE_ROOT:-smoke/run12}")"
rm -rf "$SMOKE_ROOT"; mkdir -p "$SMOKE_ROOT"
export OUTPUT_ROOT="$SMOKE_ROOT/outputs" CORRUPTION_ROOT="$SMOKE_ROOT/corruptions" RESULTS_DIR="$SMOKE_ROOT/results"
export STAGE0_NO_TMUX_HINT=1

if [[ -n "${SMOKE_SHM_MB:-}" ]]; then
  OLD_SHM_KB=$(df --output=size -k /dev/shm | tail -1 | tr -d ' ')
  mount -o remount,size="${SMOKE_SHM_MB}M" /dev/shm
  trap 'mount -o remount,size='"${OLD_SHM_KB}"'k /dev/shm' EXIT
fi
echo "/dev/shm: $(df -h /dev/shm | tail -1 | awk '{print $2}')"

if [[ "${SMOKE_FAKE_DATA:-0}" == 1 ]]; then
  export DATA_ROOT="$(realpath -m "${SMOKE_DATA_ROOT:-smoke/data}")"
  uv run python tests/make_fake_data.py
  uv run python -m stage0.prepare_data --skip-download --datasets cub
else
  : "${DATA_ROOT:?set DATA_ROOT (prepared real data) or SMOKE_FAKE_DATA=1}"
fi
INIT=""; [[ "${SMOKE_RANDOM_INIT:-0}" == 1 ]] && INIT="--random-init"
TRAIN_ARGS="--epochs 1 --batch-size 16 --micro-batch 8 --limit 64 $INIT"
DEV_ENV=(); [[ "$GPUS" == cpu ]] && DEV_ENV=(env CUDA_VISIBLE_DEVICES=) || DEV_ENV=(env CUDA_VISIBLE_DEVICES="${GPUS%%,*}")
T0=$(date +%s)

echo; echo "################ 0. unit checks ################"
uv run python tests/test_units.py
uv run python tests/test_stage12.py

echo; echo "################ 1. Stage 0 prerequisites (CUB, toy scale) ################"
uv run python -m stage0.scheduler --gpus "$GPUS" --runs-per-gpu 3 --poll 1 --datasets cub \
  --corruption-workers 4 --train-args "$TRAIN_ARGS" --eval-args "--limit 64" --diag-args "--limit 64" \
  --corruption-args "--n 8"
fingerprint() { (cd "$OUTPUT_ROOT/cub" && find teacher ce kd lrsel lr_selection.json -type f -print0 | sort -z \
  | xargs -0 sha1sum) ; }
fingerprint > "$SMOKE_ROOT/stage0_before.sha1"
echo "  ok   fingerprinted $(wc -l < "$SMOKE_ROOT/stage0_before.sha1") Stage 0 files"

echo; echo "################ 2. Stage 1: attribution cache + fidelity ################"
"${DEV_ENV[@]}" uv run python -m stage0.make_attribution_cache --dataset cub --limit 64 --batch-size 16
"${DEV_ENV[@]}" uv run python -m stage0.make_attribution_cache --dataset cub --limit 64 --batch-size 16 | grep -q "nothing to do"
echo "  ok   cache rebuild is a no-op"
"${DEV_ENV[@]}" uv run python -m stage0.fidelity --dataset cub --limit 64 --batch-size 16
uv run python - <<PY
import csv
rows = list(csv.DictReader(open("$RESULTS_DIR/stage1_fidelity.csv")))
crit = {r["criterion"] for r in rows}
want = {"maskedkd", "rollout", "random", "teacher_cache:attn_last", "teacher_cache:rollout",
        "teacher_oracle:attn_last", "teacher_oracle:rollout"}
assert crit == want, crit
for r in rows:
    assert 0 <= float(r["agree"]) <= 1 and float(r["kl"]) >= -1e-6, r
    if r["keep"] == "1.0":
        assert float(r["agree"]) == 1.0 and float(r["kl"]) == 0.0
assert not [r for r in rows if r["view"] == "test_clean" and r["criterion"].startswith("teacher_cache")]
assert len({r["seed"] for r in rows if r["criterion"] == "random"}) == 3
g = {float(r["keep"]): float(r["teacher_gflops"]) for r in rows}
assert 0.4 < g[0.5] / g[1.0] < 0.55, g
print(f"  ok   stage1_fidelity.csv: {len(rows)} rows, criteria {sorted(crit)}")
PY
test -s "$RESULTS_DIR/stage1_fidelity.md" && test -s "$RESULTS_DIR/stage1_fidelity.png" && echo "  ok   .md / .png written"

echo; echo "################ 3. Stage 2 scheduler (2 runs, 1 epoch) ################"
uv run python -m stage0.scheduler --stage 2 --gpus "$GPUS" --runs-per-gpu 3 --poll 1 --datasets cub \
  --criteria maskedkd random --keeps 0.5 --seeds 0 --train-args "$TRAIN_ARGS" --eval-args "--limit 64"
for c in maskedkd random; do
  rd="$OUTPUT_ROOT/cub/${c}_k0.5/seed0"
  test -f "$rd/DONE" && test -f "$rd/eval.json" && head -1 "$rd/log.csv" | grep -q mask_agree
  uv run python -c "import json; c=json.load(open('$rd/config.json')); assert c['mode']=='maskedkd' and c['mask_criterion']=='$c' and c['keep_k']==98 and c['teacher_ckpt'].endswith('cub/teacher/seed0/best.pt'), c"
  echo "  ok   ${c}_k0.5/seed0: DONE, eval.json, mask_agree logged, config ok"
done
test -s "$RESULTS_DIR/stage2_summary.md" && grep -q "maskedkd" "$RESULTS_DIR/stage2_summary.md" && echo "  ok   stage2_summary.md written"
test -f "$OUTPUT_ROOT/_scheduler_stage2/state.json" && echo "  ok   Stage 2 scheduler state kept apart from Stage 0's"
OUTPUT_ROOT="$OUTPUT_ROOT" ./status.sh --stage 2 | tail -2

echo; echo "################ 4. pairing: kd vs maskedkd vs random (2 epochs) ################"
LR=$(uv run python -c "import json;print(json.load(open('$OUTPUT_ROOT/cub/lr_selection.json'))['student']['lr'])")
P="--dataset cub --seed 1 --lr $LR --epochs 2 --batch-size 16 --micro-batch 8 --limit 64 --num-workers 2 $INIT"
STAGE0_BATCH_HASH_LOG="$SMOKE_ROOT/hash_kd.txt" "${DEV_ENV[@]}" uv run python -m stage0.train --mode kd $P --subdir pair_kd > /dev/null
STAGE0_BATCH_HASH_LOG="$SMOKE_ROOT/hash_mkd.txt" "${DEV_ENV[@]}" uv run python -m stage0.train --mode maskedkd \
  --mask-criterion maskedkd --keep 0.3 --ckpt-epochs 1 $P --subdir pair_maskedkd > /dev/null
STAGE0_BATCH_HASH_LOG="$SMOKE_ROOT/hash_rnd.txt" "${DEV_ENV[@]}" uv run python -m stage0.train --mode maskedkd \
  --mask-criterion random --keep 0.3 --ckpt-epochs 1 $P --subdir pair_random > /dev/null
test "$(wc -l < "$SMOKE_ROOT/hash_kd.txt")" -eq 8
cmp "$SMOKE_ROOT/hash_kd.txt" "$SMOKE_ROOT/hash_mkd.txt" && cmp "$SMOKE_ROOT/hash_kd.txt" "$SMOKE_ROOT/hash_rnd.txt"
echo "  ok   kd / maskedkd / random: identical input-batch and mixup-target hashes (8 steps)"
for d in pair_maskedkd pair_random; do
  test -f "$OUTPUT_ROOT/cub/$d/seed1/ckpt_e1.pt" && ! test -f "$OUTPUT_ROOT/cub/$d/seed1/ckpt_e2.pt"
done
test ! -e "$OUTPUT_ROOT/cub/pair_kd/seed1/ckpt_e1.pt"
echo "  ok   ckpt_e1.pt written for maskedkd modes only"

echo; echo "################ Stage 0 outputs untouched ################"
fingerprint > "$SMOKE_ROOT/stage0_after.sha1"
cmp "$SMOKE_ROOT/stage0_before.sha1" "$SMOKE_ROOT/stage0_after.sha1"
echo "  ok   cub/{teacher,ce,kd,lrsel} and lr_selection.json byte-identical after Stage 1/2"

echo; echo "STAGE 1+2 SMOKE TEST PASSED in $(( $(date +%s) - T0 ))s (/dev/shm $(df -h /dev/shm | tail -1 | awk '{print $2}')). Outputs: $SMOKE_ROOT"
