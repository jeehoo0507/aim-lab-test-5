#!/usr/bin/env bash
# Stage 3 pilot (TAM) smoke test at toy scale (CUB only).
#
#   GPUS=cpu SMOKE_FAKE_DATA=1 SMOKE_RANDOM_INIT=1 [SMOKE_SHM_MB=64] ./smoke_test_stage3_pilot.sh   # offline / no GPU
#   GPUS=0 DATA_ROOT=/real/data HF_HOME=... ./smoke_test_stage3_pilot.sh                              # server
#
#  0. unit checks: Stage 0, Stage 1/2, Stage 3 pilot
#  1. Stage 0 prerequisites for CUB at toy scale via the Stage 0 scheduler; fingerprint of the Stage 0 run dirs
#  2. attribution cache + cache diagnosis (diagnose_cache: same-view oracle overlap must be 100%)
#  3. Stage 2 scheduler, two calls as on the server: maskedkd rollout tam tam_var @ keep 0.3 and
#     maskedkd tam_oracle @ keep 0.15, seeds 0 1, 1 epoch -> eval -> stage2 summary with pilot rows and verdict
#  4. pairing: kd / maskedkd / rollout / tam / tam_var / tam_oracle (same seed, 2 epochs) see identical input
#     batches and mixup targets
#  5. maskedkd and random training on this branch == on the stage12 branch (git worktree; final weights equal)
# SMOKE_SHM_MB=64 remounts /dev/shm to that size for the test (root only) and restores it afterwards.
set -euo pipefail
cd "$(dirname "$0")"
REPO="$PWD"
: "${GPUS:?set GPUS (e.g. GPUS=0, or GPUS=cpu for a CPU-only test)}"
SMOKE_ROOT="$(realpath -m "${SMOKE_ROOT:-smoke/run3}")"
rm -rf "$SMOKE_ROOT"; mkdir -p "$SMOKE_ROOT"
export OUTPUT_ROOT="$SMOKE_ROOT/outputs" CORRUPTION_ROOT="$SMOKE_ROOT/corruptions" RESULTS_DIR="$SMOKE_ROOT/results"
export STAGE0_NO_TMUX_HINT=1
WT="$SMOKE_ROOT/stage12_worktree"
cleanup() {
  git -C "$REPO" worktree remove --force "$WT" 2>/dev/null || true
  [[ -n "${OLD_SHM_KB:-}" ]] && mount -o remount,size="${OLD_SHM_KB}k" /dev/shm
  true
}
trap cleanup EXIT
if [[ -n "${SMOKE_SHM_MB:-}" ]]; then
  OLD_SHM_KB=$(df --output=size -k /dev/shm | tail -1 | tr -d ' ')
  mount -o remount,size="${SMOKE_SHM_MB}M" /dev/shm
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
uv run python tests/test_stage3.py

echo; echo "################ 1. Stage 0 prerequisites (CUB, toy scale) ################"
uv run python -m stage0.scheduler --gpus "$GPUS" --runs-per-gpu 3 --poll 1 --datasets cub \
  --corruption-workers 4 --train-args "$TRAIN_ARGS" --eval-args "--limit 64" --diag-args "--limit 64" \
  --corruption-args "--n 8"
fingerprint() { (cd "$OUTPUT_ROOT/cub" && find teacher ce kd lrsel lr_selection.json -type f -print0 | sort -z \
  | xargs -0 sha1sum) ; }
fingerprint > "$SMOKE_ROOT/stage0_before.sha1"
echo "  ok   fingerprinted $(wc -l < "$SMOKE_ROOT/stage0_before.sha1") Stage 0 files"

echo; echo "################ 2. attribution cache + cache diagnosis ################"
"${DEV_ENV[@]}" uv run python -m stage0.make_attribution_cache --dataset cub --limit 64 --batch-size 16 > /dev/null
"${DEV_ENV[@]}" uv run python -m stage0.diagnose_cache --dataset cub --limit 64 --batch-size 16 | tail -1
grep -q "Sanity (same view's oracle twice, first batch): overlap 100.0%" "$RESULTS_DIR/stage3_cache_diag.md"
grep -q "^## 결론" "$RESULTS_DIR/stage3_cache_diag.md" && test -s "$RESULTS_DIR/stage3_cache_diag.csv"
echo "  ok   stage3_cache_diag.{csv,md}: same-view oracle overlap 100%, conclusion line written"

echo; echo "################ 3. Stage 2 scheduler: pilot criteria (two calls, seeds 0 1) ################"
uv run python -m stage0.scheduler --stage 2 --gpus "$GPUS" --runs-per-gpu 3 --poll 1 --datasets cub \
  --criteria maskedkd rollout tam tam_var tam_r --keeps 0.3 --seeds 0 1 --train-args "$TRAIN_ARGS" --eval-args "--limit 64" \
  --attribution-args "--limit 64 --batch-size 16"
uv run python -m stage0.scheduler --stage 2 --gpus "$GPUS" --runs-per-gpu 3 --poll 1 --datasets cub \
  --criteria maskedkd tam_oracle --keeps 0.15 --seeds 0 1 --train-args "$TRAIN_ARGS" --eval-args "--limit 64"
for c in maskedkd_k0.3 rollout_k0.3 tam_k0.3 tam_var_k0.3 tam_r_k0.3 maskedkd_k0.15 tam_oracle_k0.15; do for s in 0 1; do
  test -f "$OUTPUT_ROOT/cub/$c/seed$s/DONE" && test -f "$OUTPUT_ROOT/cub/$c/seed$s/eval.json"
done; done
echo "  ok   14 runs DONE + eval.json"
uv run python - <<PY
import csv, json
out = "$OUTPUT_ROOT/cub"
def last(rd):
    return list(csv.DictReader(open(f"{rd}/log.csv")))[-1]
for c, keep, k in (("tam", 0.3, 59), ("tam_var", 0.3, 59), ("tam_r", 0.3, 59), ("tam_oracle", 0.15, 29)):
    for s in (0, 1):
        rd = f"{out}/{c}_k{keep:g}/seed{s}"
        cfg = json.load(open(f"{rd}/config.json"))
        assert cfg["mask_criterion"] == c and cfg["tam_kind"] == "attn_last" and cfg["tam_gap"] == [0.1, 0.5]
        assert cfg["tam_budget"] == ("bucket" if c == "tam_var" else "fixed")
        assert cfg["tam_delta"] == (0.33 if c == "tam_var" else None)
        assert cfg["tam_teacher_signal"] == ("oracle" if c == "tam_oracle" else "cache")
        assert cfg["selection_gflops"] > (17 if c == "tam_oracle" else 0)
        r = last(rd)
        assert float(r["tam_gap_ratio"]) == 0.1 and r["mask_agree"] and abs(float(r["mean_k"]) - k) <= 1
        assert r["bucket_k"] == ("40/59/78" if c == "tam_var" else f"{k}/{k}/{k}"), r["bucket_k"]
        assert float(r["teacher_gflops"]) > 0
        if c == "tam_oracle":
            assert "cache_oracle_overlap" not in r
        else:
            assert 0.0 <= float(r["cache_oracle_overlap"]) <= 1.0
    print(f"  ok   {c}_k{keep:g} seeds 0,1: config keys, mean_k, bucket_k, teacher_gflops"
          + ("" if c == "tam_oracle" else ", cache_oracle_overlap") + " logged")
cfg = json.load(open(f"{out}/rollout_k0.3/seed0/config.json"))
assert 0.3 < cfg["selection_gflops"] < 0.6 and "tam_gap" not in cfg
for c in ("maskedkd_k0.3", "maskedkd_k0.15"):
    cfg = json.load(open(f"{out}/{c}/seed0/config.json"))
    assert not {"tam_gap", "tam_teacher_signal", "selection_gflops"} & set(cfg)
print("  ok   rollout config records selection_gflops; maskedkd configs unchanged")
PY
S="$RESULTS_DIR/stage2_summary.md"
grep -q "선택 비용 GFLOPs/img" "$S" && grep -q "| rollout | 0.3 |" "$S" && grep -q "| tam_var | 0.3 | .* (measured)" "$S"
grep -q "| tam_oracle | 0.15 | .*진단용, 비용 비교 대상 아님" "$S"
grep -q "tam − rollout, keep 0.3" "$S" && grep -q "tam_oracle − maskedkd, keep 0.15" "$S"
grep -q "학습 중 cache_oracle_overlap" "$S" && grep -q "### 파일럿 판단" "$S"
echo "  ok   stage2_summary.md: pilot rows, selection cost, paired differences, cache overlap, verdict:"
grep -A2 "### 파일럿 판단" "$S" | tail -1

echo; echo "################ 4. pairing: kd / maskedkd / rollout / tam / tam_var / tam_oracle (2 epochs) ################"
LR=$(uv run python -c "import json;print(json.load(open('$OUTPUT_ROOT/cub/lr_selection.json'))['student']['lr'])")
P="--dataset cub --seed 1 --lr $LR --epochs 2 --batch-size 16 --micro-batch 8 --limit 64 --num-workers 2 $INIT"
run() { local tag=$1; shift; STAGE0_BATCH_HASH_LOG="$SMOKE_ROOT/hash_$tag.txt" "${DEV_ENV[@]}" uv run python -m stage0.train "$@" $P --subdir "pair_$tag" > /dev/null; }
run kd --mode kd
run mkd --mode maskedkd --mask-criterion maskedkd --keep 0.3
run roll --mode maskedkd --mask-criterion rollout --keep 0.3
run tam --mode maskedkd --mask-criterion tam --keep 0.3
run tamvar --mode maskedkd --mask-criterion tam_var --keep 0.3
run tamor --mode maskedkd --mask-criterion tam_oracle --keep 0.15
test "$(wc -l < "$SMOKE_ROOT/hash_kd.txt")" -eq 8
for t in mkd roll tam tamvar tamor; do cmp "$SMOKE_ROOT/hash_kd.txt" "$SMOKE_ROOT/hash_$t.txt"; done
echo "  ok   kd / maskedkd / rollout / tam / tam_var / tam_oracle: identical input-batch and mixup-target hashes (8 steps)"
uv run python -c "
import csv; r=[float(x['tam_gap_ratio']) for x in csv.DictReader(open('$OUTPUT_ROOT/cub/pair_tam/seed1/log.csv'))]
assert r == [0.1, 0.5], r; print('  ok   gap curriculum over 2 epochs:', r)"

echo; echo "################ 5. maskedkd / random training identical to stage12 ################"
REF=stage12; git -C "$REPO" rev-parse -q --verify stage12 >/dev/null || REF=origin/stage12
git -C "$REPO" worktree add -q --detach "$WT" "$REF"
for c in maskedkd random; do
  "${DEV_ENV[@]}" uv run python -m stage0.train --mode maskedkd --mask-criterion $c --keep 0.3 $P --subdir "bit_new_$c" > /dev/null
  (cd "$WT" && "${DEV_ENV[@]}" "$REPO/.venv/bin/python" -m stage0.train --mode maskedkd --mask-criterion $c --keep 0.3 \
     $P --subdir "bit_old_$c" > /dev/null)
  uv run python - <<PY
import csv, torch
o = "$OUTPUT_ROOT/cub"
a = torch.load(f"{o}/bit_new_$c/seed1/last.pt", map_location="cpu", weights_only=False)["model"]
b = torch.load(f"{o}/bit_old_$c/seed1/last.pt", map_location="cpu", weights_only=False)["model"]
same = all(torch.equal(a[k], b[k]) for k in a)
ra = [{k: v for k, v in r.items() if k != "epoch_time_s"} for r in csv.DictReader(open(f"{o}/bit_new_$c/seed1/log.csv"))]
rb = [{k: v for k, v in r.items() if k != "epoch_time_s"} for r in csv.DictReader(open(f"{o}/bit_old_$c/seed1/log.csv"))]
strict = "$GPUS" == "cpu"
if strict:
    assert same and ra == rb, ("$c", same, ra, rb)
    print("  ok   $c: final weights and log.csv bit-identical to stage12")
else:
    print(f"  info $c: weights identical={same}, logs identical={ra == rb} (GPU kernels are not bit-deterministic)")
PY
done

echo; echo "################ Stage 0 outputs untouched ################"
fingerprint > "$SMOKE_ROOT/stage0_after.sha1"
cmp "$SMOKE_ROOT/stage0_before.sha1" "$SMOKE_ROOT/stage0_after.sha1"
echo "  ok   cub/{teacher,ce,kd,lrsel} and lr_selection.json byte-identical"

echo; echo "STAGE 3 PILOT SMOKE TEST PASSED in $(( $(date +%s) - T0 ))s (/dev/shm $(df -h /dev/shm | tail -1 | awk '{print $2}')). Outputs: $SMOKE_ROOT"
