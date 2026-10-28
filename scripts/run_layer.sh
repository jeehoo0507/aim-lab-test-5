#!/usr/bin/env bash
# Round 3 in one command (after run_more.sh): does MaskedKD recover when it reads a better student layer?
#   0) pick the layer L* on the VALIDATION split (Full-KD students, rule fixed in stage0/select_layer.py)  ~5-10 min
#   1) train (P at a time, priority order): mk_L* 0.15 | mk_L* 0.1 | mk_L* 0.05 | oracle_bird 0.1 | oracle_bird 0.05 |
#      maskedkd 0.1  (x seeds 0 1 2 = 18 runs)  [+ OPTIONAL=1: MaskedKD without head 2 at 0.15]
#   2) part-C analysis of all runs + summary with verdicts R1-R7 -> results/more.md, why_more/why_more_C_2.md
#
#   export DATA_ROOT=$HOME/stage0/data OUTPUT_ROOT=$HOME/stage0/outputs HF_HOME=$HOME/stage0/hf GPUS=0
#   setsid nohup ./scripts/run_layer.sh > $HOME/layer.log 2>&1 < /dev/null &
set -euo pipefail
cd "$(dirname "$0")/.."
: "${OUTPUT_ROOT:?set OUTPUT_ROOT}"; : "${DATA_ROOT:?set DATA_ROOT}"
[[ -n "${GPUS:-}" && "$GPUS" != cpu ]] && export CUDA_VISIBLE_DEVICES=${GPUS%%,*}
RES=${RESULTS_DIR:-results}
echo; echo "======== [$(date '+%F %T')] 0 pick the layer on val ========"
uv run python -m stage0.select_layer ${SELECT_ARGS:-}
L=$(uv run python -c "import json,os; print(json.load(open(os.path.join('$RES','why_more','layer_select.json')))['layer'])")
echo "L* = $L"
JOBS="mk_L$L:0.15 mk_L$L:0.1 mk_L$L:0.05 oracle_bird:0.1 oracle_bird:0.05 maskedkd:0.1"
[[ "${OPTIONAL:-0}" == 1 ]] && JOBS="$JOBS mk_dh2:0.15"
export JOBS SKIP_AB=1 C_SUFFIX=_2
exec ./scripts/run_more.sh
