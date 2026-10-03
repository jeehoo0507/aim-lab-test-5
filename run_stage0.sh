#!/usr/bin/env bash
# Stage 0 full pipeline: lr sweeps -> teacher / CE / KD runs -> diagnosis -> eval -> summary.
#
#   GPUS="0,1" [RUNS_PER_GPU=4] DATA_ROOT=... OUTPUT_ROOT=... HF_HOME=... ./run_stage0.sh
#
# GPUS is required (shared server: nothing is grabbed by default). RUNS_PER_GPU defaults to the value
# preflight.py measured (outputs/preflight.json). Re-running the same command resumes: finished runs
# (DONE) are skipped, interrupted ones continue from last.pt. Extra args go to stage0/scheduler.py,
# e.g. --fallback-alpha1 (only after the team confirms), --datasets cub.
# Progress: ./status.sh      Logs: $OUTPUT_ROOT/{dataset}/{mode}/seed{seed}/train.log
set -euo pipefail
cd "$(dirname "$0")"

: "${GPUS:?set GPUS, e.g. GPUS=\"0,1\" (comma-separated ids to use)}"
: "${DATA_ROOT:?set DATA_ROOT}"
: "${OUTPUT_ROOT:?set OUTPUT_ROOT}"
[[ -n "${HF_HOME:-}" ]] || echo "[warn] HF_HOME not set; timm weights go to ~/.cache/huggingface"

if [[ -z "${TMUX:-}" && -z "${STY:-}" && -t 1 && -z "${STAGE0_NO_TMUX_HINT:-}" ]]; then
  cat <<MSG
[hint] This runs for many hours. Start it inside tmux or nohup so it survives a dropped SSH session:
  tmux new -s stage0 'GPUS=$GPUS DATA_ROOT=$DATA_ROOT OUTPUT_ROOT=$OUTPUT_ROOT ./run_stage0.sh'
  nohup env GPUS=$GPUS ./run_stage0.sh > \$OUTPUT_ROOT/run_stage0.log 2>&1 &
Continuing in 10 s (Ctrl-C to abort)...
MSG
  sleep 10
fi

[[ -f "$OUTPUT_ROOT/preflight.json" ]] || echo "[warn] $OUTPUT_ROOT/preflight.json missing: run 'uv run python -m stage0.preflight' first"
mkdir -p "$OUTPUT_ROOT"
exec uv run python -m stage0.scheduler "$@"
