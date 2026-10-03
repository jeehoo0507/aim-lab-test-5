#!/usr/bin/env bash
# Progress table for run_stage0.sh / smoke_test.sh:  OUTPUT_ROOT=... ./status.sh [--all]
set -euo pipefail
cd "$(dirname "$0")"
: "${OUTPUT_ROOT:?set OUTPUT_ROOT}"
exec uv run python -m stage0.status "$@"
