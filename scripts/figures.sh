#!/usr/bin/env bash
# regenerate every figure in the write-up from the committed data in results/: training dynamics into
# OUT_DIR/dynamics, the mechanism figures and seed table into OUT_DIR/mechanism.
#
#   scripts/figures.sh [OUT_DIR]     (default figures/out)
#
# needs no gpu and no training runs; uv installs the environment on first use.
set -euo pipefail
cd "$(dirname "$0")/.."
out=${1:-figures/out}
uv run python figures/ablations.py --out "$out/dynamics"
uv run python figures/collapse.py --out "$out/dynamics"
uv run python figures/mechanism.py --out "$out/mechanism"
