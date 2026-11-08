#!/usr/bin/env bash
# wait until the decay-off runs have started and every suite runner has exited, then run the closing plan's seed-42
# suite, 3 jobs at a time on gpu 0. the process pattern is written as a regex so this script never matches itself.
set -euo pipefail
cd "$(dirname "$0")/.."
until [ -f runs/decay-off-s41.log ] && ! pgrep -f "suite[.]py" >/dev/null; do sleep 60; done
echo "starting closing_s42 at $(date)"
uv run python src/suite.py --suite closing_s42 --devices 0,0,0 > runs/closing-s42.log 2>&1
echo "closing_s42 exited at $(date) with $?"
