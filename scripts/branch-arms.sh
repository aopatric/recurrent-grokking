#!/usr/bin/env bash
# branch one grokked checkpoint into the four optimizer arms of src/branch.py and run them side by side on
# one gpu, each logging to runs/branches/<run>_from<step>_<arm>.log.
#
#   scripts/branch-arms.sh [SOURCE] [STEP] [extra branch.py args...]
#
# defaults to huginn's reference run at step 8944, inside a grokked stretch (val acc 1.0 at 8800 and 9000).
set -euo pipefail

here=$(cd "$(dirname "$0")/.." && pwd)
cd "$here"
source=${1:-vast-runs/t1_huginn_ref_p113_h3_x6_s41}
step=${2:-8944}
shift $(( $# < 2 ? $# : 2 ))

mkdir -p runs/branches
pids=()
for arm in control lrdecay nowd_embed clip; do
  log="runs/branches/$(basename "$source")_from${step}_${arm}.log"
  uv run python src/branch.py --source "$source" --step "$step" --arm "$arm" "$@" > "$log" 2>&1 &
  pids+=($!)
  echo "started $arm (pid $!), logging to $log"
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit $status
