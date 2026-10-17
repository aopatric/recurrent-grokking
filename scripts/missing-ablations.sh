#!/usr/bin/env bash
# run the follow-up ablations on a rented gpu machine, stopping each run at step 20000, and stream the results
# back into the same folder as the earlier runs, with a live verdict table.
#
#   scripts/missing-ablations.sh HOST PORT [--into DIR] [--every SECONDS] [--table N] [--no-smoke] [--until N]
#
# this is scripts/vast.sh up with the run list missing_ablations from configs/suites.yaml: seeds 42 and 43 of
# the state-norm ut and of huginn with a state norm in place of the sandwich, plus four seed-41 runs that each
# add one huginn piece to the state-norm ut with sampled depth. that is 8 runs, one round on 8 gpus. results
# stream into vast-runs/ at the repo root by default, next to the runs already there; the runner log already
# there is kept as suite-<time>.log, since the new instance's log replaces it. --until 40000 runs them longer.
set -euo pipefail

if [ $# -lt 2 ]; then
  awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"
  exit 2
fi
host=$1 port=$2
shift 2

here=$(cd "$(dirname "$0")/.." && pwd)
if [ -f "$here/vast-runs/suite.log" ]; then
  cp "$here/vast-runs/suite.log" "$here/vast-runs/suite-$(date +%Y%m%d-%H%M%S).log"
fi

exec "$here/scripts/vast.sh" up "$host" "$port" --table 10 --suite missing_ablations --until 20000 "$@"
