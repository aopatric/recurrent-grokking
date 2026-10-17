#!/usr/bin/env bash
# run seed 41 of every config in all three tiers on a rented gpu machine, stopping each run at step 20000,
# and stream the results back with a live verdict table.
#
#   scripts/single-seed-three-suites.sh HOST PORT [--into DIR] [--every SECONDS] [--table N] [--no-smoke]
#
# this is scripts/vast.sh up with the run list seed41_all_tiers from configs/suites.yaml. results stream into
# runs/vast by default (view live with: uv run tensorboard --logdir runs/vast), the verdict table is printed
# every 10 rounds, and once nothing is running a final copy brings the checkpoints.
set -euo pipefail

if [ $# -lt 2 ]; then
  awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"
  exit 2
fi
host=$1 port=$2
shift 2

exec "$(dirname "$0")/vast.sh" up "$host" "$port" --table 10 "$@" --suite seed41_all_tiers --until 20000
