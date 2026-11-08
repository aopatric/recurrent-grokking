#!/usr/bin/env bash
# run the shared vs unshared benchmark on the local gpu: the sandwich block looped 8x against 8 distinct sandwich
# layers, each under nanda's constant lr and the fixed recipe (tail lr decay + clipping), 4 runs sharing gpu 0.
#
#   scripts/shared-vs-unshared.sh [--seed 42] [--with-parcae] [extra suite.py args...]
#
# seed 41 by default (suite shared_vs_unshared); --seed 42 runs shared_vs_unshared_s42. --with-parcae queues the
# parcae suite for the same seed after it. runs land in runs/<name>, logs in runs/<name>.log; ctrl-c pauses every
# job and rerunning the same command resumes. read the result with:
#   uv run python src/verdicts.py runs/t5_*
set -euo pipefail

here=$(cd "$(dirname "$0")/.." && pwd)
cd "$here"
suites=(shared_vs_unshared) parcae=(parcae)
extra=()
while [ $# -gt 0 ]; do
  case $1 in
    --seed) [ "$2" = 41 ] || { suites=("shared_vs_unshared_s$2"); parcae=("parcae_s$2"); }; shift 2 ;;
    --with-parcae) want_parcae=1; shift ;;
    *) extra+=("$1"); shift ;;
  esac
done
if [ -n "${want_parcae:-}" ]; then suites+=("${parcae[@]}"); fi

exec uv run python src/suite.py --suite "${suites[@]}" --devices 0,0,0,0 "${extra[@]}"
