#!/usr/bin/env bash
# retrain the write-up's runs on this machine's gpus: by default the 15 that ran on an rtx 4090, then the nine
# branch arms from the committed step-8944 checkpoint. runs land in runs/ and runs/branches/; ctrl-c pauses every
# job, and rerunning the same command resumes. afterwards, scripts/process.sh turns them into results/ and
# the figures.
#
#   scripts/train-local.sh [--gpu 4090|5090|all] [--devices 0,0,0] [--dry-run]
#
# --gpu picks the runs by the gpu they were trained on (each run's meta.json records it): 4090 (default) is every
# run after the seed-41 sweep, plus the branch arms; 5090 is the seed-41 sweep, which scripts/train-vast.sh runs
# on rented gpus instead; all is both. --devices goes to src/suite.py (default 0,0,0: three runs share gpu 0, about
# 3-3.5 hours each on an rtx 4090). --dry-run lists what would run and trains nothing.
#
# runs reproduce exactly only on the gpu model they were trained on; elsewhere, which runs grok reproduces, but
# grokking time can differ by up to 2x.
set -euo pipefail
cd "$(dirname "$0")/.."

gpu=4090 devices=0,0,0 dry=()
while [ $# -gt 0 ]; do
  case $1 in
    --gpu) gpu=$2; shift 2 ;;
    --devices) devices=$2; shift 2 ;;
    --dry-run) dry=(--dry-run); shift ;;
    *) echo "error: unknown option $1" >&2; exit 2 ;;
  esac
done
case $gpu in
  4090|5090) groups=("$gpu") ;;
  all) groups=(5090 4090) ;;
  *) echo "error: --gpu takes 4090, 5090 or all" >&2; exit 2 ;;
esac

if [ ${#dry[@]} = 0 ]; then
  [ -f data/synthetic/p113_h3_x6/meta.json ] || uv run python src/make_data.py --name p113_h3_x6
  uv run python src/verify_data.py data/synthetic/p113_h3_x6
fi
for g in "${groups[@]}"; do
  runs=$(uv run python scripts/export.py --list "$g")
  echo "== $(wc -w <<<"$runs") runs that trained on an rtx $g"
  # shellcheck disable=SC2086  # one run name per word
  uv run python src/suite.py --runs $runs --devices "$devices" --until 20000 "${dry[@]}"
done
if [[ " ${groups[*]} " == *" 4090 "* ]]; then
  echo "== the nine branch arms"
  DRY_RUN=$([ ${#dry[@]} = 0 ] && echo 0 || echo 1) experiments/stability/branch-arms.sh
fi
