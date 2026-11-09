#!/usr/bin/env bash
# retrain the write-up's runs on a rented gpu machine over ssh: by default the 20 runs of the seed-41 sweep, which
# ran on 8x rtx 5090 (about 3 hours). experiments/remote/vast.sh ships the committed code and the data, smoke-tests,
# launches, and streams runs/ back into vast-runs/ until training ends, checkpoints included. afterwards,
# scripts/process.sh turns them into results/ and the figures.
#
#   scripts/train-vast.sh HOST PORT [--gpu 5090|4090|all] [--dry-run]
#
# --gpu picks the runs by the gpu they were trained on (see scripts/train-local.sh); the branch arms aren't
# included, since they run locally from a committed checkpoint. --dry-run lists the runs and contacts nothing.
# vast.sh ships HEAD, so commit first (it refuses uncommitted changes to tracked files). setting up an instance:
# experiments/remote/vast.md.
#
# runs reproduce exactly only on the gpu model they were trained on; elsewhere, which runs grok reproduces, but
# grokking time can differ by up to 2x.
set -euo pipefail
cd "$(dirname "$0")/.."

[ $# -ge 2 ] || { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit 2; }
host=$1 port=$2
shift 2
gpu=5090 dry=0
while [ $# -gt 0 ]; do
  case $1 in
    --gpu) gpu=$2; shift 2 ;;
    --dry-run) dry=1; shift ;;
    *) echo "error: unknown option $1" >&2; exit 2 ;;
  esac
done
case $gpu in
  4090|5090) runs=$(uv run python scripts/export.py --list "$gpu") ;;
  all) runs="$(uv run python scripts/export.py --list 5090) $(uv run python scripts/export.py --list 4090)" ;;
  *) echo "error: --gpu takes 5090, 4090 or all" >&2; exit 2 ;;
esac

echo "== $(wc -w <<<"$runs") runs for $host:$port"
if [ "$dry" = 1 ]; then
  echo "$runs"
  exit 0
fi
[ -f data/synthetic/p113_h3_x6/meta.json ] || uv run python src/make_data.py --name p113_h3_x6
uv run python src/verify_data.py data/synthetic/p113_h3_x6
# shellcheck disable=SC2086  # one run name per word
experiments/remote/vast.sh up "$host" "$port" --table 10 --runs $runs --until 20000
