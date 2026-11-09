#!/usr/bin/env bash
# branch one grokked checkpoint into the nine optimizer arms of src/branch.py, a few at a time on one gpu, each
# logging to runs/branches/<run>_from<step>_<arm>.log.
#
#   experiments/stability/branch-arms.sh [SOURCE] [STEP] [extra branch.py args...]
#
# defaults to the committed checkpoint of huginn's reference run at step 8944, inside a grokked stretch (val acc 1.0
# at 8800 and 9000). PARALLEL (default 3) arms share the gpu at once; nine at once don't fit in 24 GB. an arm that
# already finished is skipped; a partial one stops the script, since src/branch.py won't overwrite it.
# DRY_RUN=1 prints what would run and exits.
set -euo pipefail

here=$(cd "$(dirname "$0")/../.." && pwd)
cd "$here"
source=${1:-results/runs/t1_huginn_ref_p113_h3_x6_s41}
step=${2:-8944}
shift $(( $# < 2 ? $# : 2 ))
parallel=${PARALLEL:-3}
arms=(control wd0 wd05 wd2 wd2_lr05 nowd_embed clip lrdecay clip_lrdecay)

# 0 if an arm's run directory holds the checkpoint at its last step, which src/branch.py saves when it finishes
finished() {
  [ -f "$1/meta.json" ] || return 1
  local n
  n=$(uv run python -c 'import json, sys; print(json.load(open(sys.argv[1]))["branch"]["steps"])' "$1/meta.json")
  [ -f "$1/checkpoints/$(printf 'step_%07d.pt' "$n")" ]
}

todo=()
for arm in "${arms[@]}"; do
  dir="runs/branches/$(basename "$source")_from${step}_${arm}"
  if finished "$dir"; then
    echo "skip $arm: $dir already finished"
  elif [ -e "$dir" ]; then
    echo "error: $dir exists but didn't finish; delete it to rerun" >&2
    exit 1
  else
    todo+=("$arm")
  fi
done
if [ "${DRY_RUN:-0}" = 1 ]; then
  echo "would run ${#todo[@]} arms from $source at step $step, $parallel at a time: ${todo[*]}"
  exit 0
fi

mkdir -p runs/branches
status=0
for ((i = 0; i < ${#todo[@]}; i += parallel)); do
  pids=()
  for arm in "${todo[@]:i:parallel}"; do
    log="runs/branches/$(basename "$source")_from${step}_${arm}.log"
    uv run python src/branch.py --source "$source" --step "$step" --arm "$arm" "$@" > "$log" 2>&1 &
    pids+=($!)
    echo "started $arm (pid $!), logging to $log"
  done
  for pid in "${pids[@]}"; do wait "$pid" || status=1; done
done
exit $status
