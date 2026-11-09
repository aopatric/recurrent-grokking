#!/usr/bin/env bash
# after training: run the mechanism analyses on the trained runs, export what the figures read into results/, and
# draw the figures with their notes. `git diff --stat results/` then shows what differs from the committed data.
#
#   scripts/process.sh [--mech-root DIR] [--out DIR] [--figures DIR] [SOURCE ...]
#
# SOURCE: folders holding the trained runs (default vast-runs runs runs/branches, where the training scripts put
# them); each run must be in exactly one. --mech-root: where the analyses write their mech-* folders (default
# runs). --out: where the export writes runs/ and mechanism/ (default results, replacing them). --figures: where the
# figures go (default figures/out). takes about 15 minutes on an rtx 4090.
#
# some analyses read fixed checkpoint steps (the --at lists below), chosen from the committed runs' timing; on
# another gpu model a retrained run reaches those steps at a different point in training.
set -euo pipefail
cd "$(dirname "$0")/.."

mech=runs out=results figs=figures/out sources=()
while [ $# -gt 0 ]; do
  case $1 in
    --mech-root) mech=$2; shift 2 ;;
    --out) out=$2; shift 2 ;;
    --figures) figs=$2; shift 2 ;;
    -*) echo "error: unknown option $1" >&2; exit 2 ;;
    *) sources+=("$1"); shift ;;
  esac
done
[ ${#sources[@]} -gt 0 ] || sources=(vast-runs runs runs/branches)

# full run names of configs at seeds: names "41 42" CONFIG...
names() { local c s; for c in "${@:2}"; do for s in $1; do echo "${c}_p113_h3_x6_s$s"; done; done; }
# the directory of each named run, from the first source holding it
dirs() {
  local name src
  for name; do
    for src in "${sources[@]}"; do
      if [ -f "$src/$name/metrics.jsonl" ]; then echo "$src/$name"; continue 2; fi
    done
    echo "error: $name isn't in ${sources[*]}" >&2
    return 1
  done
}
mech_py() { uv run python "experiments/mechanism/$1.py" "${@:2}"; }

twophase=(t6_huginn_twophase t6_ut_statenorm_twophase t6_ut_sandwich_twophase)
# every run the analyses read, found before anything runs
tp41=$(dirs $(names 41 "${twophase[@]}"))
tp42=$(dirs $(names 42 "${twophase[@]}"))
anchor=$(dirs $(names "41 42" t1_anchor_nope_ref))
fixed=$(dirs $(names "41 42" t1_huginn_fixedT8))
huginn_tp=$(dirs $(names "41 42" t6_huginn_twophase))
ref=$(dirs $(names "41 42" t1_huginn_ref))
ref42=$(dirs $(names 42 t1_huginn_ref))
variants=$(dirs $(names 41 t1_huginn_embedstart t2_huginn_fullbackprop t1_huginn_mindepth2))
sandwich_tp=$(dirs $(names "41 42" t6_ut_sandwich_twophase))
statenorm41=$(dirs $(names 41 t6_ut_statenorm_twophase))
random_depth=$(dirs $(names 41 t2_ut_statenorm_ref t2_ut_statenorm_sampled))
ref41=$(dirs $(names 41 t1_huginn_ref))

# shellcheck disable=SC2086  # one run directory per word
{
  echo "== mechanism analyses into $mech/mech-*"
  mech_py passes $tp41 --out "$mech/mech-passes"
  mech_py passes $tp42 --out "$mech/mech-passes-s42"
  mech_py past-depth $tp41 --out "$mech/mech-long"
  mech_py past-depth $tp42 --out "$mech/mech-long-s42"
  mech_py past-depth $fixed --out "$mech/mech-long-fixed"
  mech_py frequencies $anchor $tp41 $tp42 --out "$mech/mech-freq"
  mech_py state-use $huginn_tp --at 6632,10313,12861,16038,20000 --inject --out "$mech/mech-state-twophase"
  mech_py state-use $fixed --at 6968,20000 --inject --out "$mech/mech-state-fixed"
  mech_py state-use $ref $variants --at best --out "$mech/mech-state-variants"
  mech_py state-use $ref42 --at 14736,18915,20000 --inject --out "$mech/mech-state-ref-late"
  mech_py internals $anchor $huginn_tp $sandwich_tp $statenorm41 --freqs "$mech/mech-freq" --out "$mech/mech-internals"
  mech_py past-depth $random_depth --out "$mech/mech-long-random-depth"
  mech_py past-depth $random_depth --at 18915 --out "$mech/mech-long-random-depth-18915"
  mech_py past-depth $ref41 --at 6968 --out "$mech/mech-long-ref-6968"
  mech_py state-use $ref41 --at 6968 --inject --out "$mech/mech-state-ref-6968"
}

echo "== export into $out"
uv run python scripts/export.py --sources "${sources[@]}" --mech-root "$mech" --out "$out"

echo "== figures into $figs"
uv run python figures/ablations.py --root "$out/runs" --out "$figs/dynamics"
uv run python figures/collapse.py --root "$out/runs" --out "$figs/dynamics"
uv run python figures/mechanism.py --root "$out/runs" --mech "$out/mechanism" --out "$figs/mechanism"
echo "== done; compare with the committed data: git diff --stat $out/"
