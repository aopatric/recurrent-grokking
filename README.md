# recurrent grokking

Looped (recurrent-depth) transformers trained on n-hop modular addition mod 113, compared with Nanda et al.'s one-layer model: when they grok, why grokked models collapse, and what the loop computes once they do. The write-up is [On Serial Depth and Monitorability in Looped Transformers](https://aopatric.github.io/blog/2026/recurrent-grokking/).

## Setup

```bash
git clone --depth 1 https://github.com/aopatric/recurrent-grokking   # --depth 1 skips old history
cd recurrent-grokking
uv sync
```

Python 3.12 and [uv](https://docs.astral.sh/uv/). `uv run pytest` runs the tests.

## Reproduce the figures

```bash
scripts/figures.sh      # writes figures/out/dynamics/ and figures/out/mechanism/ in a few seconds; no gpu, no training
```

The figures read only the committed data in `results/`.

| figure | script | runs |
|---|---|---|
| `grokking_outcomes.png`, `.md` | `figures/ablations.py` | the 28 runs on Nanda's recipe for the full 20k steps listed in the script, one row each: the seed-41 ablation sweep, the seed-42 runs and the `t5_*_const` runs; the `.md` adds the instability numbers for the 17 that grokked, with the step logs of the `t5_*_const` runs and `t1_huginn_ref` (seed 41) before grokking |
| `fixes.png`, `.md` | `figures/collapse.py` | nine branches from `t1_huginn_ref`'s step-8944 checkpoint (seed 41); three pairs split at step 14000 into Nanda's recipe and the two-phase recipe, each identical until the switch: `t5_ut_sandwich_const` vs `t6_ut_sandwich_twophase` (seeds 42 and 41) and `t1_huginn_ref` vs `t6_huginn_twophase` (seed 42); the `.md` adds every `t6_*_twophase` run at both seeds |
| `attention.png`, `attention.md` | `figures/mechanism.py` | final checkpoints of the one-layer model and the three `t6_*_twophase` runs at seed 41; the notes in `attention.md` cover seed 42 too |
| `clocks.png`, `clocks.md` | `figures/mechanism.py` | final checkpoints of the one-layer model and the `t6_*_twophase` runs, seeds 41 and 42 (the state-norm loop at 41 only) |
| `compute_early.png`, `compute_early.md` | `figures/mechanism.py` | final checkpoints of the `t6_*_twophase` runs, seeds 41 and 42 (the state-norm loop at 41 only); the `.md` adds the one-layer model, every pass, and the probe inside the Huginn-style loop |
| `mechanism_seeds.md` | `figures/mechanism.py` | the mechanism results at seeds 41 and 42 side by side, with their mean where both runs grokked |

`figures/ablations.py --supporting` also draws the four 2×3 ablation figures behind the outcomes figure (`looping`, `normalization`, `injection`, `weight_decay`; add `--smooth none` for no rolling median), and `figures/collapse.py --supporting` the `instability` figure (only the runs that grokked) and the older `collapse` and `two_phase` figures, into `figures/out/dynamics/supporting/`. `figures/mechanism.py --supporting` also draws mechanism figures that aren't in the write-up (`passes`, `past_depth`, `frequencies`, `state_use`, `dilution`, `decoding`) into `figures/out/mechanism/supporting/`.

## What's in `results/`

- `results/runs/<run>/`: `config.yaml`, `meta.json` and `metrics.jsonl` exactly as training wrote them. Runs with step-level figures also have `steps.csv` (step, depth, loss, grad norm), cut from the full per-step log. Runs the analyses read keep those checkpoints (33 files, about 66 MB), and `t1_huginn_ref` (seed 41) keeps its step-8944 checkpoint, the branch source. `results/` is about 100 MB in all.
- `results/mechanism/`: the outputs of `experiments/mechanism/`, one folder per analysis (`-s42` marks the seed-42 runs of an analysis first done at seed 41). Every run they analyse is in `results/runs/` with the checkpoints they read.

`scripts/export.py` rebuilds `results/` from local run folders and checks every value it writes against the source.

## Re-run the analyses

The mechanism analyses need the dataset, which is generated, not committed:

```bash
uv run python src/make_data.py --name p113_h3_x6
uv run python src/verify_data.py data/synthetic/p113_h3_x6
r=results/runs
uv run python experiments/mechanism/passes.py $r/t6_*_twophase_p113_h3_x6_s41 --out runs/mech-passes
uv run python experiments/mechanism/passes.py $r/t6_*_twophase_p113_h3_x6_s42 --out runs/mech-passes-s42
uv run python experiments/mechanism/past-depth.py $r/t6_*_twophase_p113_h3_x6_s41 --out runs/mech-long
uv run python experiments/mechanism/past-depth.py $r/t6_*_twophase_p113_h3_x6_s42 --out runs/mech-long-s42
uv run python experiments/mechanism/past-depth.py $r/t1_huginn_fixedT8_p113_h3_x6_s4[12] --out runs/mech-long-fixed
uv run python experiments/mechanism/frequencies.py $r/t1_anchor_nope_ref_p113_h3_x6_s4[12] $r/t6_*_twophase_p113_h3_x6_s4[12] --out runs/mech-freq
uv run python experiments/mechanism/state-use.py $r/t6_huginn_twophase_p113_h3_x6_s4[12] --at 6632,10313,12861,16038,20000 --inject --out runs/mech-state-twophase
uv run python experiments/mechanism/state-use.py $r/t1_huginn_fixedT8_p113_h3_x6_s4[12] --at 6968,20000 --inject --out runs/mech-state-fixed
uv run python experiments/mechanism/state-use.py $r/t1_huginn_ref_p113_h3_x6_s4[12] $r/t1_huginn_embedstart_p113_h3_x6_s41 $r/t2_huginn_fullbackprop_p113_h3_x6_s41 $r/t1_huginn_mindepth2_p113_h3_x6_s41 --at best --out runs/mech-state-variants
uv run python experiments/mechanism/state-use.py $r/t1_huginn_ref_p113_h3_x6_s42 --at 14736,18915,20000 --inject --out runs/mech-state-ref-late
uv run python experiments/mechanism/internals.py $r/t1_anchor_nope_ref_p113_h3_x6_s4[12] $r/t6_huginn_twophase_p113_h3_x6_s4[12] $r/t6_ut_sandwich_twophase_p113_h3_x6_s4[12] $r/t6_ut_statenorm_twophase_p113_h3_x6_s41 --freqs runs/mech-freq --out runs/mech-internals
uv run python experiments/mechanism/past-depth.py $r/t2_ut_statenorm_ref_p113_h3_x6_s41 $r/t2_ut_statenorm_sampled_p113_h3_x6_s41 --out runs/mech-long-random-depth
uv run python experiments/mechanism/past-depth.py $r/t2_ut_statenorm_ref_p113_h3_x6_s41 $r/t2_ut_statenorm_sampled_p113_h3_x6_s41 --at 18915 --out runs/mech-long-random-depth-18915
uv run python experiments/mechanism/past-depth.py $r/t1_huginn_ref_p113_h3_x6_s41 --at 6968 --out runs/mech-long-ref-6968
uv run python experiments/mechanism/state-use.py $r/t1_huginn_ref_p113_h3_x6_s41 --at 6968 --inject --out runs/mech-state-ref-6968
```

The analyses need no gpu, but run faster on one; `results/` holds only the checkpoints they read, so `--at best` picks the same checkpoint from it. On an RTX 4090 these commands reproduce every number in `results/mechanism/` exactly (checked 2026-11-09, from a fresh shallow clone); `past-depth` also records the paths it was given, which differ.

Each run's `meta.json` records the sha256 of every data file it trained on; `src/analysis.load_run` lists any file that differs as `data_changed`. Regenerated data matches those hashes; only the dataset's own `meta.json` differs, in the commit it records.

## Retrain from scratch

```bash
scripts/train-vast.sh HOST PORT    # the seed-41 sweep: 20 runs on a rented 8× RTX 5090, about 3 hours, streamed into vast-runs/
scripts/train-local.sh             # the 15 later runs on an RTX 4090 (about 17 hours, three at a time), then the nine branch arms
scripts/process.sh                 # the analyses, the export into results/ and the figures, about 15 minutes
git diff --stat results/           # what differs from the committed data
```

Both training scripts take `--dry-run` to list what they would run, and `--gpu 5090|4090|all` to move a group to the other machine; both generate and check the dataset first if it isn't there. `train-vast.sh` ships the committed code, so commit any changes first; [`experiments/remote/vast.md`](experiments/remote/vast.md) covers renting an instance. A rerun resumes unfinished runs and skips finished ones; a branch arm that stopped partway has to be deleted first.

**What reproduces exactly.** Runs are bit-identical on the same GPU model. The seed-41 sweep ran on RTX 5090s and every other run on an RTX 4090; each run's `meta.json` records its GPU. On other GPUs grokking time can differ by up to 2×, so expect the same runs to grok but not the same curves, and the analyses that read fixed checkpoint steps (the `--at` lists) to land at other points in training. The branch arms start from the committed step-8944 checkpoint, so they reproduce on an RTX 4090 without retraining the sweep.

**How each group first ran.** Runs are defined in `configs/train.yaml` and grouped into suites in `configs/suites.yaml`. A run trains with `uv run python src/train.py --name <run>`, a list of runs or a suite with `uv run python src/suite.py --runs <run> ... | --suite <suite> --devices 0` (repeat a device to share it, e.g. `--devices 0,0,0`). The scripts above take their run lists from `scripts/export.py --list 5090|4090`. The `t7_*` and `t8_*` runs and the margin check aren't in the figures.

| runs | command | where they ran |
|---|---|---|
| seed-41 ablation sweep (`t1_*`, `t2_*`, `t3_*`) | `experiments/remote/vast.sh up HOST PORT --table 10 --suite seed41_all_tiers --until 20000` | 8× RTX 5090 (see `experiments/remote/vast.md`) |
| shared vs unshared (`t5_*`) | `experiments/stability/shared-vs-unshared.sh [--seed 42]` | RTX 4090 |
| two-phase recipe (`t6_*`) | suite `two_phase` | RTX 4090 |
| injection without the rest of huginn (`t7_*`) | suite `inject_fixedpoint` | RTX 4090 |
| weight decay off after 14k (`t8_*`) | suite `decay_off` | RTX 4090 |
| seed 42 of the main comparisons | suite `closing_s42` | RTX 4090 |
| seed 42 of the one-layer model, fixed-depth Huginn and the Huginn reference, for the mechanism analyses | `uv run python src/suite.py --runs t1_anchor_nope_ref_p113_h3_x6_s42 t1_huginn_fixedT8_p113_h3_x6_s42 t1_huginn_ref_p113_h3_x6_s42 --devices 0,0,0 --until 20000` | RTX 4090 (their seed-41 runs were on the RTX 5090s) |
| branches from huginn's step 8944 | `uv run python src/branch.py --source results/runs/t1_huginn_ref_p113_h3_x6_s41 --step 8944 --arm <arm>`; `experiments/stability/branch-arms.sh` runs all nine, three at a time | RTX 4090 |
| softmax-collapse margin check | `uv run python experiments/stability/check-margins.py <run dirs>` | RTX 4090 |

**Code versions.** Each `meta.json` records the commit it ran from. The ablation sweep ran from `4f6a68d`. The runs from 2026-10-20 on record `f23708c` with uncommitted changes. Commit `d295455` is the final state of those changes; on the same RTX 4090 it reproduces exactly the first 300 steps of `t6_ut_sandwich_twophase` and of all nine branch arms (rechecked 2026-11-08), and so does a fresh shallow clone of the current code for `t6_ut_sandwich_twophase` (2026-11-09). Nothing after the two-phase switch at step 14000 has been re-checked. The three seed-42 runs from 2026-11-08 record `d295455` with uncommitted changes; those changes moved scripts and figures and left `src/` and `configs/` untouched.

**Names.** The write-up's "plain loop" is `ut` in run and config names (one shared layer looped, without the universal transformer's timestep embeddings or halting). "State-norm" and "sandwich" name where the RMS norms sit.

## Layout

```
configs/       run, suite and dataset definitions
src/           models, data, training, branching, analysis helpers
scripts/       what you run: train-local.sh, train-vast.sh, process.sh, export.py, figures.sh
experiments/   what the scripts call: mechanism/ (the analyses), stability/, remote/ (rented gpus)
figures/       figure code; output goes to figures/out/dynamics/ (training) and figures/out/mechanism/
results/       committed data the figures read
tests/
```

Code is MIT licensed. Correspondence to [aopatric@mit.edu](mailto:aopatric@mit.edu).
