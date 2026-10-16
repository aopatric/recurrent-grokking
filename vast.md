# Running the experiment suite on vast.ai

This is a walkthrough for running the ablation suite on a rented GPU machine and streaming the results back to your own computer while it trains. Every command runs **on your machine, from the repo root**. `scripts/vast.sh` drives the instance over ssh, so nothing is installed or edited on the instance by hand.

The short version, once an instance is rented:

```bash
export H=ssh5.vast.ai P=41234                       # from the instance's "Connect" button
scripts/vast.sh up $H $P --into runs/vast --suite tier1 tier2 tier3 --until 20000
uv run tensorboard --logdir runs/vast               # in a second terminal: live charts
```

The rest of this page explains each piece.

---

## 0. How the pieces fit

| piece | what it does |
|---|---|
| `configs/train.yaml` | every run: model, depth rule, optimizer, seed. Suite runs are named `<tier>_<model>_<change>_p113_h3_x6_s<seed>`, at seeds 41, 42, 43 |
| `configs/suites.yaml` | named run lists: `tier1` (where grokking breaks and comes back), `tier2` (follow-ups and regularization), `tier3` (optional) |
| `src/train.py` | trains one run into `runs/<name>/`. It is resumable and logs metrics, per-step stats and TensorBoard events |
| `src/suite.py` | the runner: keeps one run per GPU, resumes stopped runs, pauses everything cleanly on a signal |
| `src/smoke.py` | trains each distinct config for a few steps; reports GPU memory and time per 20k steps |
| `src/verdicts.py` | turns run folders into the results table (groks / stable / slow / fails / diverged), per run or per config |
| `scripts/vast.sh` | runs all of the above on the instance and copies `runs/` back |

A run folder contains:
- `metrics.jsonl`: evals every 100 steps, and depth sweeps;
- `steps.jsonl`: per-step loss, depth and weight stats;
- `tb/`: TensorBoard events;
- `checkpoints/`;
- `latest.pt`: the resume point;
- `config.yaml` and `meta.json`.

The run is the same whether it trained locally or on vast.

## 1. One-time setup

1. Create a vast.ai account and add credit.
2. Add your public ssh key under **Account → Keys**. Use `cat ~/.ssh/id_ed25519.pub`, or create a key first with `ssh-keygen -t ed25519`.
3. Locally you need `git`, `ssh`, `rsync` (3.1 or newer) and `sha256sum`, plus the project environment (`uv sync`).

## 2. Before each session (on your machine, no GPU needed)

```bash
uv run pytest -q                                                         # about 30 s on CPU
uv run python src/suite.py --suite tier1 tier2 tier3 --until 20000 --dry-run --devices 0
git status                                                               # commit everything you want shipped
```

- The dry run lists what the runner would do with each run: start, resume, or skip and why.
- **Only committed code is shipped.** `push` refuses if tracked files have uncommitted changes.

## 3. Rent an instance

On the **Search** page:
- **GPU:** RTX 4090 or 3090; 8 GPUs is a good size. The models are tiny, so A100s and H100s buy nothing.
- **CUDA:** filter for **max CUDA ≥ 13.0**. The lockfile pins a CUDA 13 build of torch.
- **RAM:** at least 32 GB, since each job holds a few GB of host memory.
- **Disk:** at least 30 GB. All 72 suite runs come to roughly 8 GB.
- **Image:** any Ubuntu or CUDA image with ssh launch, e.g. `vastai/base-image` or a `pytorch/pytorch` image.

**Expected time on 8×4090**, from a smoke test on a 4090, judged at 20k steps:

| suites | runs | wall time |
|---|---|---|
| tier 1 | 33 | ~4.5 h |
| tiers 1+2 | 63 | ~8.5 h |
| all tiers | 72 | ~9 h |

Allow about 25% more.

When it's running, click **Connect**. For `ssh -p 41234 root@ssh5.vast.ai`, HOST is `ssh5.vast.ai` and PORT is `41234`:

```bash
export H=ssh5.vast.ai P=41234
```

## 4. Run the whole suite and stream it back

```bash
scripts/vast.sh up $H $P --into runs/vast --suite tier1 tier2 tier3 --until 20000
```

`up` does four things in order. It stops at the first one that fails.

1. **push**:
   - ships the committed HEAD as a git bundle (so every run records its commit) and `data/synthetic`, with checksums;
   - installs uv and the locked environment;
   - checks that torch sees the GPUs.
2. **smoke**: trains each distinct config of the chosen suites for 30 steps on one GPU, about 10 minutes, and prints a table with peak memory and estimated hours. **Any failure aborts before launching.** Skip it with `--no-smoke` if you've just run it.
3. **launch**: starts the runner in the background on the instance, so it keeps going after you disconnect.
   - It spreads runs over every GPU, one per GPU.
   - Runs go seed by seed, so every config gets one seed before any gets three.
   - Each run stops at step 20,000 (`--until`) with a resumable checkpoint.
4. **watch**: every 60 s (`--every`), it copies the instance's `runs/` into `runs/vast/` and prints what's training:
   ```
   21:04:10  8 training  |  runner: 2026-10-17 21:03:55  t1_huginn_ref_p113_h3_x6_s41: start on 3
       t1_anchor_nope_ref_p113_h3_x6_s41 (process 812): step 14200, val 0.998
       t1_ut_ref_p113_h3_x6_s41 (process 813): step 9100, val 0.031
       ...
   ```
   - Checkpoints are left out while runs are going.
   - When nothing is running for two rounds in a row, it makes one last full copy, checkpoints included, and exits.

**Live charts:** in a second terminal run `uv run tensorboard --logdir runs/vast` and open http://localhost:6006.
- Files are updated in place, so TensorBoard picks up new points on its own.
- The **Custom Scalars** tab overlays val accuracy by hop, the progress measures, and the per-step weight and update norms for each run.

**Ctrl-C only stops the streaming;** training keeps going. To pick it back up, from this or any other machine:

```bash
scripts/vast.sh watch $H $P --into runs/vast
```

## 5. Doing it step by step

`up` is just these commands in a row, and each can be run on its own:

```bash
scripts/vast.sh push   $H $P
scripts/vast.sh smoke  $H $P --suite tier1 tier2 tier3
scripts/vast.sh launch $H $P --suite tier1 tier2 tier3 --until 20000
scripts/vast.sh watch  $H $P --into runs/vast [--every 60]
scripts/vast.sh status $H $P            # live jobs, the runner log, gpu use, verdicts per config
scripts/vast.sh pull   $H $P --into runs/vast    # one full copy, checkpoints included
scripts/vast.sh shell  $H $P            # a shell on the instance, in the checkout
```

- **Smoke test results:**
  - On a 4090 the largest config (`huginn_fullbackprop`) peaked at 10.4 GB. If one runs out of memory, add a smaller `chunk_size` (e.g. 2048) to that entry's `train:` block, commit, and `push` again. That changes only how random starts are drawn, not the gradient.
  - On a 4090, a Huginn config should show about 0.2 s per step. Much higher points to a slow instance.
- **`push` refuses while anything is training on the instance**, so code never changes under a running suite.

## 6. Pause, resume, extend, recover

```bash
scripts/vast.sh stop   $H $P                                                  # every job saves and exits
scripts/vast.sh launch $H $P --suite tier1 tier2 tier3 --until 20000          # carries on where it stopped
scripts/vast.sh launch $H $P --runs t1_huginn_embedstart_p113_h3_x6_s41       # extend one run toward 40k
```

What the runner does with each run:
- **skips** runs that are complete, diverged, already at `--until`, or being trained by a live process;
- **resumes** runs that have a `latest.pt`;
- **starts** the rest.

Other behaviour to know:
- **Every run is resumable from its first eval.** `latest.pt` is written at step 0.
- **Resuming is deterministic,** so a paused run ends where an uninterrupted one would.
- **Relaunching right after `stop` is fine.** Wait until `status` says `nothing running`.
- **Judged weights are kept.** An early stop also saves `checkpoints/step_0020000.pt`, so extending a run never loses the model you judged.
- **A config entry is frozen once its run exists.** Resuming refuses if the entry differs from the run's saved `config.yaml`; for a change, add a new entry.
- **Divergence stops a run.** If a run's loss stops being finite, it stops and records `diverged_at` in `meta.json`. The runner won't relaunch it, and verdicts call it `diverged`.
- **Extending:** a run judged `slow` (restricted loss still falling at 20k) should be extended to 40k before you call it a failure. So should a run that grokked so late that its stability window runs past 20k.
- **Preemption:**
  - same instance restarted: `launch` again;
  - new instance: `push`, copy the run folders up (`rsync -az runs/vast/ -e "ssh -p $P" root@$H:/workspace/time-loops/runs/`), then `launch`.

## 7. Analyse the results locally

```bash
uv run python src/verdicts.py --all --by-config --root runs/vast          # one row per config, 3 seeds combined
uv run python src/verdicts.py --all --root runs/vast                      # one row per run
uv run python src/verdicts.py --all --by-config --root runs/vast --at 40000   # after extending slow runs
uv run python src/plot_progress.py runs/vast/t1_huginn_ref_p113_h3_x6_s41     # progress figure for one run
```

**The verdicts:**

| verdict | meaning |
|---|---|
| **groks** | val ≥ 0.9 by the judged step; t_grok is when |
| **stable** | fraction of evals at ≥ 0.9 in [t_grok + 1k, t_grok + 10k] |
| **slow** | not grokked, but restricted loss still falling |
| **fails** | not grokked, restricted loss flat |
| **diverged** | the loss stopped being finite |
| **fixed point** | looped runs only: the answer holds from the training depth out to 16 passes |

**For one run in depth:** open `notebooks/run-analysis.ipynb` and set `RUN_DIR` to its folder. It rebuilds everything from the folder and analyses the last checkpoint (`step_0020000.pt` for a run stopped at 20k).

The seeds vary initial weights, depth draws and random starts. Every run shares the dataset's single train/val split.

**When everything is copied, destroy the instance on the vast.ai page.** A stopped instance still bills for disk.

## 8. Adding an experiment

1. Add entries to `configs/train.yaml`: one per seed (41, 42, 43), named `<tier>_<model>_<change>_p113_h3_x6_s<seed>`, with a plain-words comment above them.
2. Add the names to a list in `configs/suites.yaml`, seed by seed. Use an existing tier or a new list.
3. Check locally with `uv run python src/suite.py --suite <list> --dry-run --devices 0`, then commit.
4. Run `scripts/vast.sh up $H $P --into runs/vast --suite <list> --until 20000`.

**Never edit an entry once its run exists;** add a new entry instead. Resuming enforces this.

## Troubleshooting

| symptom | fix |
|---|---|
| `torch sees no gpu` after push | the driver is too old for CUDA 13: rent a different instance |
| `uncommitted changes would not be shipped` | commit, or set `ALLOW_DIRTY=1` to ship HEAD anyway |
| `a config failed its smoke test` from `up` | read the smoke table: usually out of memory (lower that entry's `chunk_size`), then commit and run `up` again |
| `training is running on the instance` from `push` | `stop`, wait for `status` to say `nothing running`, then push |
| `stop the running jobs first` from `launch` | same: `stop`, wait, launch |
| `watch` shows `unreachable` | the connection dropped; it retries every round. Training is unaffected |
| `status` lists training jobs but no runner (the runner was killed) | `stop` signals those jobs directly; they pause at their next step |
| a run shows `skip: being trained by process N` | a live process is training it; check `status` before doing anything |
| a run shows `skip: run directory exists with nothing to resume from` | it was killed during its first eval, before step 0 was saved; delete `runs/<name>` on the instance (`shell`) and launch again |
| resuming refuses: `the config entry changed` | the entry was edited after the run started; restore it, or add a new entry |
| TensorBoard doesn't update | it reads `runs/vast` as `watch` copies into it; check that `watch` is still running and printing rounds |
