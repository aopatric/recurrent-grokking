# Running the suites on vast.ai

`scripts/vast.sh` does everything from your own machine over ssh: it ships the code and data to a rented GPU box, runs the suites there, and copies the run folders back so you can analyse them locally. Nothing needs to be installed by hand on the instance.

## Once: account and key

1. Create a vast.ai account and add credit.
2. Add your public ssh key under **Account → Keys**. Use `cat ~/.ssh/id_ed25519.pub`, or create a key first with `ssh-keygen -t ed25519`.
3. Locally you need `git`, `ssh`, `rsync` and `sha256sum`.

## 1. Rent an instance

On the **Search** page:
- **GPU:** RTX 4090 or 3090. These models are tiny, so A100s and H100s buy nothing. One machine with several GPUs is simplest: the runner keeps one run on each GPU.
- **CUDA:** filter for **max CUDA ≥ 13.0**. The lockfile pins a CUDA 13 build of torch, and older drivers can't run it.
- **RAM:** at least 32 GB. Each job holds a few GB of host memory for torch and CUDA, and there is one job per GPU.
- **Disk:** at least 30 GB. Each run folder is about 100 MB, and all 72 suite runs come to roughly 8 GB.
- **Image:** any Ubuntu or CUDA image with ssh launch, e.g. `vastai/base-image` or a `pytorch/pytorch` image. The script installs everything else.

Once it's running, click **Connect**. You'll see something like `ssh -p 41234 root@ssh5.vast.ai`. **HOST** is `ssh5.vast.ai` and **PORT** is `41234`. The examples below use:

```bash
export H=ssh5.vast.ai P=41234
```

## 2. Ship the code and data

```bash
git status                       # commit first: only the committed HEAD is shipped
scripts/vast.sh push $H $P
```

This command:
- copies HEAD as a git bundle, so every run records its commit;
- copies `data/synthetic` and checks the data checksums;
- installs uv and the locked environment;
- confirms that torch can see the GPUs.

- It refuses if tracked files have uncommitted changes (`ALLOW_DIRTY=1` ships HEAD anyway). Untracked files are only listed, since they aren't shipped.
- It also refuses while a runner or training job is alive on the instance, so code never changes under a running suite. `stop` first.
- Re-run `push` after any new commit. Run folders on the instance are never touched.

## 3. Smoke test (≈ 10 min)

```bash
scripts/vast.sh smoke $H $P --suite tier1 tier2 tier3
```

It trains each distinct config for 30 steps and prints peak GPU memory and a rough number of hours per 20k steps.
- **Check the peak memory column.** On a 4090 the largest config (`huginn_fullbackprop`) peaked at 10.4 GB. If one fails with out-of-memory or comes near the card's size, add a smaller `chunk_size` (e.g. 2048) to that entry's `train:` block, then commit and push again. This changes only how random starts are drawn, not the gradient.
- **Do this before the run exists.** Once a run has started, its config entry is frozen: resuming refuses if the entry differs from the run's saved `config.yaml`. For a change after that, add a new entry.
- **Budget:** multiply the hours column by 3 seeds to estimate each tier.

## 4. Launch

```bash
scripts/vast.sh launch $H $P --suite tier1 --until 20000          # tier 1, judged at 20k steps
scripts/vast.sh launch $H $P --suite tier1 tier2 --until 20000    # or queue both tiers
```

- The runner spreads the runs over every GPU, one run per GPU at a time.
- Runs are ordered seed by seed, so every config gets one seed before any gets three.
- It keeps going after you disconnect. Its output goes to `runs/suite.log` on the instance.
- Only one runner can go at a time. To run on part of the GPUs, add `--devices 0,1`. These count among the GPUs the instance exposes, even if it sets `CUDA_VISIBLE_DEVICES`.
- If a run's loss stops being finite, it stops there and is recorded as diverged (`diverged_at` in its `meta.json`). The runner won't relaunch it.

## 5. Watch and pull

```bash
scripts/vast.sh status $H $P     # runner state, the last log lines, gpu use, verdicts per config
scripts/vast.sh pull $H $P       # copy runs/ into your local runs/; safe to repeat while running
scripts/vast.sh shell $H $P      # a shell on the instance, in the checkout
```

For live charts, pull and then run `uv run tensorboard --logdir runs` locally.

## 6. Pause, resume, extend

```bash
scripts/vast.sh stop $H $P                                                  # every job saves and exits
scripts/vast.sh launch $H $P --suite tier1 --until 20000                    # carries on where it stopped
scripts/vast.sh launch $H $P --runs t1_huginn_embedstart_p113_h3_x6_s41     # extend one slow run to 40k
```

- **What the runner does with each run:** skips runs that are finished, diverged, already at `--until`, or being trained by a live process; resumes runs with a `latest.pt`; starts the rest.
- **Every run is resumable from its first eval**, because `latest.pt` is written at step 0.
- **Resuming is deterministic**, so a paused run ends where an uninterrupted one would.
- **Relaunching right after `stop` is fine.** Wait for `status` to say `nothing running`, then launch.
- **After an early stop,** the weights at that step are also kept as `checkpoints/step_0020000.pt`, so extending a run never loses the model you judged.
- **If the instance is preempted:**
  - same instance: just `launch` again;
  - new instance: `push`, copy the run folders up with `rsync`, then `launch`.

## 7. Analyse locally

```bash
scripts/vast.sh pull $H $P
uv run python src/verdicts.py --all --by-config      # one row per config across its 3 seeds, judged at 20k
uv run python src/verdicts.py --all                  # one row per run
uv run python src/verdicts.py --all --at 40000       # after extending slow runs
```

The seeds vary initial weights, depth draws and random starts. Every run shares the dataset's single train/val split.

For any run folder you can also use `notebooks/run-analysis.ipynb` (point it at the folder), `src/plot_progress.py` and TensorBoard.

**When everything is pulled, destroy the instance on the vast.ai page.** Stopped instances still bill for disk.

## Troubleshooting

| symptom | fix |
|---|---|
| `torch sees no gpu` after push | the driver is too old for CUDA 13: rent a different instance |
| `uncommitted changes would not be shipped` | commit, or set `ALLOW_DIRTY=1` |
| `stop the running jobs first` from `launch` | `stop` first, wait for `status` to say `nothing running`, then launch |
| a run shows `skip: run directory exists with nothing to resume from` | it was killed during its first eval, before step 0 was saved; delete `runs/<name>` on the instance and launch again |
| `skip: being trained by process N` | a live process is training it; check `status` before doing anything |
| `resuming refuses: the config entry changed` | the entry was edited after the run started; restore it, or add a new entry |
| `status` lists training jobs but no runner (the runner was killed) | `stop` signals those jobs directly; they pause at their next step |
| `training is running on the instance` from `push` | `stop`, wait for `status` to say `nothing running`, then push |
