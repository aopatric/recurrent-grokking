"""branch a grokked checkpoint into a short run with one optimizer change, to test fixes for post-grok collapses.

    uv run python src/branch.py --source vast-runs/t1_huginn_ref_p113_h3_x6_s41 --step 8944 --arm control

arms (each changes one thing from the source run's optimizer):
    control     the source settings: constant lr, weight decay on everything
    lrdecay     lr decays linearly after warmup to decay_to * lr by the last step
    nowd_embed  no weight decay on the token embedding
    clip        global gradient-norm clipping at 1.0
    clip_lrdecay  clip and lrdecay together
    wd0, wd05, wd2  weight decay 0, 0.5 or 2 instead of 1, at the same constant lr
    wd2_lr05    weight decay 2 at half the lr: the same lr * weight decay product as control

the wd arms test the collapse mechanism causally: theory says the equilibrium step size, and so the collapse
rate, depends on lr * weight decay. so wd0 should fade out, wd05 < control < wd2, and wd2_lr05 should look like
control (same product) while plain wd2 doesn't.

the step checkpoints hold weights only, so every arm starts from a fresh adam; the control arm is the
baseline for that. every arm draws the same depths and random starts, so arms differ only in the optimizer.
the run directory matches train.py's (metrics.jsonl, steps.jsonl, tb/), so the same analysis and
tensorboard work on it, with steps counted from the branch point.
"""

import argparse
import json
import math
import platform
import sys
from pathlib import Path

import torch as t
import yaml

import depth as dp
import models
import progress
import tb_log
import train as tr_mod
from make_data import git_info

ARMS = {
    "control": {},
    "lrdecay": {"decay_to": 0.01},
    "nowd_embed": {"no_decay": ["embed.W_E"]},
    "clip": {"clip": 1.0},
    "clip_lrdecay": {"clip": 1.0, "decay_to": 0.01},
    "wd0": {"weight_decay": 0.0},
    "wd05": {"weight_decay": 0.5},
    "wd2": {"weight_decay": 2.0},
    "wd2_lr05": {"weight_decay": 2.0, "lr": 5e-4},
}


def lr_lambda(warm, steps, decay_to):
    """the lr multiplier for a step: linear warmup, then constant or linear decay.

    args:
        warm: warmup steps.
        steps: total branch steps.
        decay_to: final multiplier, or None for a constant lr after warmup.
    returns:
        function from step to multiplier.
    """
    def f(s):
        """multiplier at step s."""
        if warm and s < warm:
            return s / warm
        if decay_to is None:
            return 1.0
        frac = (s - warm) / max(steps - warm, 1)
        return 1.0 + (decay_to - 1.0) * min(frac, 1.0)
    return f


def branch(source, step, arm, steps, eval_every, warmup, out_root, device, log=print):
    """train one arm from a source run's checkpoint and write its run directory.

    args:
        source: source run directory, with config.yaml and checkpoints/.
        step: checkpoint step to branch from.
        arm: key of ARMS.
        steps: branch length.
        eval_every: steps between evaluations.
        warmup: lr warmup steps, to soften the fresh adam state.
        out_root: directory the branch run is written under.
        device: torch device string.
        log: function for progress lines.
    returns:
        path to the run directory.
    raises:
        tr_mod.RunConfigError: if the run directory exists or the checkpoint is missing.
    """
    source = Path(source)
    ck_path = source / "checkpoints" / f"step_{step:07d}.pt"
    if not ck_path.exists():
        raise tr_mod.RunConfigError(f"{ck_path} does not exist")
    entry = yaml.safe_load((source / "config.yaml").read_text())
    run_dir = Path(out_root) / f"{source.name}_from{step}_{arm}"
    if run_dir.exists():
        raise tr_mod.RunConfigError(f"{run_dir} exists; delete it to rerun")
    cfg = yaml.safe_load(Path("configs/train.yaml").read_text())

    spec = ARMS[arm]
    tr_mod.setup_precision()
    data = tr_mod.load_data(Path(cfg["data_root"]) / entry["data"], device)
    mcfg, rule, grad_iters, tr, op = tr_mod.resolve(entry, data["meta"])
    seed = entry["seed"]

    t.manual_seed(seed)
    model = models.LoopedTransformer(mcfg).to(device)
    model.load_state_dict(t.load(ck_path, map_location=device, weights_only=True))
    no_decay = op["no_decay"] + spec.get("no_decay", [])
    op = {**op, "weight_decay": spec.get("weight_decay", op["weight_decay"]), "lr": spec.get("lr", op["lr"])}
    opt = t.optim.AdamW(tr_mod.param_groups(model, op["weight_decay"], no_decay), lr=op["lr"],
                        betas=tuple(op["betas"]))
    sched = t.optim.lr_scheduler.LambdaLR(opt, lr_lambda(warmup, steps, spec.get("decay_to")))
    # same seeds in every arm, so depth draws and random starts line up step for step
    rng = dp.depth_rng(seed)
    state_gen = t.Generator(device=device).manual_seed(seed)

    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(entry, sort_keys=False))
    meta = {"git": git_info(), "python": platform.python_version(), "torch": t.__version__,
            "device": tr_mod._device_name(device),
            "branch": {"source": str(source), "step": step, "arm": arm, "spec": spec, "steps": steps,
                       "warmup": warmup, "optim": {**op, "no_decay": no_decay}}}
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    train_sets = data["sets"]["train"]
    grid_tokens, grid_labels, grid_train = data["grid"]
    eval_T = rule.eval_depth()
    metrics_path = run_dir / "metrics.jsonl"
    writer = tb_log.open_writer(run_dir, 0, False)

    def eval_gen():
        """fresh generator so every evaluation sees the same random starts."""
        return t.Generator(device=device).manual_seed(seed + 1)

    def log_eval(done, train_loss):
        """write an evaluation line with train, val, held-out sets and progress measures."""
        model.eval()
        row = {"kind": "eval", "step": done, "lr": sched.get_last_lr()[0], "depth": eval_T,
               "train_step_loss": train_loss,
               "train": tr_mod.evaluate(model, train_sets, eval_T, tr["chunk_size"], eval_gen()),
               "val": tr_mod.evaluate(model, data["sets"]["val"], eval_T, tr["chunk_size"], eval_gen()),
               "held_out": {name: tr_mod.evaluate(model, data["sets"][name], eval_T, tr["chunk_size"], eval_gen())
                            for name in tr_mod._EVAL_FILES if name in data["sets"]},
               "progress": progress.progress_measures(model, grid_tokens, grid_labels, grid_train, eval_T,
                                                       generator=eval_gen())}
        model.train()
        with open(metrics_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        tb_log.write_row(writer, {k: v for k, v in row.items() if k != "held_out"})
        accs = " ".join(f"h{h}={v['acc']:.3f}" for h, v in row["val"].items())
        shown = "-" if train_loss is None else f"{train_loss:.3e}"
        log(f"[{arm}] step {done:>5}  lr {row['lr']:.2e}  train loss {shown}  val acc {accs}")

    try:
        log_eval(0, None)
        model.train()
        for s in range(steps):
            r = rule.sample(rng)
            opt.zero_grad(set_to_none=True)
            total = tr_mod.accumulate_grads(model, train_sets, r, grad_iters, tr["chunk_size"], state_gen)
            if not math.isfinite(total) or not tr_mod._finite_grads(model):
                meta["diverged_at"] = s + 1
                (run_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
                log(f"[{arm}] diverged at step {s + 1}; stopping")
                break
            # logged before clipping, so grad_norm compares across arms
            before = {n: prm.detach().clone() for n, prm in model.named_parameters()}
            grad_norm, _ = tr_mod.step_stats(model, before)
            if "clip" in spec:
                t.nn.utils.clip_grad_norm_(model.parameters(), spec["clip"])
            opt.step()
            sched.step()
            done = s + 1
            _, groups = tr_mod.step_stats(model, before)
            row = {"step": done, "depth": r, "loss": total, "grad_norm": grad_norm, "groups": groups}
            with open(run_dir / "steps.jsonl", "a") as f:
                f.write(json.dumps(row) + "\n")
            tb_log.write_step(writer, row)
            if done % eval_every == 0 or done == steps:
                log_eval(done, total)
            if done % 1000 == 0 or done == steps:
                tr_mod._atomic_save(model.state_dict(), run_dir / "checkpoints" / f"step_{done:07d}.pt")
    finally:
        writer.close()
    return run_dir


def main(argv=None):
    """run the cli.

    args:
        argv: argument list, or None for sys.argv.
    returns:
        exit code: 0 ok, 2 bad arguments or run directory state.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="source run directory")
    ap.add_argument("--step", type=int, required=True, help="checkpoint step to branch from")
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--out", default="runs/branches")
    ap.add_argument("--device", default="cuda" if t.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)
    try:
        run_dir = branch(args.source, args.step, args.arm, args.steps, args.eval_every, args.warmup,
                         args.out, args.device, log=lambda s: print(s, flush=True))
    except (OSError, yaml.YAMLError, tr_mod.RunConfigError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(f"wrote {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
