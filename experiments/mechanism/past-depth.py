"""run stable grokked looped models far past their training depth: does the answer settle (fixed point / contraction)
or break?

    uv run python experiments/mechanism/past-depth.py [RUN_DIR ...] [--T 256] [--perturb-at 8] [--eps 0.01] [--at STEP] [--out runs/mech-long]

for each hop count it records, at every pass up to T: readout loss and accuracy, the relative change of the state,
and the growth of a small perturbation added to the state at pass --perturb-at (ratio < 1 per pass means the loop
contracts around the trajectory, > 1 means it amplifies). each run is read at its last checkpoint, or at --at.
writes <out>/<run>.json and <out>/summary.md.
"""

import argparse
import json
import sys
from pathlib import Path

import torch as t

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import analysis  # noqa: E402
import progress  # noqa: E402

DEFAULT_RUNS = ["runs/t6_huginn_twophase_p113_h3_x6_s41", "runs/t6_ut_statenorm_twophase_p113_h3_x6_s41",
                "runs/t6_ut_sandwich_twophase_p113_h3_x6_s41"]
SHOW = (1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256)


def embed(model, x):
    """embedding, positions and prelude, as in LoopedTransformer.forward.

    args:
        model: LoopedTransformer.
        x: token ids.
    returns:
        the injected input e.
    """
    e = model.embed.embed(x)
    if model.pos_embed is not None:
        e = model.pos_embed.embed(e)
    for blk in model.prelude:
        e = blk(e)
    return e


def step(model, s, e):
    """one pass of the shared core, as in LoopedTransformer.forward.

    args:
        model: LoopedTransformer with a shared core.
        s: state.
        e: injected input.
    returns:
        the next state.
    """
    u = model.inject(s, e)
    for blk in model.core[0]:
        u = blk(u)
    if model.core_norm is not None:
        u = model.core_norm(u)
    return u


@t.no_grad()
def long_run(model, run, x, y, T, perturb_at, eps, device):
    """per-pass loss, accuracy, state change and perturbation growth.

    args:
        model: LoopedTransformer with a shared core.
        run: dict from analysis.load_run.
        x, y: tokens and answers.
        T: deepest pass.
        perturb_at: pass after which a perturbation is added.
        eps: perturbation size relative to the state norm at "=" positions (applied at every position).
        device: torch device string.
    returns:
        dict of per-pass lists: loss, acc, change, pert (perturbation norm relative to the state; nan before the
        perturbation), plus pert_acc (accuracy of the perturbed trajectory).
    """
    e = embed(model, x)
    s = model.initial_state(e, analysis.eval_generator(run, device))
    loss, acc, change, pert, pert_acc = [], [], [], [], []
    s2 = None
    g = t.Generator(device=device).manual_seed(0)
    for i in range(T):
        s_new = step(model, s, e)
        change.append(float(((s_new - s).norm(dim=-1) / s_new.norm(dim=-1).clamp_min(1e-12))[:, -1].mean()))
        s = s_new
        logits = model.readout(s, i)
        loss.append(float(progress.cross_entropy_high_precision(logits, y)))
        acc.append(float((logits.argmax(-1) == y).float().mean()))
        if i + 1 == perturb_at:
            d = t.randn(s.shape, generator=g, device=device, dtype=s.dtype)
            d = d / d.norm(dim=-1, keepdim=True) * s.norm(dim=-1, keepdim=True) * eps
            s2 = s + d
        elif s2 is not None:
            s2 = step(model, s2, e)
        if s2 is not None:
            pert.append(float(((s2 - s).norm(dim=-1) / s.norm(dim=-1).clamp_min(1e-12))[:, -1].mean()))
            pert_acc.append(float((model.readout(s2, i).argmax(-1) == y).float().mean()))
        else:
            pert.append(float("nan"))
            pert_acc.append(float("nan"))
    return {"loss": loss, "acc": acc, "change": change, "pert": pert, "pert_acc": pert_acc}


def pick(run, at):
    """the checkpoint to read.

    args:
        run: dict from analysis.load_run.
        at: training step, or None for the last checkpoint.
    returns:
        (step, path).
    raises:
        ValueError: if the run has no checkpoint at that step.
    """
    cks = analysis.checkpoints(run)
    if at is None:
        return cks[-1]
    for step_, path in cks:
        if step_ == at:
            return step_, path
    raise ValueError(f"{run['name']}: no checkpoint at step {at}")


def main(argv=None):
    """run the cli.

    args:
        argv: argument list, or None for sys.argv.
    returns:
        exit code.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", default=DEFAULT_RUNS)
    ap.add_argument("--T", type=int, default=256)
    ap.add_argument("--perturb-at", type=int, default=8)
    ap.add_argument("--eps", type=float, default=0.01)
    ap.add_argument("--n", type=int, default=1000, help="examples per hop count")
    ap.add_argument("--at", type=int, help="checkpoint step to read in every run (default: each run's last)")
    ap.add_argument("--out", default="runs/mech-long")
    ap.add_argument("--device", default="cuda" if t.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    show = [i for i in SHOW if i <= args.T]
    md = ["# looped models run past their training depth", "",
          f"perturbation: relative size {args.eps} added to the state after pass {args.perturb_at}; "
          "\"pert\" is its size relative to the state afterwards (shrinking = contraction).", ""]
    for rd in args.runs:
        run = analysis.load_run(rd, device=args.device)
        step_, path = pick(run, args.at)
        model = analysis.load_model(run, path, args.device)
        sets = run["data"]["sets"]
        groups = {int(h): v for h, v in sets["val"].items()}
        groups.update({int(h): v for h, v in sets.get("extrap_depth", {}).items()})
        res = {h: long_run(model, run, x[:args.n], y[:args.n], args.T, args.perturb_at, args.eps, args.device)
               for h, (x, y) in sorted(groups.items())}
        # --at is left out when unset, so the outputs read at the last checkpoint keep their earlier form
        params = {k: v for k, v in vars(args).items() if not (k == "at" and v is None)}
        (out / f"{run['name']}.json").write_text(json.dumps({"step": step_, **params, "by_hop": res}))
        md += [f"## {run['name']} (step {step_})", ""]
        for key, fmt in (("loss", "{:.3f}"), ("acc", "{:.2f}"), ("change", "{:.1e}"), ("pert", "{:.1e}"),
                         ("pert_acc", "{:.2f}")):
            md += [f"**{key}** by pass:", "", "| hops | " + " | ".join(f"p{i}" for i in show) + " |",
                   "|---|" + "---|" * len(show)]
            for h, r in res.items():
                md.append(f"| {h} | " + " | ".join(fmt.format(r[key][i - 1]) for i in show) + " |")
            md.append("")
        print(f"done {run['name']}", flush=True)
    (out / "summary.md").write_text("\n".join(md))
    print(f"wrote {out / 'summary.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
