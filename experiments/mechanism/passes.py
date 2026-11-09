"""what happens across loop passes in stable grokked looped models: per-pass accuracy by hop count, a logit lens
with a clock fit, attention from "=" and how much the state still changes.

    uv run python experiments/mechanism/passes.py [RUN_DIR ...] [--T 16] [--out runs/mech-passes] [--device cuda]

with no run dirs it analyses the two-phase runs (state-norm UT, sandwich UT, huginn) at their final checkpoint.
writes <out>/<run>.json and a combined <out>/summary.md.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch as t

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import analysis  # noqa: E402
import mech  # noqa: E402
import progress  # noqa: E402

DEFAULT_RUNS = ["runs/t6_ut_statenorm_twophase_p113_h3_x6_s41", "runs/t6_ut_sandwich_twophase_p113_h3_x6_s41",
                "runs/t6_huginn_twophase_p113_h3_x6_s41"]


def hop_groups(run):
    """closed-chain sets by hop count: val for the trained hops, extrap_depth for the longer ones.

    args:
        run: dict from analysis.load_run.
    returns:
        dict from hop count (int) to (tokens, answers).
    """
    sets = run["data"]["sets"]
    out = {int(h): v for h, v in sets["val"].items()}
    out.update({int(h): v for h, v in sets.get("extrap_depth", {}).items()})
    return dict(sorted(out.items()))


@t.no_grad()
def per_pass(model, run, groups, T, device):
    """per-pass readout accuracy, clock-fit r², attention entropy from "=" and relative state change.

    args:
        model: LoopedTransformer with a shared core.
        run: dict from analysis.load_run.
        groups: dict from hop count to (tokens, answers).
        T: deepest pass.
        device: torch device string.
    returns:
        dict from hop count to {"acc", "clock_r2", "attn_entropy", "attn_max", "state_change"}, each a list of T
        values (attention ones are per core layer: a list of T lists).
    """
    p = run["cfg"].p
    basis = progress.fourier_basis(p, device)
    # key frequencies: the 5 largest fourier components of the unembedding (W_U is d_model x p)
    keys = (mech.freq_norms(model.unembed.W_U, basis)[1:].argsort(descending=True)[:5] + 1).tolist()
    attn_names = sorted({n for n, m in model.named_modules() if n.startswith("core.") and n.endswith("attn.hook_attn")})
    out = {}
    for h, (x, y) in groups.items():
        x, y = x[:1000], y[:1000]
        gen = analysis.eval_generator(run, device)
        cache = {}
        model.cache_all(cache)
        try:
            logits = model(x, T, return_all=True, generator=gen)
        finally:
            model.remove_all_hooks()
        acc = (logits.argmax(-1) == y[:, None]).float().mean(0).tolist()
        r2 = [mech.clock_fit(logits[:, i].float(), y, keys, p)[1] for i in range(T)]
        n_tok = x.shape[1]
        ent, mx = [], []
        for i in range(T):
            e_i, m_i = [], []
            for name in attn_names:
                a = cache[(name, i)][:, :, -1, :]  # (batch, head, key) from the "=" query
                e_i.append(float((mech.entropy(a) / math.log(n_tok)).mean()))
                m_i.append(float(a.max(-1).values.mean()))
            ent.append(e_i)
            mx.append(m_i)
        states = [cache[("hook_state", i)][:, -1] for i in range(T)]
        change = [float("nan")] + [float(((states[i] - states[i - 1]).norm(dim=-1) / states[i].norm(dim=-1)).mean())
                                   for i in range(1, T)]
        out[h] = {"acc": acc, "clock_r2": r2, "attn_entropy": ent, "attn_max": mx, "state_change": change,
                  "keys": keys}
    return out


def t_star(acc, frac=0.95):
    """the first pass reaching frac of the best accuracy over all passes.

    args:
        acc: list of per-pass accuracies.
        frac: fraction of the best.
    returns:
        1-based pass index, or None if the best is near chance.
    """
    best = max(acc)
    if best < 0.1:
        return None
    return next(i + 1 for i, a in enumerate(acc) if a >= frac * best)


def summarize(name, res, T):
    """markdown section for one run.

    args:
        name: run name.
        res: dict from per_pass.
        T: deepest pass.
    returns:
        markdown string.
    """
    show = [i for i in (1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 24, 32) if i <= T]
    lines = [f"## {name}", "", "per-pass accuracy, and T* (first pass reaching 95% of the best):", "",
             "| hops | " + " | ".join(f"p{i}" for i in show) + " | best | T* |",
             "|---|" + "---|" * (len(show) + 2)]
    for h, r in res.items():
        lines.append(f"| {h} | " + " | ".join(f"{r['acc'][i - 1]:.2f}" for i in show)
                     + f" | {max(r['acc']):.2f} | {t_star(r['acc'])} |")
    lines += ["", "clock-fit r² of the readout after each pass (key frequencies "
              f"{next(iter(res.values()))['keys']}):", "",
              "| hops | " + " | ".join(f"p{i}" for i in show) + " |", "|---|" + "---|" * len(show)]
    for h, r in res.items():
        lines.append(f"| {h} | " + " | ".join(f"{r['clock_r2'][i - 1]:.2f}" for i in show) + " |")
    lines += ["", "attention from \"=\" in the core, normalized entropy (1 = even average) / mean max weight:", "",
              "| hops | " + " | ".join(f"p{i}" for i in show) + " |", "|---|" + "---|" * len(show)]
    for h, r in res.items():
        lines.append(f"| {h} | " + " | ".join(
            "/".join(f"{e:.2f}" for e in r["attn_entropy"][i - 1]) + " · "
            + "/".join(f"{m:.2f}" for m in r["attn_max"][i - 1]) for i in show) + " |")
    lines += ["", "relative change of the state at \"=\" from the previous pass:", "",
              "| hops | " + " | ".join(f"p{i}" for i in show) + " |", "|---|" + "---|" * len(show)]
    for h, r in res.items():
        lines.append(f"| {h} | " + " | ".join(f"{r['state_change'][i - 1]:.3f}" for i in show) + " |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    """run the cli.

    args:
        argv: argument list, or None for sys.argv.
    returns:
        exit code.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", default=DEFAULT_RUNS)
    ap.add_argument("--T", type=int, default=16)
    ap.add_argument("--out", default="runs/mech-passes")
    ap.add_argument("--device", default="cuda" if t.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sections = []
    for rd in args.runs:
        run = analysis.load_run(rd, device=args.device)
        step, path = analysis.checkpoints(run)[-1]
        model = analysis.load_model(run, path, args.device)
        res = per_pass(model, run, hop_groups(run), args.T, args.device)
        (out / f"{run['name']}.json").write_text(json.dumps({"step": step, "T": args.T, "by_hop": res}))
        sections.append(summarize(f"{run['name']} (step {step})", res, args.T))
        print(f"done {run['name']}", flush=True)
    (out / "summary.md").write_text("# per-pass analysis of stable grokked looped models\n\n" + "\n".join(sections))
    print(f"wrote {out / 'summary.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
