"""does a looped model use the state it carries between passes, and does its injection hold the answer in place?

    uv run python experiments/mechanism/state-use.py RUN_DIR ... [--at final|best|all|STEP,STEP] [--inject]
        [--n 1000] [--out runs/mech-state] [--device cuda]

at each chosen checkpoint, on the first --n closed chains of each trained hop count, it records:
- normal: accuracy after the run's eval depth of passes.
- noise: accuracy when the carried state is replaced by fresh random noise before every pass (the noise is drawn
  like a random start). a model that ignores its state keeps its accuracy; one that relies on it drops to chance.
- blocks: norms of the state and input halves of the concat injection matrix W_A (spectral and frobenius).
- with --inject: after the eval depth, the loop keeps running to --T passes either normally, with the injected
  input zeroed, or with the injection skipped (the state goes straight into the core); for each, the accuracy and
  the number of distinct answers 1, 8 and 56 passes later and at the last pass.

--at picks checkpoints: final, best (the highest mean accuracy over the trained hops), all, or a list of steps.
writes <out>/<run>.json and <out>/summary.md.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch as t

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import analysis  # noqa: E402

NOISE_SEED = 0
INJECT_MODES = ("normal", "zero_input", "skip")
# passes after the eval depth at which the injection-removal probe reads out (plus the last pass)
AFTER = (1, 8, 56)


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


def core(model, u):
    """the shared core and its norm, as in LoopedTransformer.forward.

    args:
        model: LoopedTransformer with a shared core.
        u: input to the core.
    returns:
        the next state.
    """
    for blk in model.core[0]:
        u = blk(u)
    if model.core_norm is not None:
        u = model.core_norm(u)
    return u


def noise_like(model, e, generator):
    """a fresh random state, drawn like the model's random start (truncated normal, scaled by 1 / sqrt(d)).

    args:
        model: LoopedTransformer.
        e: tensor whose shape, device and dtype to match.
        generator: torch generator.
    returns:
        tensor shaped like e.
    """
    s = t.randn(e.shape, generator=generator, device=e.device, dtype=e.dtype)
    while (bad := s.abs() > 3).any():
        s[bad] = t.randn(int(bad.sum()), generator=generator, device=e.device, dtype=e.dtype)
    return s / np.sqrt(model.cfg.d_model)


def accuracy(logits, y):
    """share of rows whose top logit is the answer.

    args:
        logits: tensor of shape (n, p).
        y: answers, shape (n,).
    returns:
        float.
    """
    return float((logits.argmax(-1) == y).float().mean())


@t.no_grad()
def probe(model, run, x, y, T_long, with_inject, device):
    """normal and noise-state accuracy, and optionally the injection-removal runs, for one batch.

    args:
        model: LoopedTransformer with a shared core.
        run: dict from analysis.load_run.
        x, y: tokens and answers.
        T_long: last pass of the injection-removal runs.
        with_inject: run the injection-removal probe.
        device: torch device string.
    returns:
        dict with normal and noise accuracy, plus inject when asked: mode -> passes after the eval depth (as a
        string) -> {"acc", "distinct"}.
    """
    T = run["T"]
    e = embed(model, x)
    s = model.initial_state(e, analysis.eval_generator(run, device))
    g = t.Generator(device=device).manual_seed(NOISE_SEED)
    s_noise = None
    for _ in range(T):
        s = core(model, model.inject(s, e))
        s_noise = core(model, model.inject(noise_like(model, e, g), e))
    out = {"normal": accuracy(model.readout(s, T - 1), y), "noise": accuracy(model.readout(s_noise, T - 1), y)}
    if with_inject:
        out["inject"] = {}
        zero = t.zeros_like(e)
        marks = {a for a in AFTER if T + a <= T_long} | {T_long - T}
        for mode in INJECT_MODES:
            s2, res = s, {}
            for i in range(T, T_long):
                if mode == "normal":
                    s2 = core(model, model.inject(s2, e))
                elif mode == "zero_input":
                    s2 = core(model, model.inject(s2, zero))
                else:
                    s2 = core(model, s2)
                if i + 1 - T in marks:
                    pred = model.readout(s2, i).argmax(-1)
                    res[str(i + 1 - T)] = {"acc": float((pred == y).float().mean()),
                                           "distinct": int(pred.unique().numel())}
            out["inject"][mode] = res
    return out


@t.no_grad()
def blocks(model):
    """norms of the state and input halves of the concat injection matrix.

    args:
        model: LoopedTransformer.
    returns:
        dict of norms, or None when the model doesn't use concat injection.
    """
    if model.cfg.injection != "concat":
        return None
    d = model.cfg.d_model
    out = {}
    for name, W in (("state", model.W_A[:, :d]), ("input", model.W_A[:, d:])):
        out[name] = {"spectral": float(t.linalg.matrix_norm(W, ord=2)), "frobenius": float(W.norm())}
    return out


def choose(run, at, groups, n, device):
    """the checkpoints to probe.

    args:
        run: dict from analysis.load_run.
        at: "final", "best", "all", or comma-separated steps.
        groups: dict from hop count to (tokens, answers).
        n: examples per hop count.
        device: torch device string.
    returns:
        list of (step, path).
    raises:
        ValueError: if a listed step has no checkpoint.
    """
    cks = analysis.checkpoints(run)
    if at == "final":
        return cks[-1:]
    if at == "all":
        return cks
    if at == "best":
        scores = []
        for step, path in cks:
            model = analysis.load_model(run, path, device)
            accs = [probe(model, run, x[:n], y[:n], run["T"], False, device)["normal"] for x, y in groups.values()]
            scores.append(float(np.mean(accs)))
        # ties go to the later checkpoint
        best = max(range(len(cks)), key=lambda i: (scores[i], i))
        return [cks[best]]
    steps = {int(s) for s in at.split(",")}
    chosen = [c for c in cks if c[0] in steps]
    if len(chosen) != len(steps):
        raise ValueError(f"{run['name']}: no checkpoint at {sorted(steps - {c[0] for c in chosen})}")
    return chosen


def summarize(name, rows):
    """markdown lines for one run.

    args:
        name: run name.
        rows: list of per-checkpoint dicts.
    returns:
        list of markdown lines.
    """
    lines = [f"## {name}", "", "| step | normal | noise state | W_A state / input (spectral) | "
             + " | ".join(f"{m}: distinct answers per hop, " + " / ".join(f"+{a}" for a in AFTER) for m in INJECT_MODES)
             + " |", "|---|---|---|---|" + "---|" * len(INJECT_MODES)]
    for r in rows:
        b = r["blocks"]
        bl = f"{b['state']['spectral']:.2f} / {b['input']['spectral']:.2f}" if b else "—"
        inj = []
        for m in INJECT_MODES:
            if "inject" not in r["by_hop"][next(iter(r["by_hop"]))]:
                inj.append("—")
                continue
            inj.append("; ".join(f"h{h}: " + " / ".join(str(v["inject"][m][str(a)]["distinct"]) for a in AFTER
                                                         if str(a) in v["inject"][m])
                                 for h, v in r["by_hop"].items()))
        lines.append(f"| {r['step']} | {r['normal']:.2f} | {r['noise']:.2f} | {bl} | " + " | ".join(inj) + " |")
    return lines + [""]


def main(argv=None):
    """run the cli.

    args:
        argv: argument list, or None for sys.argv.
    returns:
        exit code.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--at", default="final")
    ap.add_argument("--inject", action="store_true", help="also run the injection-removal probe")
    ap.add_argument("--T", type=int, default=256, help="last pass of the injection-removal probe")
    ap.add_argument("--n", type=int, default=1000, help="examples per hop count")
    ap.add_argument("--out", default="runs/mech-state")
    ap.add_argument("--device", default="cuda" if t.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    md = ["# use of the carried state", "", "accuracies are means over the trained hop counts. noise state: fresh "
          "noise in place of the state before every pass. distinct answers: among --n inputs per hop count, the given "
          "number of passes after the eval depth.", ""]
    for rd in args.runs:
        run = analysis.load_run(rd, device=args.device)
        groups = {int(h): v for h, v in sorted(run["data"]["sets"]["val"].items())}
        rows = []
        for step, path in choose(run, args.at, groups, args.n, args.device):
            model = analysis.load_model(run, path, args.device)
            by_hop = {h: probe(model, run, x[:args.n], y[:args.n], args.T, args.inject, args.device)
                      for h, (x, y) in groups.items()}
            row = {"step": step, "by_hop": by_hop, "blocks": blocks(model),
                   "normal": float(np.mean([r["normal"] for r in by_hop.values()])),
                   "noise": float(np.mean([r["noise"] for r in by_hop.values()]))}
            rows.append(row)
        (out / f"{run['name']}.json").write_text(json.dumps({"T": run["T"], "n": args.n, "checkpoints": rows}))
        md += summarize(run["name"], rows)
        print(f"done {run['name']}", flush=True)
    (out / "summary.md").write_text("\n".join(md))
    print(f"wrote {out / 'summary.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
