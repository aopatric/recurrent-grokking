"""how many fourier frequencies each model's readout uses, on the hop-1 grid (every a + b mod p).

    uv run python experiments/mechanism/frequencies.py RUN_DIR ... [--out runs/mech-freq] [--device cuda]

frequencies are ranked by the size of the grid logits' component along cos/sin(w(a+b)), as in the progress
measures. for each k it records the excluded loss (train-pair loss after removing the top k frequencies from the
logits) and the clock-fit r² (share of the row-centred logit variance explained by cos/sin(w(a+b-c)) at the top k
frequencies). each run is read at its final checkpoint and eval depth. writes <out>/<run>.json and <out>/summary.md.
"""

import argparse
import json
import sys
from pathlib import Path

import torch as t

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import analysis  # noqa: E402
import mech  # noqa: E402
import progress  # noqa: E402

KS = (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 56)
SHOW = (5, 10, 20, 40, 56)


@t.no_grad()
def sweep(model, run, ks, device):
    """excluded loss and clock-fit r² for the top k frequencies, for each k.

    args:
        model: LoopedTransformer.
        run: dict from analysis.load_run.
        ks: numbers of frequencies to try.
        device: torch device string.
    returns:
        dict with ks, order (frequencies, largest first), freq_norms (by frequency 1..p//2), excluded_loss and
        clock_r2 (one value per k), and the grid's test accuracy and loss with nothing removed.
    """
    p = run["cfg"].p
    tokens, labels, is_train = run["data"]["grid"]
    logits = model(tokens, run["T"], generator=analysis.eval_generator(run, device)).float()
    dirs = progress.xpy_directions(progress.fourier_basis(p, device))
    coef = dirs @ logits
    norms = (coef[0::2].pow(2).sum(-1) + coef[1::2].pow(2).sum(-1)).sqrt()
    order = (norms.argsort(descending=True) + 1).tolist()
    ce = progress.cross_entropy_high_precision
    excluded, r2 = [], []
    for k in ks:
        keys = order[:k]
        rest = logits - progress.key_component(logits, dirs, keys)
        excluded.append(float(ce(rest[is_train], labels[is_train])))
        r2.append(mech.clock_fit(logits, labels, keys, p)[1])
    return {"ks": list(ks), "order": order, "freq_norms": [float(v) for v in norms],
            "excluded_loss": excluded, "clock_r2": r2,
            "grid_test_acc": float((logits[~is_train].argmax(-1) == labels[~is_train]).float().mean()),
            "grid_test_loss": float(ce(logits[~is_train], labels[~is_train]))}


def summarize(name, step, res):
    """markdown lines for one run.

    args:
        name: run name.
        step: checkpoint step.
        res: dict from sweep.
    returns:
        list of markdown lines.
    """
    cols = [k for k in SHOW if k in res["ks"]]
    at = {k: i for i, k in enumerate(res["ks"])}
    return [f"| {name} (step {step}) | {res['grid_test_acc']:.2f} | "
            + " | ".join(f"{res['excluded_loss'][at[k]]:.2f}" for k in cols) + " | "
            + " | ".join(f"{res['clock_r2'][at[k]]:.2f}" for k in cols) + f" | {res['order'][:5]} |"]


def main(argv=None):
    """run the cli.

    args:
        argv: argument list, or None for sys.argv.
    returns:
        exit code.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default="runs/mech-freq")
    ap.add_argument("--device", default="cuda" if t.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cols = [str(k) for k in SHOW]
    md = ["# readout frequencies on the hop-1 grid", "",
          "excluded loss: train-pair loss with the top k frequencies removed (high = the model needs them). "
          "clock r²: share of logit variance a clock of the answer explains with the top k frequencies.", "",
          "| run | grid test acc | " + " | ".join(f"excl k={k}" for k in cols) + " | "
          + " | ".join(f"r² k={k}" for k in cols) + " | top 5 |",
          "|---|---|" + "---|" * (2 * len(cols) + 1)]
    for rd in args.runs:
        run = analysis.load_run(rd, device=args.device)
        step, path = analysis.checkpoints(run)[-1]
        model = analysis.load_model(run, path, args.device)
        res = sweep(model, run, KS, args.device)
        (out / f"{run['name']}.json").write_text(json.dumps({"step": step, "T": run["T"], **res}))
        md += summarize(run["name"], step, res)
        print(f"done {run['name']}", flush=True)
    (out / "summary.md").write_text("\n".join(md) + "\n")
    print(f"wrote {out / 'summary.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
