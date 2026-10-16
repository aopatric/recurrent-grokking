"""summarize runs as grokking verdicts, one table row per run, from their metrics.jsonl.

    uv run python src/verdicts.py runs/huginn_sampled_k1_p113_h3_x6 [runs/other ...] [--at 20000] [--json]
    uv run python src/verdicts.py --all [--root runs]

val is the mean val accuracy over the trained hops. the verdicts:
    groks        val >= 0.9 at some eval up to --at; t_grok is the first such step.
    stable       fraction of evals with val >= 0.9 in [t_grok + 1k, t_grok + 10k] (partial if the run ends sooner).
    slow         not grokked, but restricted loss fell by more than 0.1 over the last 5k steps: extend the run.
    fails        not grokked, with flat restricted loss.
    fits train   train >= 0.95 at the last eval up to --at (with val < 0.2 this is memorization).
    fixed point  for looped runs, val accuracy at pass 16 within 0.02 of the eval depth, at every hop.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

GROK = 0.9
FIT_TRAIN = 0.95
STABLE_WINDOW = (1000, 10000)
SLOW_DROP, SLOW_SPAN = 0.1, 5000
FIXED_T, FIXED_TOL = 16, 0.02


def _mean_acc(split):
    """mean accuracy over hops.

    args:
        split: dict from hop label to {"loss", "acc"}.
    returns:
        float.
    """
    return float(np.mean([v["acc"] for v in split.values()]))


def fixed_point(sweep):
    """whether a sweep row's val accuracy holds from the eval depth out to pass 16.

    args:
        sweep: a "sweep" metrics row.
    returns:
        True or False, or None if the sweep doesn't reach pass 16.
    """
    by_hop = sweep.get("by_T", {}).get("val")
    if not by_hop or any(len(accs) < FIXED_T for accs in by_hop.values()):
        return None
    d = sweep["depth"]
    return all(abs(accs[FIXED_T - 1] - accs[d - 1]) <= FIXED_TOL for accs in by_hop.values())


def verdict(rows, at=20000):
    """judge one run from its metrics rows.

    args:
        rows: metrics.jsonl rows, in step order.
        at: last step to judge grokking and memorization at.
    returns:
        dict with verdict, t_grok, stable, window_end, val, train, fits_train, restricted, fixed_point
        and last_step.
    """
    evals = [r for r in rows if r["kind"] == "eval"]
    steps = np.array([r["step"] for r in evals])
    val = np.array([_mean_acc(r["val"]) for r in evals])
    upto = steps <= at
    i_end = int(np.flatnonzero(upto)[-1])
    out = {"last_step": int(steps[-1]), "val": round(float(val[i_end]), 3),
           "train": round(_mean_acc(evals[i_end]["train"]), 3),
           "restricted": round(evals[i_end]["progress"]["restricted_loss"], 3),
           "fits_train": _mean_acc(evals[i_end]["train"]) >= FIT_TRAIN,
           "t_grok": None, "stable": None, "window_end": None}

    hit = np.flatnonzero(upto & (val >= GROK))
    if hit.size:
        t_grok = int(steps[hit[0]])
        lo, hi = t_grok + STABLE_WINDOW[0], t_grok + STABLE_WINDOW[1]
        win = (steps >= lo) & (steps <= hi)
        out.update(verdict="groks", t_grok=t_grok, window_end=int(min(hi, steps[-1])),
                   stable=round(float(np.mean(val[win] >= GROK)), 2) if win.any() else None)
    else:
        then = np.flatnonzero(steps <= steps[i_end] - SLOW_SPAN)
        drop = (evals[then[-1]]["progress"]["restricted_loss"] - out["restricted"]) if then.size else 0.0
        out["verdict"] = "slow" if drop > SLOW_DROP else "fails"

    sweeps = [r for r in rows if r["kind"] == "sweep" and r["step"] <= at]
    out["fixed_point"] = fixed_point(sweeps[-1]) if sweeps else None
    return out


def load_rows(run_dir):
    """read a run's metrics rows.

    args:
        run_dir: run directory.
    returns:
        list of rows.
    """
    with open(Path(run_dir) / "metrics.jsonl") as f:
        return [json.loads(line) for line in f]


def table(results):
    """render verdicts as a markdown table.

    args:
        results: dict from run name to verdict dict.
    returns:
        table string.
    """
    def show(v):
        """format one cell."""
        return "—" if v is None else "yes" if v is True else "no" if v is False else str(v)

    cols = ["verdict", "t_grok", "stable", "window_end", "val", "train", "fits_train", "restricted",
            "fixed_point", "last_step"]
    lines = ["| run | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for name, r in results.items():
        lines.append(f"| `{name}` | " + " | ".join(show(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def main(argv=None):
    """run the cli.

    args:
        argv: argument list, or None for sys.argv.
    returns:
        exit code: 0 ok, 2 no runs found.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="*")
    ap.add_argument("--all", action="store_true", help="every run under --root")
    ap.add_argument("--root", default="runs")
    ap.add_argument("--at", type=int, default=20000)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    dirs = [Path(d) for d in args.run_dirs]
    if args.all:
        dirs += sorted(p.parent for p in Path(args.root).glob("*/metrics.jsonl"))
    results = {d.name: verdict(load_rows(d), args.at) for d in dirs if (d / "metrics.jsonl").exists()}
    if not results:
        print("error: no runs with a metrics.jsonl", file=sys.stderr)
        return 2
    print(json.dumps(results, indent=2) if args.json else table(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
