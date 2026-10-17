"""summarize runs as grokking verdicts, one table row per run (or per config across seeds), from their logs.

    uv run python src/verdicts.py runs/huginn_sampled_k1_p113_h3_x6 [runs/other ...] [--at 20000] [--json]
    uv run python src/verdicts.py --all [--root runs] [--by-config]

val is the mean val accuracy over the trained hops. the verdicts:
    groks        val >= 0.9 at some eval up to --at; t_grok is the first such step.
    stable       fraction of evals with val >= 0.9 in [t_grok + 1k, t_grok + 10k] (partial if the run ends sooner).
    slow         not grokked, but restricted loss fell by more than 0.1 between the 1k-step windows ending
                 5k steps apart: extend the run.
    fails        not grokked, with flat restricted loss.
    diverged     not grokked, and the loss stopped being finite (training was stopped there).
    in progress  not grokked yet, and the run hasn't reached --at (still training, or paused early).
    fits train   train >= 0.95 at the last eval up to --at (with val < 0.2 this is memorization).
    fixed point  for looped runs, val accuracy at pass 16 within 0.02 of the eval depth, at every hop.
--by-config groups runs whose names differ only in their _s<seed> suffix.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

GROK = 0.9
FIT_TRAIN = 0.95
STABLE_WINDOW = (1000, 10000)
SLOW_DROP, SLOW_SPAN, SLOW_AVG = 0.1, 5000, 1000
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


def _restricted_drop(evals, steps, end):
    """how much restricted loss fell between the window ending at end and the one ending SLOW_SPAN earlier.

    args:
        evals: eval rows.
        steps: their steps, as an array.
        end: last step judged.
    returns:
        mean of the earlier window minus mean of the later one, or 0 if the run is too short.
    """
    def window(last):
        """restricted losses of the evals in (last - SLOW_AVG, last]."""
        sel = (steps > last - SLOW_AVG) & (steps <= last)
        return [evals[i]["progress"]["restricted_loss"] for i in np.flatnonzero(sel)]

    late, early = window(end), window(end - SLOW_SPAN)
    return float(np.mean(early) - np.mean(late)) if late and early else 0.0


def verdict(rows, at=20000, diverged_at=None):
    """judge one run from its metrics rows.

    args:
        rows: metrics.jsonl rows, in step order.
        at: last step to judge grokking and memorization at.
        diverged_at: step at which the run's loss stopped being finite, or None.
    returns:
        dict with verdict, t_grok, stable, window_end, val, train, fits_train, restricted, fixed_point,
        diverged_at and last_step.
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
           "t_grok": None, "stable": None, "window_end": None, "diverged_at": diverged_at}

    hit = np.flatnonzero(upto & (val >= GROK))
    if hit.size:
        t_grok = int(steps[hit[0]])
        lo, hi = t_grok + STABLE_WINDOW[0], t_grok + STABLE_WINDOW[1]
        win = (steps >= lo) & (steps <= hi)
        out.update(verdict="groks", t_grok=t_grok, window_end=int(min(hi, steps[-1])),
                   stable=round(float(np.mean(val[win] >= GROK)), 2) if win.any() else None)
    elif diverged_at is not None:
        out["verdict"] = "diverged"
    elif steps[-1] < at:
        out["verdict"] = "in progress"
    else:
        drop = _restricted_drop(evals, steps, int(steps[i_end]))
        out["verdict"] = "slow" if drop > SLOW_DROP else "fails"

    sweeps = [r for r in rows if r["kind"] == "sweep" and r["step"] <= at]
    out["fixed_point"] = fixed_point(sweeps[-1]) if sweeps else None
    order = ["verdict", "t_grok", "stable", "window_end", "val", "train", "fits_train", "restricted", "fixed_point",
             "diverged_at", "last_step"]
    return {k: out[k] for k in order}


def load_rows(run_dir):
    """read a run's metrics rows, dropping a last line cut off mid-write (e.g. copied while training).

    args:
        run_dir: run directory.
    returns:
        list of rows.
    raises:
        ValueError: if a line other than the last is not valid json.
    """
    lines = [ln for ln in (Path(run_dir) / "metrics.jsonl").read_text().splitlines() if ln.strip()]
    rows = []
    for i, ln in enumerate(lines):
        try:
            rows.append(json.loads(ln))
        except json.JSONDecodeError:
            if i < len(lines) - 1:
                raise ValueError(f"{run_dir}/metrics.jsonl: line {i + 1} is not valid json")
    return rows


def judge(run_dir, at=20000):
    """judge one run directory.

    args:
        run_dir: run directory.
        at: last step to judge at.
    returns:
        verdict dict.
    """
    try:
        div = json.loads((Path(run_dir) / "meta.json").read_text()).get("diverged_at")
    except (OSError, ValueError):
        div = None
    return verdict(load_rows(run_dir), at, div)


def config_name(run):
    """a run's name without its seed suffix.

    args:
        run: run name.
    returns:
        the name with any trailing _s<digits> removed.
    """
    return re.sub(r"_s\d+$", "", run)


def summarize(results):
    """combine per-run verdicts over the seeds of each config.

    args:
        results: dict from run name to verdict dict.
    returns:
        dict from config name to a summary: seeds, groks (k/n), t_grok_median, stable_mean, fixed_point (k/n
        of seeds where it was measured), and counts of slow, fails, diverged and in-progress runs.
    """
    groups = {}
    for name, r in results.items():
        groups.setdefault(config_name(name), []).append(r)
    out = {}
    for cfg, rs in groups.items():
        n = len(rs)
        grok_t = [r["t_grok"] for r in rs if r["t_grok"] is not None]
        stable = [r["stable"] for r in rs if r["stable"] is not None]
        fp = [r["fixed_point"] for r in rs if r["fixed_point"] is not None]
        out[cfg] = {
            "seeds": n,
            "groks": f"{len(grok_t)}/{n}",
            "t_grok_median": int(np.median(grok_t)) if grok_t else None,
            "stable_mean": round(float(np.mean(stable)), 2) if stable else None,
            "fixed_point": f"{sum(fp)}/{len(fp)}" if fp else None,
            "slow": sum(r["verdict"] == "slow" for r in rs),
            "fails": sum(r["verdict"] == "fails" for r in rs),
            "diverged": sum(r["verdict"] == "diverged" for r in rs),
            "in_progress": sum(r["verdict"] == "in progress" for r in rs),
            "val_median": round(float(np.median([r["val"] for r in rs])), 3),
        }
    return out


def table(results, first="run"):
    """render verdicts or summaries as a markdown table.

    args:
        results: dict from name to a dict of columns (every row has the same columns).
        first: header of the name column.
    returns:
        table string.
    """
    def show(v):
        """format one cell."""
        return "—" if v is None else "yes" if v is True else "no" if v is False else str(v)

    cols = list(next(iter(results.values())))
    lines = [f"| {first} | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
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
    ap.add_argument("--by-config", action="store_true", help="one row per config, combining its seeds")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    dirs = [Path(d) for d in args.run_dirs]
    if args.all:
        dirs += sorted(p.parent for p in Path(args.root).glob("*/metrics.jsonl"))
    results = {d.name: judge(d, args.at) for d in dirs if (d / "metrics.jsonl").exists()}
    if not results:
        print("error: no runs with a metrics.jsonl", file=sys.stderr)
        return 2
    if args.by_config:
        results = summarize(results)
    print(json.dumps(results, indent=2) if args.json else table(results, "config" if args.by_config else "run"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
