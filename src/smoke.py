"""train each distinct config for a few steps and report peak gpu memory and a rough time per 20k steps.

    uv run python src/smoke.py --suite tier1 [--steps 30] [--device cuda]
    uv run python src/smoke.py --runs NAME [NAME ...]

runs that differ only in name and seed share a config, so only the first of each is tried. outputs go to a
temporary directory and are deleted afterwards.
"""

import argparse
import copy
import json
import sys
import tempfile
import time
from pathlib import Path

import torch as t
import yaml

import suite
import train as tr

# evals and sweeps in a 20k-step run with the usual settings (every 100 and 1000 steps, plus checkpoints)
_EVALS_PER_20K, _SWEEPS_PER_20K = 200, 40


def distinct(names, cfg):
    """keep the first run of each config, ignoring name and seed.

    args:
        names: run names.
        cfg: validated training config.
    returns:
        list of run entries, one per distinct config.
    """
    entries = {r["name"]: r for r in cfg["runs"]}
    seen, out = set(), []
    for name in names:
        key = json.dumps({k: v for k, v in entries[name].items() if k not in ("name", "seed")}, sort_keys=True)
        if key not in seen:
            seen.add(key)
            out.append(entries[name])
    return out


def smoke(entry, cfg, device, steps):
    """train one entry for a few steps in a scratch directory and time it.

    args:
        entry: run entry.
        cfg: training config (only data_root is used).
        device: torch device string.
        steps: training steps.
    returns:
        dict with peak_gb, step_s, eval_s, sweep_s and hours_20k, or error if training failed.
    """
    marks = []
    with tempfile.TemporaryDirectory() as tmp:
        scratch = {**copy.deepcopy(cfg), "output_root": tmp}
        if device.startswith("cuda"):
            t.cuda.empty_cache()
            t.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        try:
            tr.train(entry, scratch, device, max_steps=steps, log=lambda line: marks.append(time.perf_counter()))
        except (RuntimeError, tr.RunConfigError) as e:
            return {"error": str(e).splitlines()[0][:120]}
        end = time.perf_counter()
    # marks: after the step-0 eval, then after the last step's eval; the sweep and checkpoint follow
    eval_s = marks[0] - start
    step_s = max(marks[-1] - marks[0] - eval_s, 0.0) / steps
    sweep_s = end - marks[-1]
    hours = (20000 * step_s + _EVALS_PER_20K * eval_s + _SWEEPS_PER_20K * sweep_s) / 3600
    peak = t.cuda.max_memory_allocated() / 2 ** 30 if device.startswith("cuda") else None
    return {"peak_gb": None if peak is None else round(peak, 2), "step_s": round(step_s, 4),
            "eval_s": round(eval_s, 2), "sweep_s": round(sweep_s, 2), "hours_20k": round(hours, 2)}


def main(argv=None):
    """run the cli.

    args:
        argv: argument list, or None for sys.argv.
    returns:
        exit code: 0 all ran, 1 a config failed, 2 bad arguments or config.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/train.yaml")
    which = ap.add_mutually_exclusive_group(required=True)
    which.add_argument("--runs", nargs="+")
    which.add_argument("--suite", nargs="+", help="one or more suite names, run in the order given")
    ap.add_argument("--suites", default="configs/suites.yaml")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--device", default="cuda" if t.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)
    try:
        cfg = yaml.safe_load(Path(args.config).read_text())
        tr.validate_config(cfg)
        names = args.runs or [n for s in args.suite for n in suite.load_suite(args.suites, s)]
        suite.plan(names, cfg)
    except (OSError, yaml.YAMLError, tr.RunConfigError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print("| run | peak GB | s/step | s/eval | s/sweep | est. h per 20k |")
    print("|---|---|---|---|---|---|")
    failed = False
    for entry in distinct(names, cfg):
        r = smoke(entry, cfg, args.device, args.steps)
        if "error" in r:
            failed = True
            print(f"| `{entry['name']}` | failed: {r['error']} | | | | |", flush=True)
        else:
            print(f"| `{entry['name']}` | {r['peak_gb'] if r['peak_gb'] is not None else '—'} | {r['step_s']} | "
                  f"{r['eval_s']} | {r['sweep_s']} | {r['hours_20k']} |", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
