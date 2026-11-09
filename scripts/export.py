"""copy the runs behind the write-up's figures from local run folders into results/, trimmed to what the figures read.

    uv run python scripts/export.py [--sources vast-runs runs runs/branches] [--out results]
    uv run python scripts/export.py --list 5090|4090     the planned training runs that ran on that gpu

every run keeps config.yaml, meta.json and metrics.jsonl unchanged. runs with step-level figures also get
steps.csv (step, depth, loss and grad_norm from steps.jsonl). the mechanism outputs are copied from runs/mech-*,
and every run they analyse is exported with the checkpoints they read (the steps recorded in each output), so the
analyses can be re-run from results/ alone; the branch source keeps its step-8944 checkpoint. every file written is
read back and checked against its source.
"""

import argparse
import csv
import filecmp
import json
import math
import shutil
import sys
from pathlib import Path

SEED = 41
ABLATIONS = [
    "t1_anchor_nope_ref", "t2_ut_T2", "t1_ut_ref", "t1_unshared_L8", "t2_ut_statenorm_ref", "t1_ut_sandwich",
    "t2_huginn_nosandwich", "t1_huginn_ref", "t1_huginn_embedstart", "t1_huginn_noinject", "t1_huginn_noprecoda",
    "t2_huginn_wd0", "t2_huginn_wd05", "t3_huginn_wd2",
]
BRANCH_SOURCE = "t1_huginn_ref"
BRANCH_STEP = 8944
BRANCH_ARMS = ["control", "wd0", "wd05", "wd2", "wd2_lr05", "lrdecay", "clip", "clip_lrdecay", "nowd_embed"]
RECIPES = ["t6_ut_sandwich_twophase", "t5_ut_sandwich_const"]
# the unnormalized loop with input injection or with a prelude and coda, for the grokking outcomes figure
OUTCOMES = ["t1_ut_inject", "t2_ut_precoda"]
# the sandwich loop and the unshared 8- and 2-layer stacks on nanda's recipe, at both seeds, for the instability figure
STACKS = ["t5_ut_sandwich_const", "t5_unshared_sandwich_const", "t5_unshared_sandwich_L2_const"]
MECHANISM = ["t6_huginn_twophase", "t6_ut_statenorm_twophase", "t6_ut_sandwich_twophase"]
# the fixes figure's seed-42 pairs: each two-phase run and its constant-lr twin, the same run until the switch
TWINS_S42 = ["t6_ut_sandwich_twophase", "t5_ut_sandwich_const", "t6_huginn_twophase", "t1_huginn_ref"]
ANCHOR = "t1_anchor_nope_ref"
FIXED = "t1_huginn_fixedT8"
VARIANTS = ["t1_huginn_ref", "t1_huginn_embedstart", "t2_huginn_fullbackprop", "t1_huginn_mindepth2"]
# the state-norm loop at a fixed depth of 8 and at a random depth, run past their training depth
RANDOM_DEPTH = ["t2_ut_statenorm_ref", "t2_ut_statenorm_sampled"]
RANDOM_DEPTH_EARLIER = 18915
# the huginn reference partly recovered from a collapse, where it holds wrong answers as a fixed point
FIXED_POINT_STEP = 6968
SEED_2 = 42
COPIED = ("config.yaml", "meta.json", "metrics.jsonl")
STEP_FIELDS = ("step", "depth", "loss", "grad_norm")


def run_name(config, seed=SEED):
    """full run name of a config at a seed.

    args:
        config: config name without the data and seed suffix.
        seed: seed.
    returns:
        run name.
    """
    return f"{config}_p113_h3_x6_s{seed}"


def branch_name(arm):
    """run name of one branch arm.

    args:
        arm: arm name from src/branch.py.
    returns:
        run name.
    """
    return f"{run_name(BRANCH_SOURCE)}_from{BRANCH_STEP}_{arm}"


def both_seeds(configs):
    """run names of configs at both seeds.

    args:
        configs: config names.
    returns:
        list of run names, seed 41 first.
    """
    return [run_name(c) for c in configs] + [run_name(c, SEED_2) for c in configs]


def gpu_of(name):
    """the gpu a planned training run was trained on: the seed-41 ablation sweep (tiers 1-3) on rtx 5090s, every
    later run on an rtx 4090. tests/test_export.py checks this against every committed meta.json.

    args:
        name: run name (not a branch arm; those ran on the 4090).
    returns:
        "5090" or "4090".
    """
    return "5090" if name.startswith(("t1_", "t2_", "t3_")) and name.endswith(f"_s{SEED}") else "4090"


def runs_on(gpu):
    """the planned training runs (not the branch arms) that were trained on a gpu, in plan order.

    args:
        gpu: "5090" or "4090".
    returns:
        list of run names.
    """
    runs, _, _ = plan()
    return [r for r in runs if "_from" not in r and gpu_of(r) == gpu]


def mech_plan():
    """the mechanism outputs to export.

    returns:
        dict from the name under results/mechanism/ to (folder under the mech root, run names it must hold).
    """
    return {
        "passes": ("mech-passes", [run_name(c) for c in MECHANISM]),
        "past-depth": ("mech-long", [run_name(c) for c in MECHANISM]),
        "passes-s42": ("mech-passes-s42", [run_name(c, SEED_2) for c in MECHANISM]),
        "past-depth-s42": ("mech-long-s42", [run_name(c, SEED_2) for c in MECHANISM]),
        "past-depth-fixed": ("mech-long-fixed", both_seeds([FIXED])),
        "frequencies": ("mech-freq", both_seeds([ANCHOR] + MECHANISM)),
        "state-use-twophase": ("mech-state-twophase", both_seeds([MECHANISM[0]])),
        "state-use-fixed": ("mech-state-fixed", both_seeds([FIXED])),
        "state-use-variants": ("mech-state-variants", [run_name(c) for c in VARIANTS]
                               + [run_name(VARIANTS[0], SEED_2)]),
        # the seed-42 reference matches the two-phase huginn until its switch, so it is the constant-lr branch
        "state-use-constant-lr": ("mech-state-ref-late", [run_name(VARIANTS[0], SEED_2)]),
        # the state-norm loop didn't grok at seed 42
        "internals": ("mech-internals", both_seeds([ANCHOR, MECHANISM[0], MECHANISM[2]]) + [run_name(MECHANISM[1])]),
        "past-depth-random-depth": ("mech-long-random-depth", [run_name(c) for c in RANDOM_DEPTH]),
        f"past-depth-random-depth-{RANDOM_DEPTH_EARLIER}": (f"mech-long-random-depth-{RANDOM_DEPTH_EARLIER}",
                                                         [run_name(c) for c in RANDOM_DEPTH]),
        f"past-depth-ref-{FIXED_POINT_STEP}": (f"mech-long-ref-{FIXED_POINT_STEP}", [run_name(VARIANTS[0])]),
        f"state-use-ref-{FIXED_POINT_STEP}": (f"mech-state-ref-{FIXED_POINT_STEP}", [run_name(VARIANTS[0])]),
    }


def plan():
    """the runs to export, before the checkpoints the mechanism outputs read are added.

    returns:
        (runs, step_runs, checkpoints): run names in order, runs that get steps.csv, and checkpoint file names
        per run.
    """
    runs = [run_name(c) for c in ABLATIONS + OUTCOMES] + [branch_name(a) for a in BRANCH_ARMS]
    runs += [run_name(c) for c in RECIPES + MECHANISM] + both_seeds(STACKS)
    runs += [name for _, names in mech_plan().values() for name in names]
    runs = list(dict.fromkeys(runs))
    step_runs = [run_name(BRANCH_SOURCE)] + [branch_name(a) for a in BRANCH_ARMS] + [run_name(c) for c in RECIPES]
    step_runs += both_seeds(STACKS) + [run_name(c, SEED_2) for c in TWINS_S42]
    step_runs = list(dict.fromkeys(step_runs))
    checkpoints = {run_name(BRANCH_SOURCE): [f"step_{BRANCH_STEP:07d}.pt"]}
    return runs, step_runs, checkpoints


def steps_read(path):
    """the checkpoint steps a mechanism output read.

    args:
        path: one output json (a single "step", or a "checkpoints" list of rows with a "step").
    returns:
        list of steps.
    """
    data = json.loads(Path(path).read_text())
    return [c["step"] for c in data["checkpoints"]] if "checkpoints" in data else [data["step"]]


def checkpoint_file(run_dir, step):
    """the checkpoint file holding one step of a run, as analysis.checkpoints names them.

    args:
        run_dir: run directory.
        step: training step.
    returns:
        file name inside checkpoints/.
    raises:
        FileNotFoundError: if the run has no checkpoint at that step.
    """
    ck = Path(run_dir) / "checkpoints"
    name = f"step_{step:07d}.pt"
    if (ck / name).exists():
        return name
    meta = json.loads((Path(run_dir) / "meta.json").read_text())
    if (ck / "final.pt").exists() and meta["steps"] == step:
        return "final.pt"
    raise FileNotFoundError(f"{run_dir}: no checkpoint at step {step}")


def find_run(name, sources):
    """the one source folder holding a run.

    args:
        name: run name.
        sources: folders to look in.
    returns:
        path to the run directory.
    raises:
        FileNotFoundError: if no source has the run.
        ValueError: if more than one source has it.
    """
    hits = [Path(s) / name for s in sources if (Path(s) / name / "metrics.jsonl").exists()]
    if not hits:
        raise FileNotFoundError(f"{name}: not found in {', '.join(map(str, sources))}")
    if len(hits) > 1:
        raise ValueError(f"{name}: found in more than one source: {', '.join(map(str, hits))}")
    return hits[0]


def _parse(field, text):
    """read one steps.csv value back.

    args:
        field: column name.
        text: cell text.
    returns:
        int, float or None.
    """
    if text == "":
        return None
    return int(text) if field in ("step", "depth") else float(text)


def _same(a, b):
    """equality that treats two nans as equal.

    args:
        a, b: values.
    returns:
        bool.
    """
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


def write_steps(src, dst):
    """write steps.csv from steps.jsonl, then check every value against the source.

    args:
        src: steps.jsonl path.
        dst: steps.csv path.
    returns:
        number of rows.
    raises:
        ValueError: if a value read back differs from the source.
    """
    rows = [json.loads(ln) for ln in src.read_text().splitlines() if ln.strip()]
    with open(dst, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(STEP_FIELDS)
        # repr keeps every float exact
        w.writerows([["" if r.get(k) is None else repr(r[k]) for k in STEP_FIELDS] for r in rows])
    with open(dst, newline="") as f:
        back = list(csv.DictReader(f))
    if len(back) != len(rows):
        raise ValueError(f"{dst}: {len(back)} rows written, {len(rows)} in the source")
    for r, b in zip(rows, back):
        for k in STEP_FIELDS:
            if not _same(r.get(k), _parse(k, b[k])):
                raise ValueError(f"{dst}: step {r.get('step')}, {k} differs from the source")
    return len(rows)


def copy_checked(src, dst):
    """copy a file and check the copy is byte-identical.

    args:
        src: source path.
        dst: destination path.
    returns:
        size in bytes.
    raises:
        ValueError: if the copy differs.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    if not filecmp.cmp(src, dst, shallow=False):
        raise ValueError(f"{dst}: copy differs from {src}")
    return dst.stat().st_size


def export(sources, out, mech_root, log=print):
    """export every planned run and mechanism output into out.

    args:
        sources: folders holding run directories, searched in order.
        out: output folder (its runs/ and mechanism/ are rewritten).
        mech_root: folder holding the mech-* output folders.
        log: print function.
    returns:
        total bytes written.
    """
    target = (Path(out) / "runs").resolve()
    for s in sources:
        if Path(s).resolve() in (target, *target.parents) or target in Path(s).resolve().parents:
            raise ValueError(f"source {s} overlaps {target}, which the export replaces")
    runs, step_runs, checkpoints = plan()
    found = {name: find_run(name, sources) for name in runs}
    mech = {(k, name): Path(mech_root) / folder / f"{name}.json"
            for k, (folder, names) in mech_plan().items() for name in names}
    missing = [str(p) for p in mech.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("missing: " + ", ".join(missing))
    for (_, name), path in mech.items():
        for step in steps_read(path):
            c = checkpoint_file(found[name], step)
            checkpoints.setdefault(name, [])
            if c not in checkpoints[name]:
                checkpoints[name].append(c)
    missing += [str(found[r] / "checkpoints" / c) for r, cs in checkpoints.items() for c in cs
                if not (found[r] / "checkpoints" / c).exists()]
    missing += [str(found[r] / "steps.jsonl") for r in step_runs if not (found[r] / "steps.jsonl").exists()]
    if missing:
        raise FileNotFoundError("missing: " + ", ".join(missing))

    out = Path(out)
    for sub in ("runs", "mechanism"):
        shutil.rmtree(out / sub, ignore_errors=True)
    total = 0
    for name, src in found.items():
        dst = out / "runs" / name
        for f in COPIED:
            total += copy_checked(src / f, dst / f)
        if name in step_runs:
            n = write_steps(src / "steps.jsonl", dst / "steps.csv")
            total += (dst / "steps.csv").stat().st_size
            log(f"{name}: steps.csv, {n} rows")
        for c in checkpoints.get(name, []):
            total += copy_checked(src / "checkpoints" / c, dst / "checkpoints" / c)
            log(f"{name}: checkpoints/{c}")
    for (k, name), src in mech.items():
        total += copy_checked(src, out / "mechanism" / k / src.name)
    log(f"{len(found)} runs and {len(mech)} mechanism outputs, {total / 1e6:.1f} MB in {out}")
    return total


def main(argv=None):
    """run the cli.

    args:
        argv: argument list, or None for sys.argv.
    returns:
        exit code: 0 ok.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", nargs="+", default=["vast-runs", "runs", "runs/branches"])
    ap.add_argument("--mech-root", default="runs", help="folder holding the mech-* output folders")
    ap.add_argument("--out", default="results")
    ap.add_argument("--list", choices=["5090", "4090"],
                    help="print the planned training runs that were trained on this gpu, one per line, and exit")
    args = ap.parse_args(argv)
    if args.list:
        print("\n".join(runs_on(args.list)))
        return 0
    export(args.sources, args.out, args.mech_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
