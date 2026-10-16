"""live tensorboard logging of a run's metrics rows, and backfill from an existing metrics.jsonl.

    uv run python src/tb_log.py runs/anchor_p113_h1 [runs/other ...]
    uv run tensorboard --logdir runs

event files go in <run_dir>/tb.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

# charts that overlay several tags; tensorboard otherwise draws one tag per chart
LAYOUT = {
    "progress": {
        "hop-1 losses": ["Multiline", [r"progress/grid_train_loss", r"progress/grid_test_loss",
                                       r"progress/restricted_loss", r"progress/excluded_loss"]],
        "fourier sparsity (gini)": ["Multiline", [r"progress/gini_.*"]],
    },
    "accuracy": {
        "val by hop": ["Multiline", [r"val_acc/.*"]],
        "train by hop": ["Multiline", [r"train_acc/.*"]],
        "held-out sets": ["Multiline", [r"(extrap_depth|systematic|systematic_extrap)_acc/.*"]],
    },
    "per step": {
        "update ratio by group": ["Multiline", [r"update_ratio/.*"]],
        "weight norm by group": ["Multiline", [r"weight_norm/.*"]],
        "gradient norm by group": ["Multiline", [r"grad_norm/.*"]],
    },
}


def open_writer(run_dir, start=0, resume=False):
    """open the event writer for a run.

    args:
        run_dir: run directory.
        start: step training resumes from.
        resume: whether this continues an earlier run; its events logged after start are dropped.
    returns:
        SummaryWriter writing to run_dir/tb.
    """
    # events at start itself were logged before the resume point was saved, so they stay
    writer = SummaryWriter(Path(run_dir) / "tb", purge_step=start + 1 if resume else None, flush_secs=30)
    if not resume:
        writer.add_custom_scalars(LAYOUT)
    return writer


def scalars(row):
    """flatten one metrics row into tensorboard tags; list-valued fields are skipped.

    args:
        row: an "eval" or "sweep" row from metrics.jsonl.
    returns:
        dict from tag to float.
    """
    out = {}
    if row["kind"] == "eval":
        if row["train_step_loss"] is not None:
            out["train/step_loss"] = row["train_step_loss"]
        out["train/lr"] = row["lr"]
        for split in ("train", "val"):
            for hop, v in row[split].items():
                out[f"{split}_acc/h{hop}"] = v["acc"]
                out[f"{split}_loss/h{hop}"] = v["loss"]
        for key, v in row["progress"].items():
            if isinstance(v, (int, float)):
                out[f"progress/{key}"] = v
    else:
        for name, groups in row["at_eval_depth"].items():
            for label, v in groups.items():
                out[f"{name}_acc/h{label}"] = v["acc"]
                out[f"{name}_loss/h{label}"] = v["loss"]
    return out


def write_row(writer, row):
    """log one metrics row.

    args:
        writer: SummaryWriter.
        row: metrics row.
    """
    for tag, v in scalars(row).items():
        writer.add_scalar(tag, v, row["step"])


def write_step(writer, row, every=10):
    """log one per-step row: depth, loss and gradient norm every step, group norms every few steps.

    args:
        writer: SummaryWriter.
        row: a steps.jsonl row.
        every: interval for the per-group scalars.
    """
    s = row["step"]
    writer.add_scalar("step/depth", row["depth"], s)
    writer.add_scalar("step/loss", row["loss"], s)
    writer.add_scalar("step/grad_norm", row["grad_norm"], s)
    if s % every == 0:
        for group, v in row["groups"].items():
            writer.add_scalar(f"weight_norm/{group}", v["w"], s)
            writer.add_scalar(f"grad_norm/{group}", v["grad"], s)
            writer.add_scalar(f"update_ratio/{group}", v["update"], s)


def backfill(run_dir):
    """rebuild a run's event files from its metrics.jsonl and steps.jsonl, replacing any that exist.

    args:
        run_dir: run directory.
    returns:
        number of rows written.
    """
    run_dir = Path(run_dir)
    shutil.rmtree(run_dir / "tb", ignore_errors=True)
    with open(run_dir / "metrics.jsonl") as f:
        rows = [json.loads(line) for line in f]
    steps = []
    if (run_dir / "steps.jsonl").exists():
        with open(run_dir / "steps.jsonl") as f:
            steps = [json.loads(line) for line in f]
    writer = open_writer(run_dir)
    try:
        for row in rows:
            write_row(writer, row)
        for row in steps:
            write_step(writer, row)
    finally:
        writer.close()
    return len(rows) + len(steps)


def main(argv=None):
    """run the cli.

    args:
        argv: argument list, or None for sys.argv.
    returns:
        exit code: 0 ok, 2 missing metrics.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="+")
    args = ap.parse_args(argv)
    for run_dir in args.run_dirs:
        try:
            n = backfill(run_dir)
        except FileNotFoundError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        print(f"{run_dir}: wrote {n} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
