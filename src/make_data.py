"""generate modular-addition chain datasets from a yaml config.

    uv run python src/make_data.py --name p113_h3_x6 [--dry-run]
    uv run python src/make_data.py --all

paths in the config are relative to the working directory.
"""

import argparse
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

import data_gen as dg


def git_info() -> dict | None:
    """read the current git commit and whether the tree has uncommitted changes.

    returns:
        {"commit": sha, "dirty": bool}, or None outside a git checkout.
    """
    here = Path(__file__).resolve().parent
    try:
        run = lambda *a: subprocess.run(  # noqa: E731
            ["git", *a], cwd=here, capture_output=True, text=True, check=True
        ).stdout.strip()
        return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}
    except (OSError, subprocess.CalledProcessError):
        return None


def dry_run_table(entry: dict) -> tuple:
    """render the per-block space table for --dry-run.

    args:
        entry: validated dataset entry.
    returns:
        (printable table, True if every block is feasible).
    """
    p, msf = entry["p"], entry["multihop"]["max_space_frac"]
    split, specs = dg.plan(entry)
    lines = [
        f"{entry['name']}: p={p}, train hops 1..{entry['max_train_hops']}, "
        f"test hops ..{entry['max_test_hops']}, max_space_frac={msf}",
        f"  {'file':<24}{'hop':>4}{'k':>3}{'raw_space':>24}{'space':>22}{'requested':>11}{'used':>8}  ok",
    ]
    for file, n in [("train.jsonl", len(split.train)), ("val.jsonl", len(split.val))]:
        lines.append(f"  {file:<24}{1:>4}{'-':>3}{p * p:>24,}{p * p:>22,}{n:>11,}{n / (p * p):>8.1%}  OK")
    ok = True
    for s in specs:
        good = s.feasible(msf)
        ok &= good
        used = f"{s.budget / s.space:.1%}" if s.space else "inf"
        lines.append(
            f"  {s.file:<24}{s.hop:>4}{s.novel_step or '-':>3}{s.raw_space:>24,}{s.space:>22,}"
            f"{s.n:>11,}{used:>8}  {'OK' if good else 'FAIL'}"
        )
    return "\n".join(lines), ok


def write_dataset(out_dir: Path, files: dict, vocab: dict, meta: dict) -> None:
    """write a dataset atomically.

    args:
        out_dir: final output directory; must not exist.
        files: dict from filename to rows.
        vocab: token vocabulary.
        meta: run metadata.
    """
    # write into a hidden temp dir, then rename, so a failed run never leaves a partial out_dir
    tmp = out_dir.parent / f".{out_dir.name}.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    try:
        for name, rows in files.items():
            with open(tmp / name, "w") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")
        (tmp / "vocab.json").write_text(json.dumps(vocab, indent=2) + "\n")
        (tmp / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        tmp.rename(out_dir)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def main(argv=None) -> int:
    """run the cli.

    args:
        argv: argument list, or None for sys.argv.
    returns:
        exit code: 0 ok, 1 infeasible, 2 bad config or existing output.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/data.yaml")
    which = ap.add_mutually_exclusive_group(required=True)
    which.add_argument("--name")
    which.add_argument("--all", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    try:
        cfg = yaml.safe_load(Path(args.config).read_text())
        dg.validate_config(cfg)
    except (OSError, yaml.YAMLError, dg.ConfigError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    entries = cfg["datasets"]
    if args.name:
        entries = [d for d in entries if d["name"] == args.name]
        if not entries:
            print(f"error: no dataset named {args.name!r} in {args.config}", file=sys.stderr)
            return 2

    root = Path(cfg["output_root"])
    status = 0
    for entry in entries:
        if args.dry_run:
            table, ok = dry_run_table(entry)
            print(table)
            status = status or (0 if ok else 1)
            continue

        out_dir = root / entry["name"]
        if out_dir.exists():
            # --all skips existing outputs so adding an entry never breaks a rerun
            if args.all:
                print(f"skip {entry['name']}: {out_dir} exists")
                continue
            print(f"error: {out_dir} exists; delete it or add a new config entry", file=sys.stderr)
            return 2

        try:
            files, vocab, meta = dg.generate(entry)
        except dg.ConfigError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        meta["git"] = git_info()
        meta["python"] = platform.python_version()
        write_dataset(out_dir, files, vocab, meta)
        counts = ", ".join(f"{f}={len(r)}" for f, r in files.items())
        print(f"wrote {out_dir}: {counts}")
    return status


if __name__ == "__main__":
    sys.exit(main())
