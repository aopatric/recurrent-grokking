"""check every split invariant on generated dataset directories.

    uv run python src/verify_data.py data/synthetic/p113_h3_x6 [...]
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import data_gen as dg

ROW_KEYS = {"hop", "operands", "tokens", "answer", "intermediates"}
EXTRA_KEYS = {"systematic.jsonl": {"novel_step"}, "systematic_extrap.jsonl": {"novel_step"},
              "grid_hop1.jsonl": {"split"}}


class _Report:
    """collect violations grouped by check, keeping a few examples of each."""

    def __init__(self):
        self.hits = defaultdict(list)

    def bad(self, check: str, detail) -> None:
        """record one violation.

        args:
            check: name of the violated check.
            detail: example that violates it.
        """
        self.hits[check].append(detail)

    def errors(self) -> list:
        """summarize the recorded violations.

        returns:
            one line per check, with up to three examples.
        """
        out = []
        for check, details in self.hits.items():
            shown = "; ".join(map(str, details[:3]))
            more = f" (+{len(details) - 3} more)" if len(details) > 3 else ""
            out.append(f"{check}: {shown}{more}")
        return out


def _load_jsonl(path: Path) -> list:
    """read a json lines file.

    args:
        path: file to read.
    returns:
        list of row dicts.
    """
    with open(path) as f:
        return [json.loads(line) for line in f]


def _check_row(rep, file, row, p):
    """check a row's schema and recompute its sums and tokens.

    args:
        rep: report to record violations in.
        file: name of the file the row came from.
        row: row dict.
        p: prime modulus.
    returns:
        True if the row is well-formed enough for the split checks.
    """
    # recompute independently of data_gen so a generator bug can't hide here
    expected = ROW_KEYS | EXTRA_KEYS.get(file, set())
    if set(row) != expected:
        rep.bad(f"{file}: keys", sorted(row))
        return False
    ops = row["operands"]
    if not (isinstance(ops, list) and len(ops) >= 2 and all(isinstance(x, int) and 0 <= x < p for x in ops)):
        rep.bad(f"{file}: operands", ops)
        return False
    sums, s = [], ops[0]
    for x in ops[1:]:
        s = (s + x) % p
        sums.append(s)
    toks = []
    for i, x in enumerate(ops):
        toks += (["+"] if i else []) + [str(x)]
    toks.append("=")
    if row["hop"] != len(ops) - 1:
        rep.bad(f"{file}: hop != len(operands)-1", ops)
    if row["intermediates"] != sums or row["answer"] != sums[-1]:
        rep.bad(f"{file}: wrong intermediates/answer", ops)
    if row["tokens"] != toks:
        rep.bad(f"{file}: wrong tokens", ops)
    return True


def _step_labels(ops, p, label):
    """classify each step pair of a chain.

    args:
        ops: chain operands.
        p: prime modulus.
        label: dict from pair to "train", "val" or "heldout".
    returns:
        list of h labels, one per step.
    """
    labs, s = [], ops[0]
    for x in ops[1:]:
        labs.append(label[(s, x)])
        s = (s + x) % p
    return labs


def verify_dir(path) -> list:
    """check every split invariant on one dataset directory, recomputing from rows and meta.json.

    args:
        path: dataset directory.
    returns:
        list of readable violations; empty means ok.
    """
    path = Path(path)
    rep = _Report()
    meta = json.loads((path / "meta.json").read_text())
    entry = meta["config"]
    p, tr_h, te_h = entry["p"], entry["max_train_hops"], entry["max_test_hops"]
    mh, ex, sy = entry["multihop"], entry["extrap"], entry["systematic"]

    vocab = json.loads((path / "vocab.json").read_text())
    if vocab != dg.make_vocab(p) or meta["vocab_size"] != len(vocab):
        rep.bad("vocab", "does not match make_vocab(p)")

    expected_files = {"train.jsonl", "val.jsonl", "grid_hop1.jsonl"}
    if tr_h >= 2:
        expected_files.add("systematic.jsonl")
    if te_h > tr_h:
        expected_files |= {"extrap_depth.jsonl", "systematic_extrap.jsonl"}
    present = {f.name for f in path.glob("*.jsonl")}
    if present != expected_files:
        rep.bad("file set", f"expected {sorted(expected_files)}, found {sorted(present)}")
    data = {f: _load_jsonl(path / f) for f in sorted(present & expected_files)}
    for f, rows in data.items():
        data[f] = [r for r in rows if _check_row(rep, f, r, p)]

    # grid and pair split
    grid = data.get("grid_hop1.jsonl", [])
    label = {tuple(r["operands"]): r["split"] for r in grid}
    if len(grid) != p * p or len(label) != p * p:
        rep.bad("grid", f"{len(grid)} rows, {len(label)} distinct pairs, expected {p * p}")
    split = dg.split_pairs(p, entry["hop1"]["train_frac"], entry["hop1"]["val_frac"], entry["seed"])
    if label != split.label():
        rep.bad("grid", "split labels differ from split_pairs(config)")
    pairs = {c: {pr for pr, lab in label.items() if lab == c} for c in ("train", "val", "heldout")}

    # per-file split rules
    for f, rows in data.items():
        if f == "grid_hop1.jsonl":
            continue
        tuples = [tuple(r["operands"]) for r in rows]
        if len(set(tuples)) != len(tuples):
            rep.bad(f"{f}: duplicate rows", [t for t, n in Counter(tuples).items() if n > 1][:3])
        counts = Counter()
        for r in rows:
            h, ops = r["hop"], r["operands"]
            labs = _step_labels(ops, p, label)
            if "val" in labs and h >= 2:
                rep.bad(f"{f}: val pair inside a chain", ops)
            if f in ("train.jsonl", "val.jsonl"):
                if not 1 <= h <= tr_h:
                    rep.bad(f"{f}: hop out of range", ops)
                want = ("train" if f == "train.jsonl" else "val") if h == 1 else "train"
                if any(lab != want for lab in labs):
                    rep.bad(f"{f}: step pair not {want}", ops)
                counts[h] += 1
            elif f == "extrap_depth.jsonl":
                if not tr_h < h <= te_h:
                    rep.bad(f"{f}: hop out of range", ops)
                if any(lab != "train" for lab in labs):
                    rep.bad(f"{f}: chain not closed", ops)
                counts[h] += 1
            else:  # systematic files
                lo, hi = (2, tr_h) if f == "systematic.jsonl" else (tr_h + 1, te_h)
                if not lo <= h <= hi:
                    rep.bad(f"{f}: hop out of range", ops)
                k = r["novel_step"]
                held = [t for t, lab in enumerate(labs, 1) if lab == "heldout"]
                others_ok = all(lab == "train" for t, lab in enumerate(labs, 1) if t != k)
                if held != [k] or not others_ok:
                    rep.bad(f"{f}: not exactly one held-out step at novel_step", (ops, k, labs))
                counts[(h, k)] += 1

        if f == "train.jsonl":
            if {tuple(r["operands"]) for r in rows if r["hop"] == 1} != pairs["train"]:
                rep.bad(f, "hop-1 rows != train pairs")
            want = {h: mh["train_per_hop"] for h in range(2, tr_h + 1)} | {1: len(pairs["train"])}
        elif f == "val.jsonl":
            if {tuple(r["operands"]) for r in rows if r["hop"] == 1} != pairs["val"]:
                rep.bad(f, "hop-1 rows != val pairs")
            want = {h: mh["val_per_hop"] for h in range(2, tr_h + 1)} | {1: len(pairs["val"])}
        elif f == "extrap_depth.jsonl":
            want = {h: ex["per_hop"] for h in range(tr_h + 1, te_h + 1)}
        else:
            lo, hi = (2, tr_h) if f == "systematic.jsonl" else (tr_h + 1, te_h)
            want = {(h, k): n for h in range(lo, hi + 1)
                    for k, n in enumerate(dg.systematic_counts(sy["per_hop"], h), 1)}
        if dict(counts) != want:
            rep.bad(f"{f}: counts", f"got {dict(sorted(counts.items()))}, want {want}")

    # cross-file disjointness; the grid is excluded because it is the whole hop-1 table
    owner = {}
    for f, rows in data.items():
        if f == "grid_hop1.jsonl":
            continue
        for r in rows:
            t = tuple(r["operands"])
            if t in owner and owner[t] != f:
                rep.bad("cross-file overlap", (t, owner[t], f))
            owner[t] = f

    # meta agrees with the files
    for s in meta["sets"]:
        rows = data.get(s["file"], [])
        n = sum(1 for r in rows if r["hop"] == s["hop"] and r.get("novel_step") == s["novel_step"])
        if n != s["n"]:
            rep.bad("meta n", (s["file"], s["hop"], s["novel_step"], n, s["n"]))

    return rep.errors()


def main(argv=None) -> int:
    """run the cli.

    args:
        argv: dataset directories, or None for sys.argv.
    returns:
        exit code: 0 all ok, 1 violations found, 2 no arguments.
    """
    dirs = (argv if argv is not None else sys.argv[1:])
    if not dirs:
        print(__doc__, file=sys.stderr)
        return 2
    status = 0
    for d in dirs:
        errs = verify_dir(d)
        if errs:
            status = 1
            print(f"FAIL {d}")
            for e in errs:
                print(f"  {e}")
        else:
            n = sum(1 for f in Path(d).glob("*.jsonl") for _ in open(f))
            print(f"OK   {d} ({n:,} rows checked)")
    return status


if __name__ == "__main__":
    sys.exit(main())
