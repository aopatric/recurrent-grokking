"""tests for the modular-addition data generator, cli and verifier."""

import itertools
import json
import random

import pytest
import yaml

import data_gen as dg
import make_data
import verify_data


def small_entry(**over) -> dict:
    """build a small, feasible p=11 dataset entry.

    args:
        **over: top-level keys to override.
    returns:
        dataset entry dict.
    """
    e = {
        "name": "t_p11",
        "p": 11,
        "ops": ["+"],
        "max_train_hops": 3,
        "max_test_hops": 4,
        "seed": 0,
        "hop1": {"train_frac": 0.30, "val_frac": 0.10},
        "multihop": {"train_per_hop": 20, "val_per_hop": 5, "max_space_frac": 0.5},
        "extrap": {"per_hop": 20},
        "systematic": {"per_hop": 12},
    }
    e.update(over)
    return e


def write_config(tmp_path, *entries) -> str:
    """write a config whose output_root is inside tmp_path.

    args:
        tmp_path: directory for the config and outputs.
        *entries: dataset entries.
    returns:
        path to the written config.
    """
    cfg = {"output_root": str(tmp_path / "out"), "datasets": list(entries)}
    path = tmp_path / "data.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return str(path)


@pytest.fixture
def generated(tmp_path):
    """generate the small dataset and return its directory."""
    cfg = write_config(tmp_path, small_entry())
    assert make_data.main(["--config", cfg, "--name", "t_p11"]) == 0
    return tmp_path / "out" / "t_p11"


def step_classes(ops, p, split):
    """classify each step pair of a chain.

    args:
        ops: chain operands.
        p: prime modulus.
        split: PairSplit.
    returns:
        list of "train", "val" or "heldout", one per step.
    """
    lab, s, out = split.label(), ops[0], []
    for x in ops[1:]:
        out.append(lab[(s, x)])
        s = (s + x) % p
    return out


def chi_square_bound(df: int) -> float:
    """loose upper bound on a chi-square statistic under the null.

    args:
        df: degrees of freedom.
    returns:
        threshold roughly six standard deviations above the mean.
    """
    return df + 6 * (2 * df) ** 0.5 + 10


# validation

@pytest.mark.parametrize("over, msg", [
    ({"p": 12}, "prime"),
    ({"p": True}, "prime"),
    ({"ops": ["+", "*"]}, "ops"),
    ({"max_train_hops": 0}, "max_train_hops"),
    ({"max_test_hops": 2}, "max_test_hops"),
    ({"hop1": {"train_frac": 0.9, "val_frac": 0.1}}, "< 1"),
    ({"hop1": {"train_frac": 0.0, "val_frac": 0.1}}, "(0, 1)"),
    ({"multihop": {"train_per_hop": 0, "val_per_hop": 5, "max_space_frac": 0.5}}, "train_per_hop"),
    ({"multihop": {"train_per_hop": 20, "val_per_hop": 5, "max_space_frac": 1.5}}, "max_space_frac"),
    ({"extrap": {"per_hop": True}}, "extrap.per_hop"),
    ({"systematic": {"per_hop": 3}}, "every step position"),
    ({"extra_key": 1}, "unknown keys"),
    ({"hop1": {"train_frac": 0.3}}, "missing keys"),
])
def test_validate_rejects(over, msg):
    """each invalid field is rejected with a message naming it."""
    with pytest.raises(dg.ConfigError, match=msg.replace("(", r"\(").replace(")", r"\)")):
        dg.validate_entry(small_entry(**over))


def test_validate_config_rejects_duplicate_names():
    """two entries with the same name are rejected."""
    with pytest.raises(dg.ConfigError, match="duplicate"):
        dg.validate_config({"output_root": "x", "datasets": [small_entry(), small_entry()]})


def test_repo_config_is_valid():
    """the checked-in config passes validation."""
    cfg = yaml.safe_load(open("configs/data.yaml"))
    dg.validate_config(cfg)


# hop-1 split

@pytest.mark.parametrize("p, seed", [(5, 0), (7, 1), (11, 2)])
def test_split_partitions_pair_table(p, seed):
    """train, val and held-out pairs are disjoint, sized by fraction and cover the table."""
    sp = dg.split_pairs(p, 0.3, 0.1, seed)
    tr, va, ho = set(sp.train), set(sp.val), set(sp.heldout)
    assert len(tr) == len(sp.train) == int(0.3 * p * p)
    assert len(va) == len(sp.val) == int(0.1 * p * p)
    assert not (tr & va or tr & ho or va & ho)
    assert tr | va | ho == set(itertools.product(range(p), repeat=2))


def test_split_independent_of_hop_settings():
    """the pair split depends only on p, seed and fractions, not on hop settings."""
    a = dg.plan(small_entry())[0]
    b = dg.plan(small_entry(max_train_hops=1, max_test_hops=1))[0]
    assert a == b
    assert dg.split_pairs(11, 0.3, 0.1, 1) != a


# exact counting

@pytest.mark.parametrize("seed", range(4))
def test_dp_matches_brute_force(seed):
    """dp counts equal brute-force enumeration for closed and systematic chains at p=5."""
    p = 5
    sp = dg.split_pairs(p, 0.3, 0.1, seed)
    trp, hop = dg.partner_table(sp.train, p), dg.partner_table(sp.heldout, p)
    for h in range(1, 4):
        closed, by_k = 0, [0] * (h + 1)
        for ops in itertools.product(range(p), repeat=h + 1):
            cls = step_classes(ops, p, sp)
            if all(c == "train" for c in cls):
                closed += 1
            held = [t for t, c in enumerate(cls, 1) if c == "heldout"]
            if len(held) == 1 and cls.count("train") == h - 1:
                by_k[held[0]] += 1
        assert dg.chain_space(dg.count_completions(dg.step_tables(h, trp, hop), p)) == closed
        for k in range(1, h + 1):
            C = dg.count_completions(dg.step_tables(h, trp, hop, k), p)
            assert dg.chain_space(C) == by_k[k], (h, k)


def test_empty_space_is_handled(tmp_path, capsys):
    """an empty chain space is reported as infeasible instead of crashing."""
    # one train pair at p=5: most partial sums have no partners
    e = small_entry(p=5, hop1={"train_frac": 0.04, "val_frac": 0.04},
                    max_train_hops=2, max_test_hops=2, systematic={"per_hop": 2})
    _, specs = dg.plan(e)
    assert any(s.space == 0 for s in specs)
    with pytest.raises(dg.ConfigError, match="infeasible"):
        dg.generate(e)
    cfg = write_config(tmp_path, e)
    assert make_data.main(["--config", cfg, "--name", "t_p11", "--dry-run"]) == 1
    assert "FAIL" in capsys.readouterr().out


# sampler uniformity

@pytest.mark.parametrize("novel_step", [None, 1, 2])
def test_sampler_is_uniform(novel_step):
    """sampled chain frequencies are consistent with uniform over the full support."""
    p, h = 5, 2
    sp = dg.split_pairs(p, 0.3, 0.1, 0)
    tables = dg.step_tables(h, dg.partner_table(sp.train, p), dg.partner_table(sp.heldout, p), novel_step)
    C = dg.count_completions(tables, p)

    def allowed(ops):
        """check that every step of ops is permitted by tables."""
        s = ops[0]
        for t, x in enumerate(ops[1:]):
            if x not in tables[t][s]:
                return False
            s = (s + x) % p
        return True

    support = [ops for ops in itertools.product(range(p), repeat=h + 1) if allowed(ops)]
    assert len(support) == dg.chain_space(C) > 1
    rng, n = random.Random(123), 20000
    counts = dict.fromkeys(support, 0)
    for _ in range(n):
        counts[dg.sample_chain(rng, tables, C, p)] += 1  # keyerror if outside support
    exp = n / len(support)
    chi2 = sum((c - exp) ** 2 / exp for c in counts.values())
    assert chi2 < chi_square_bound(len(support) - 1), chi2


def test_uniformity_check_rejects_naive_sampler():
    """the uniformity check rejects a uniform-over-partners sampler, which is biased by degree."""
    p, h = 5, 2
    sp = dg.split_pairs(p, 0.3, 0.1, 0)
    trp = dg.partner_table(sp.train, p)
    C = dg.count_completions(dg.step_tables(h, trp, trp), p)
    rng, n, counts = random.Random(123), 20000, {}
    for _ in range(n):
        # naive: uniform start, uniform partner, restart on dead ends
        while True:
            ops = [rng.randrange(p)]
            s = ops[0]
            for _ in range(h):
                if not trp[s]:
                    break
                x = rng.choice(trp[s])
                ops.append(x)
                s = (s + x) % p
            if len(ops) == h + 1:
                break
        counts[tuple(ops)] = counts.get(tuple(ops), 0) + 1
    support = dg.chain_space(C)
    exp = n / support
    chi2 = sum((counts.get(c, 0) - exp) ** 2 / exp for c in counts) + exp * (support - len(counts))
    assert chi2 > chi_square_bound(support - 1), chi2


# rows and vocab

def test_make_row_example():
    """a known chain produces the expected row."""
    row = dg.make_row((31, 3, 40), 113)
    assert row == {"hop": 2, "operands": [31, 3, 40], "tokens": ["31", "+", "3", "+", "40", "="],
                   "answer": 74, "intermediates": [34, 74]}


def test_vocab_layout():
    """the vocab has pad, bos, the p numbers, then + and =, with contiguous ids."""
    v = dg.make_vocab(113)
    assert v["<pad>"] == 0 and v["<bos>"] == 1 and v["0"] == 2 and v["112"] == 114
    assert v["+"] == 115 and v["="] == 116 and len(v) == 117
    assert sorted(v.values()) == list(range(117))


# invariants on a generated dataset

def test_generated_dataset_passes_verifier(generated):
    """a generated dataset has the expected files and passes the verifier."""
    assert verify_data.verify_dir(generated) == []
    names = sorted(f.name for f in generated.iterdir())
    assert names == sorted(["train.jsonl", "val.jsonl", "extrap_depth.jsonl", "systematic.jsonl",
                            "systematic_extrap.jsonl", "grid_hop1.jsonl", "vocab.json", "meta.json"])


def test_generated_rows_respect_split(generated):
    """chains obey the closed and systematic rules, checked without the verifier."""
    entry = json.loads((generated / "meta.json").read_text())["config"]
    p = entry["p"]
    sp = dg.split_pairs(p, 0.3, 0.1, entry["seed"])
    load = lambda f: [json.loads(line) for line in open(generated / f)]  # noqa: E731
    for f in ("train.jsonl", "val.jsonl", "extrap_depth.jsonl"):
        for r in load(f):
            cls = step_classes(r["operands"], p, sp)
            if r["hop"] >= 2:
                assert set(cls) == {"train"}, (f, r)
    for f in ("systematic.jsonl", "systematic_extrap.jsonl"):
        for r in load(f):
            cls = step_classes(r["operands"], p, sp)
            assert cls[r["novel_step"] - 1] == "heldout"
            assert cls.count("train") == r["hop"] - 1
    tr = {tuple(r["operands"]) for r in load("train.jsonl")}
    for f in ("val.jsonl", "extrap_depth.jsonl", "systematic.jsonl", "systematic_extrap.jsonl"):
        assert not tr & {tuple(r["operands"]) for r in load(f)}, f


@pytest.mark.parametrize("mutate, expect", [
    ("answer", "wrong intermediates/answer"),
    ("leak_heldout_into_train", "step pair not train"),
    ("move_train_into_val", "cross-file overlap"),
    ("wrong_novel_step", "not exactly one held-out step"),
    ("drop_row", "counts"),
])
def test_verifier_catches_corruption(generated, mutate, expect):
    """the verifier flags each kind of corrupted dataset."""
    def rewrite(f, fn):
        """apply fn to the rows of file f in place."""
        path = generated / f
        rows = [json.loads(line) for line in open(path)]
        rows = fn(rows)
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))

    p = 11
    if mutate == "answer":
        def fn(rows):
            """corrupt the last answer."""
            rows[-1]["answer"] = (rows[-1]["answer"] + 1) % p
            return rows
        rewrite("train.jsonl", fn)
    elif mutate == "leak_heldout_into_train":
        sys_row = json.loads(open(generated / "systematic.jsonl").readline())

        def fn(rows):
            """replace a train chain with a systematic one."""
            i = next(i for i, r in enumerate(rows) if r["hop"] == sys_row["hop"])
            rows[i] = {k: v for k, v in sys_row.items() if k != "novel_step"}
            return rows
        rewrite("train.jsonl", fn)
    elif mutate == "move_train_into_val":
        tr_row = [json.loads(line) for line in open(generated / "train.jsonl")][-1]

        def fn(rows):
            """replace a val chain with a train one."""
            i = next(i for i, r in enumerate(rows) if r["hop"] == tr_row["hop"])
            rows[i] = tr_row
            return rows
        rewrite("val.jsonl", fn)
    elif mutate == "wrong_novel_step":
        def fn(rows):
            """point novel_step at the wrong step."""
            rows[0]["novel_step"] = rows[0]["hop"] if rows[0]["novel_step"] != rows[0]["hop"] else 1
            return rows
        rewrite("systematic.jsonl", fn)
    elif mutate == "drop_row":
        rewrite("extrap_depth.jsonl", lambda rows: rows[:-1])
    errs = verify_data.verify_dir(generated)
    assert any(expect in e for e in errs), errs


# feasibility

@pytest.mark.parametrize("over", [
    {"multihop": {"train_per_hop": 10_000, "val_per_hop": 5, "max_space_frac": 0.5}},
    {"systematic": {"per_hop": 100_000}},
    {"extrap": {"per_hop": 10**9}},
])
def test_infeasible_request_errors(tmp_path, over):
    """requests over the space cap raise, and the cli fails without writing."""
    with pytest.raises(dg.ConfigError, match="infeasible"):
        dg.generate(small_entry(**over))
    cfg = write_config(tmp_path, small_entry(**over))
    assert make_data.main(["--config", cfg, "--name", "t_p11"]) != 0
    assert not (tmp_path / "out" / "t_p11").exists()


# determinism and hop-1-only configs

def test_byte_identical_reruns(tmp_path):
    """two runs of the same config produce byte-identical files."""
    for sub in ("a", "b"):
        (tmp_path / sub).mkdir()
        cfg = write_config(tmp_path / sub, small_entry())
        assert make_data.main(["--config", cfg, "--name", "t_p11"]) == 0
    a, b = tmp_path / "a" / "out" / "t_p11", tmp_path / "b" / "out" / "t_p11"
    files = sorted(f.name for f in a.iterdir())
    assert files == sorted(f.name for f in b.iterdir())
    for f in files:
        assert (a / f).read_bytes() == (b / f).read_bytes(), f


def test_raising_max_train_hops_keeps_lower_hops():
    """adding a deeper training hop leaves the shallower rows unchanged."""
    small, _, _ = dg.generate(small_entry(max_train_hops=2, max_test_hops=2))
    big, _, _ = dg.generate(small_entry())
    for f in ("train.jsonl", "val.jsonl"):
        assert [r for r in big[f] if r["hop"] <= 2] == small[f]


def test_hop1_only_config(tmp_path):
    """a hop-1-only config writes no multi-hop, extrapolation or systematic files."""
    cfg = write_config(tmp_path, small_entry(name="t_h1", max_train_hops=1, max_test_hops=1))
    assert make_data.main(["--config", cfg, "--name", "t_h1"]) == 0
    out = tmp_path / "out" / "t_h1"
    assert sorted(f.name for f in out.iterdir()) == sorted(
        ["train.jsonl", "val.jsonl", "grid_hop1.jsonl", "vocab.json", "meta.json"])
    assert verify_data.verify_dir(out) == []


# hugging face loading

def test_huggingface_load(generated, tmp_path):
    """every file loads with datasets.load_dataset and has the expected columns."""
    datasets = pytest.importorskip("datasets")
    for f in sorted(generated.glob("*.jsonl")):
        ds = datasets.load_dataset("json", data_files=str(f), split="train", cache_dir=str(tmp_path / "hf"))
        n = sum(1 for _ in open(f))
        assert ds.num_rows == n
        cols = {"hop", "operands", "tokens", "answer", "intermediates"}
        cols |= {"novel_step"} if f.name.startswith("systematic") else set()
        cols |= {"split"} if f.name == "grid_hop1.jsonl" else set()
        assert set(ds.column_names) == cols, f.name
        assert ds.features["operands"].feature.dtype == "int64"


# dry run

def test_dry_run_writes_nothing(tmp_path, capsys):
    """--dry-run prints the table and writes nothing."""
    cfg = write_config(tmp_path, small_entry())
    assert make_data.main(["--config", cfg, "--name", "t_p11", "--dry-run"]) == 0
    assert not (tmp_path / "out").exists()
    out = capsys.readouterr().out
    assert "systematic_extrap.jsonl" in out and "FAIL" not in out


def test_dry_run_flags_infeasible(tmp_path, capsys):
    """--dry-run marks infeasible blocks FAIL and exits non-zero."""
    bad = small_entry(multihop={"train_per_hop": 10_000, "val_per_hop": 5, "max_space_frac": 0.5})
    cfg = write_config(tmp_path, bad)
    assert make_data.main(["--config", cfg, "--name", "t_p11", "--dry-run"]) == 1
    assert "FAIL" in capsys.readouterr().out
    assert not (tmp_path / "out").exists()


# overwrite protection and atomic writes

def test_no_overwrite(generated, tmp_path):
    """generating into an existing directory fails and leaves it untouched."""
    before = {f.name: f.read_bytes() for f in generated.iterdir()}
    cfg = str(tmp_path / "data.yaml")
    assert make_data.main(["--config", cfg, "--name", "t_p11"]) == 2
    assert {f.name: f.read_bytes() for f in generated.iterdir()} == before


def test_all_skips_existing(generated, tmp_path, capsys):
    """--all skips existing outputs and still generates new ones."""
    cfg = write_config(tmp_path, small_entry(), small_entry(name="t_new", max_train_hops=2, max_test_hops=2))
    assert make_data.main(["--config", cfg, "--all"]) == 0
    assert "skip t_p11" in capsys.readouterr().out
    assert verify_data.verify_dir(tmp_path / "out" / "t_new") == []


def test_failed_write_leaves_no_directory(tmp_path, monkeypatch):
    """a crash mid-write leaves neither the output nor the temp directory behind."""
    cfg = write_config(tmp_path, small_entry())
    real_dumps = json.dumps
    calls = {"n": 0}

    def flaky(obj, *a, **k):
        """fail on the 50th call to json.dumps."""
        calls["n"] += 1
        if calls["n"] == 50:
            raise RuntimeError("disk full")
        return real_dumps(obj, *a, **k)

    monkeypatch.setattr(make_data.json, "dumps", flaky)
    with pytest.raises(RuntimeError):
        make_data.main(["--config", cfg, "--name", "t_p11"])
    assert list((tmp_path / "out").iterdir()) == []
