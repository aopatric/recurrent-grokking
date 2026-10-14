"""generate modular-addition chain datasets as pure functions (no file i/o)."""

import random
from dataclasses import dataclass, field

GENERATOR_VERSION = "2.0.0"

FILES = (
    "train.jsonl",
    "val.jsonl",
    "extrap_depth.jsonl",
    "systematic.jsonl",
    "systematic_extrap.jsonl",
    "grid_hop1.jsonl",
)

_ENTRY_KEYS = {
    "name", "p", "ops", "max_train_hops", "max_test_hops", "seed",
    "hop1", "multihop", "extrap", "systematic",
}
_SECTION_KEYS = {
    "hop1": {"train_frac", "val_frac"},
    "multihop": {"train_per_hop", "val_per_hop", "max_space_frac"},
    "extrap": {"per_hop"},
    "systematic": {"per_hop"},
}


class ConfigError(ValueError):
    """invalid or infeasible dataset config."""


# validation

def is_prime(n: int) -> bool:
    """check primality by trial division.

    args:
        n: integer to test.
    returns:
        True if n is prime.
    """
    if n < 2:
        return False
    i = 2
    while i * i <= n:
        if n % i == 0:
            return False
        i += 1
    return True


def _is_int(x) -> bool:
    """check for an int that is not a bool.

    args:
        x: value to test.
    returns:
        True if x is a non-bool int.
    """
    return isinstance(x, int) and not isinstance(x, bool)


def _is_num(x) -> bool:
    """check for an int or float that is not a bool.

    args:
        x: value to test.
    returns:
        True if x is a non-bool number.
    """
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def validate_entry(entry: dict) -> None:
    """validate one dataset entry, reporting every violated rule at once.

    args:
        entry: dataset entry from the config.
    raises:
        ConfigError: if any rule is violated.
    """
    errs = []
    name = entry.get("name", "<unnamed>")

    # structure first: the value checks below index into these keys
    keys = set(entry)
    if missing := _ENTRY_KEYS - keys:
        errs.append(f"missing keys: {sorted(missing)}")
    if unknown := keys - _ENTRY_KEYS:
        errs.append(f"unknown keys: {sorted(unknown)}")
    for sec, sec_keys in _SECTION_KEYS.items():
        val = entry.get(sec)
        if val is None:
            continue
        if not isinstance(val, dict):
            errs.append(f"{sec} must be a mapping")
            continue
        if missing := sec_keys - set(val):
            errs.append(f"{sec}: missing keys {sorted(missing)}")
        if unknown := set(val) - sec_keys:
            errs.append(f"{sec}: unknown keys {sorted(unknown)}")
    if errs:
        raise ConfigError(f"[{name}] " + "; ".join(errs))

    if not isinstance(name, str) or not name or "/" in name or name.startswith("."):
        errs.append("name must be a non-empty string without '/' and not starting with '.'")
    p = entry["p"]
    if not _is_int(p) or not is_prime(p):
        errs.append(f"p must be a prime integer, got {p!r}")
    if entry["ops"] != ["+"]:
        errs.append(f"ops must be ['+'], got {entry['ops']!r}")
    if not _is_int(entry["seed"]):
        errs.append("seed must be an integer")

    tr, te = entry["max_train_hops"], entry["max_test_hops"]
    if not (_is_int(tr) and tr >= 1):
        errs.append("max_train_hops must be an integer >= 1")
    elif not (_is_int(te) and te >= tr):
        errs.append("max_test_hops must be an integer >= max_train_hops")

    tf, vf = entry["hop1"]["train_frac"], entry["hop1"]["val_frac"]
    if not (_is_num(tf) and 0 < tf < 1 and _is_num(vf) and 0 < vf < 1):
        errs.append("hop1 fractions must be numbers in (0, 1)")
    elif tf + vf >= 1:
        errs.append("hop1.train_frac + hop1.val_frac must be < 1 (held-out pairs must exist)")
    elif _is_int(p) and is_prime(p) and (int(tf * p * p) < 1 or int(vf * p * p) < 1):
        errs.append("hop1 fractions give an empty train or val pair set for this p")

    msf = entry["multihop"]["max_space_frac"]
    if not (_is_num(msf) and 0 < msf <= 1):
        errs.append("multihop.max_space_frac must be in (0, 1]")

    for sec, key in [("multihop", "train_per_hop"), ("multihop", "val_per_hop"),
                     ("extrap", "per_hop"), ("systematic", "per_hop")]:
        v = entry[sec][key]
        if not (_is_int(v) and v >= 1):
            errs.append(f"{sec}.{key} must be a positive integer")
    sys_n = entry["systematic"]["per_hop"]
    if _is_int(te) and te >= 2 and _is_int(sys_n) and sys_n < te:
        errs.append("systematic.per_hop must be >= max_test_hops so every step position gets a chain")

    if errs:
        raise ConfigError(f"[{name}] " + "; ".join(errs))


def validate_config(cfg: dict) -> None:
    """validate a whole config file: top-level shape, unique names and every entry.

    args:
        cfg: parsed config with output_root and datasets.
    raises:
        ConfigError: if the config or any entry is invalid.
    """
    if not isinstance(cfg, dict) or "output_root" not in cfg or "datasets" not in cfg:
        raise ConfigError("config must be a mapping with 'output_root' and 'datasets'")
    if not isinstance(cfg["datasets"], list) or not cfg["datasets"]:
        raise ConfigError("'datasets' must be a non-empty list")
    names = [d.get("name") for d in cfg["datasets"]]
    if dupes := sorted({n for n in names if names.count(n) > 1}):
        raise ConfigError(f"duplicate dataset names: {dupes}")
    for d in cfg["datasets"]:
        validate_entry(d)


# pair split

@dataclass(frozen=True)
class PairSplit:
    """partition of the p x p pair table into train, val and held-out pairs."""

    p: int
    train: list   # (a, b) tuples, in sample order
    val: list     # in sample order
    heldout: list  # in pair-index order

    def label(self) -> dict:
        """map every pair to its class.

        returns:
            dict from (a, b) to "train", "val" or "heldout".
        """
        lab = {pr: "heldout" for pr in self.heldout}
        lab.update({pr: "train" for pr in self.train})
        lab.update({pr: "val" for pr in self.val})
        return lab


def split_pairs(p: int, train_frac: float, val_frac: float, seed: int) -> PairSplit:
    """split the p x p pair table into train, val and held-out pairs.

    args:
        p: prime modulus.
        train_frac: fraction of pairs used for training.
        val_frac: fraction of pairs used for validation.
        seed: master seed.
    returns:
        PairSplit; depends only on these four arguments.
    """
    n_pairs = p * p
    n_train, n_val = int(train_frac * n_pairs), int(val_frac * n_pairs)
    idx = random.Random(f"{seed}:pairs").sample(range(n_pairs), n_train + n_val)
    taken = set(idx)
    as_pair = lambda i: (i // p, i % p)  # noqa: E731
    return PairSplit(
        p=p,
        train=[as_pair(i) for i in idx[:n_train]],
        val=[as_pair(i) for i in idx[n_train:]],
        heldout=[as_pair(i) for i in range(n_pairs) if i not in taken],
    )


def partner_table(pairs, p: int) -> list:
    """index a pair set by its first element.

    args:
        pairs: iterable of (s, x) pairs.
        p: prime modulus.
    returns:
        list where entry s is the sorted list of x with (s, x) in pairs.
    """
    out = [[] for _ in range(p)]
    for s, x in pairs:
        out[s].append(x)
    return [sorted(xs) for xs in out]


# counting and sampling

def step_tables(h: int, train_partners: list, heldout_partners: list, novel_step: int | None = None) -> list:
    """build the per-step partner tables that define a chain set.

    args:
        h: number of hops.
        train_partners: partner table of train pairs.
        heldout_partners: partner table of held-out pairs.
        novel_step: 1-based step that uses held-out pairs, or None for closed chains.
    returns:
        list of h partner tables, one per step.
    """
    return [heldout_partners if t == novel_step else train_partners for t in range(1, h + 1)]


def count_completions(tables: list, p: int) -> list:
    """count chain completions backward over the steps.

    args:
        tables: per-step partner tables from step_tables.
        p: prime modulus.
    returns:
        C where C[t][s] is the number of ways to finish steps t+1..h from partial sum s.
    """
    h = len(tables)
    C: list = [[] for _ in range(h + 1)]
    C[h] = [1] * p
    for t in range(h - 1, -1, -1):
        nxt, allowed = C[t + 1], tables[t]
        C[t] = [sum(nxt[(s + x) % p] for x in allowed[s]) for s in range(p)]
    return C


def chain_space(C: list) -> int:
    """total number of chains in a set.

    args:
        C: completion counts from count_completions.
    returns:
        exact number of chains.
    """
    return sum(C[0])


def _weighted_index(rng: random.Random, weights: list) -> int:
    """draw an index with probability proportional to integer weights.

    args:
        rng: random source.
        weights: non-negative ints with a positive sum.
    returns:
        the sampled index.
    """
    # integer arithmetic: float weights lose precision on big counts
    r = rng.randrange(sum(weights))
    for i, w in enumerate(weights):
        if r < w:
            return i
        r -= w
    raise AssertionError("unreachable")


def sample_chain(rng: random.Random, tables: list, C: list, p: int) -> tuple:
    """draw one chain exactly uniformly from the set defined by tables.

    args:
        rng: random source.
        tables: per-step partner tables.
        C: completion counts for tables.
        p: prime modulus.
    returns:
        operand tuple of length h + 1.
    """
    # weight every choice by its completion count so each full chain is equally likely
    s = _weighted_index(rng, C[0])
    ops = [s]
    for t, allowed in enumerate(tables):
        cand = allowed[s]
        x = cand[_weighted_index(rng, [C[t + 1][(s + x) % p] for x in cand])]
        ops.append(x)
        s = (s + x) % p
    return tuple(ops)


def sample_unique(rng, n: int, tables: list, C: list, p: int, exclude=frozenset()) -> list:
    """draw n distinct chains by rejection sampling.

    args:
        rng: random source.
        n: number of chains.
        tables: per-step partner tables.
        C: completion counts for tables.
        p: prime modulus.
        exclude: chains that must not be returned.
    returns:
        list of n operand tuples, in first-seen order.
    """
    seen, out = set(), []
    while len(out) < n:
        c = sample_chain(rng, tables, C, p)
        if c not in seen and c not in exclude:
            seen.add(c)
            out.append(c)
    return out


# rows and vocab

def partial_sums(operands, p: int) -> list:
    """compute the running sums of a chain.

    args:
        operands: chain operands x_0..x_h.
        p: prime modulus.
    returns:
        [s_1, ..., s_h]; the last element is the answer.
    """
    s, out = operands[0], []
    for x in operands[1:]:
        s = (s + x) % p
        out.append(s)
    return out


def make_tokens(operands) -> list:
    """render a chain as prompt tokens.

    args:
        operands: chain operands.
    returns:
        token strings like ["3", "+", "5", "="].
    """
    toks = []
    for i, x in enumerate(operands):
        if i:
            toks.append("+")
        toks.append(str(x))
    toks.append("=")
    return toks


def make_row(operands, p: int, **extra) -> dict:
    """build one dataset row.

    args:
        operands: chain operands.
        p: prime modulus.
        **extra: additional fields such as novel_step or split.
    returns:
        dict with hop, operands, tokens, answer, intermediates and any extra fields.
    """
    inter = partial_sums(operands, p)
    row = {
        "hop": len(operands) - 1,
        "operands": list(operands),
        "tokens": make_tokens(operands),
        "answer": inter[-1],
        "intermediates": inter,
    }
    row.update(extra)
    return row


def make_vocab(p: int) -> dict:
    """build the token vocabulary.

    args:
        p: prime modulus.
    returns:
        dict from token to id: <pad>, <bos>, "0".."p-1", "+", "=".
    """
    # <bos> is reserved here but never written into rows; the collator prepends it
    vocab = {"<pad>": 0, "<bos>": 1}
    for i in range(p):
        vocab[str(i)] = len(vocab)
    vocab["+"] = len(vocab)
    vocab["="] = len(vocab)
    return vocab


# planning

def systematic_counts(per_hop: int, h: int) -> list:
    """split a per-hop count evenly over step positions.

    args:
        per_hop: total systematic chains for this hop.
        h: number of hops.
    returns:
        list of h counts; entry k-1 is the count for novel step k.
    """
    q, r = divmod(per_hop, h)
    return [q + (1 if k < r else 0) for k in range(h)]


@dataclass
class SetSpec:
    """one sampled block of rows: a set, a hop and, for systematic sets, a novel step."""

    kind: str                  # train | val | extrap | systematic
    file: str
    hop: int
    novel_step: int | None
    n: int
    budget: int                # chains drawn from this space so far (train + val for val)
    raw_space: int
    space: int
    rng_tag: str
    tables: list = field(repr=False, default_factory=list)
    C: list = field(repr=False, default_factory=list)

    def feasible(self, max_space_frac: float) -> bool:
        """check the request against the space cap.

        args:
            max_space_frac: largest allowed fraction of the space to draw.
        returns:
            True if budget <= max_space_frac * space.
        """
        return self.budget <= max_space_frac * self.space


def plan(entry: dict) -> tuple:
    """compute the pair split and every sampled block, without sampling.

    args:
        entry: validated dataset entry.
    returns:
        (PairSplit, list of SetSpec in output order).
    """
    p, seed = entry["p"], entry["seed"]
    tr_h, te_h = entry["max_train_hops"], entry["max_test_hops"]
    mh, ex, sy = entry["multihop"], entry["extrap"], entry["systematic"]
    split = split_pairs(p, entry["hop1"]["train_frac"], entry["hop1"]["val_frac"], seed)
    trp, hop = partner_table(split.train, p), partner_table(split.heldout, p)

    specs = []

    def add(kind, file, h, n, budget, tag, k=None):
        """append a SetSpec with its exact space size."""
        tables = step_tables(h, trp, hop, k)
        C = count_completions(tables, p)
        specs.append(SetSpec(kind, file, h, k, n, budget, p ** (h + 1), chain_space(C),
                             f"{seed}:{tag}", tables, C))

    # one rng tag per block, so changing one block never shifts another
    for h in range(2, tr_h + 1):
        add("train", "train.jsonl", h, mh["train_per_hop"], mh["train_per_hop"], f"train:h{h}")
    for h in range(2, tr_h + 1):
        add("val", "val.jsonl", h, mh["val_per_hop"], mh["train_per_hop"] + mh["val_per_hop"], f"val:h{h}")
    for h in range(tr_h + 1, te_h + 1):
        add("extrap", "extrap_depth.jsonl", h, ex["per_hop"], ex["per_hop"], f"extrap:h{h}")
    for h in range(2, te_h + 1):
        file = "systematic.jsonl" if h <= tr_h else "systematic_extrap.jsonl"
        for k, n in enumerate(systematic_counts(sy["per_hop"], h), start=1):
            add("systematic", file, h, n, n, f"systematic:h{h}:k{k}", k)
    return split, specs


def infeasible(entry: dict, specs: list) -> list:
    """find blocks that request too much of their space.

    args:
        entry: dataset entry.
        specs: blocks from plan.
    returns:
        list of infeasible SetSpec.
    """
    msf = entry["multihop"]["max_space_frac"]
    return [s for s in specs if not s.feasible(msf)]


# generation

def generate(entry: dict) -> tuple:
    """generate every file of a dataset in memory.

    args:
        entry: dataset entry.
    returns:
        (dict from filename to rows, vocab, meta).
    raises:
        ConfigError: if the entry is invalid or any block is infeasible.
    """
    validate_entry(entry)
    p = entry["p"]
    split, specs = plan(entry)
    if bad := infeasible(entry, specs):
        msf = entry["multihop"]["max_space_frac"]
        lines = [
            f"  {s.file} hop={s.hop}" + (f" k={s.novel_step}" if s.novel_step else "")
            + f": needs {s.budget} but space is {s.space} (cap {msf} -> {int(msf * s.space)})"
            for s in bad
        ]
        raise ConfigError(f"[{entry['name']}] infeasible request:\n" + "\n".join(lines))

    files = {f: [] for f in FILES}
    files["train.jsonl"] += [make_row(pr, p) for pr in split.train]
    files["val.jsonl"] += [make_row(pr, p) for pr in split.val]

    train_by_hop = {}
    for s in specs:
        rng = random.Random(s.rng_tag)
        # val must not repeat a train chain of the same hop
        exclude = train_by_hop.get(s.hop, frozenset()) if s.kind == "val" else frozenset()
        chains = sample_unique(rng, s.n, s.tables, s.C, p, exclude)
        if s.kind == "train":
            train_by_hop[s.hop] = frozenset(chains)
        extra = {"novel_step": s.novel_step} if s.kind == "systematic" else {}
        files[s.file] += [make_row(c, p, **extra) for c in chains]

    label = split.label()
    files["grid_hop1.jsonl"] = [
        make_row((a, b), p, split=label[(a, b)]) for a in range(p) for b in range(p)
    ]
    files = {f: rows for f, rows in files.items() if rows}

    vocab = make_vocab(p)
    # space sizes as strings: they can exceed what JSON readers hold exactly
    meta = {
        "generator_version": GENERATOR_VERSION,
        "config": entry,
        "vocab_size": len(vocab),
        "hop1": {
            "space": str(p * p),
            "n_train": len(split.train),
            "n_val": len(split.val),
            "n_heldout": len(split.heldout),
        },
        "sets": [
            {
                "file": s.file,
                "kind": s.kind,
                "hop": s.hop,
                "novel_step": s.novel_step,
                "raw_space": str(s.raw_space),
                "space": str(s.space),
                "n": s.n,
            }
            for s in specs
        ],
    }
    return files, vocab, meta
