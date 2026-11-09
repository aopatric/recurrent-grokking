"""what happens inside each model: how attention from = pools the tokens, how the readout clock shrinks with chain
length, how sparse the model's frequencies are inside, and where the total first appears.

    uv run python experiments/mechanism/internals.py RUN_DIR ... [--freqs results/mechanism/frequencies]
        [--n 1000] [--at STEP,STEP] [--out runs/mech-internals] [--device cuda]

each run is read at its final checkpoint and eval depth. it records:
- attention: for every attention block (prelude, each core pass, coda) and head, on the first --n chains of each hop
  count 1-6, the attention from = over the h + 2 tokens: its entropy divided by log(h + 2) (1 = an even average), the
  largest gap between any token's weight and 1 / (h + 2) (mean over chains of the per-chain maximum), the weight
  on = itself, and the weights averaged over chains.
- patterns: for every block, the full attention pattern from every position on the first --n 3-hop chains, averaged
  over chains, and per head its largest gap from an even average over the tokens each position sees.
- dilution: per hop count, the clock fitted to the logits at the readout's top 5 and top 20 frequencies: its r² and
  amplitude (the fitted logits' spread across answers). an even average over h + 2 tokens shrinks the amplitude like
  1 / (h + 2).
- embedding: the gini of W_E's norm over frequencies 1..p//2, its largest frequencies, and the share of its norm² at
  the readout's top 5 and top 20 frequencies (a spread-out embedding gives k / (p // 2)); the gini of W_U's norm over
  frequencies; and the gini a random gaussian matrix of the same shape gets.
- neurons: on the hop-1 grid, for each mlp (each pass of a shared core), the share of live neurons at = whose
  activation a single frequency explains above 0.5 and 0.85, and the most common such frequencies.
- sum: on the hop-1 grid, for the residual stream at = after each block's attention and after the whole block (each
  pass): the variance-weighted share explained by grouping rows by the answer (and by the first operand, as a
  control), and how well a linear readout decodes the answer (least squares onto one-hot answers, fitted on a random
  half of the grid and scored on the other half). a residual can hold the answer in a low-variance direction, so the
  share can be low where the readout still decodes it.
- chains: the same linear decoding on chains of 2 and 3 hops, sampled uniformly (as many as the hop-1 grid has rows),
  with the model's own accuracy on them.
- by_depth: on the hop-1 grid and the 2- and 3-hop chains, with the model stopped after each pass k = 1..T: how well a
  linear readout decodes the answer from the residual the unembedding reads (after the coda, if the model has one),
  and the model's own accuracy, both scored on the same held-out half.
with --at, each listed checkpoint records only the decoding, on the hop-1 grid and the 2- and 3-hop chains, plus the
model's own accuracy after each pass; the json then holds "T" and a "checkpoints" list.
the readout's frequencies (largest first) come from frequencies.py's output in --freqs. writes <out>/<run>.json and
<out>/summary.md.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch as t

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import analysis  # noqa: E402
import mech  # noqa: E402
import progress  # noqa: E402

HOPS = ("1", "2", "3", "4", "5", "6")
DECODE_HOPS = (2, 3)
# the full attention patterns are recorded on chains of this many hops, the longest seen in training
PATTERN_HOPS = "3"
KEYS = (5, 20)
CUTS = (0.5, 0.85)
# random matrices drawn for the gini a matrix with no frequency structure gets
GINI_DRAWS = 100
SPLIT_SEED = 0


def eq_cache(model, tokens, T, generator, keep, eq_only=True):
    """run a model and keep only the chosen activations, at the = position.

    args:
        model: LoopedTransformer.
        tokens: token ids of shape (batch, pos), = last.
        T: number of core passes.
        generator: torch generator for a random start, or None.
        keep: function of a hook name, true for the hooks to keep.
        eq_only: keep only the = position; false keeps every position.
    returns:
        (logits, cache) with cache keyed by (hook name, iteration): attention patterns as (batch, heads, keys) for
        the query at =, everything else as (batch, d) at =; with eq_only false, the full tensors.
    """
    cache = {}

    def save(tensor, name, iteration):
        """store the = slice of one activation, or all of it."""
        if not eq_only:
            cache[(name, iteration)] = tensor.detach()
            return
        cache[(name, iteration)] = (tensor[:, :, -1] if name.endswith("hook_attn") else tensor[:, -1]).detach()

    for hp in model.hook_points():
        if keep(hp.name):
            hp.add_hook(save, "fwd")
    try:
        logits = model(tokens, T, generator=generator)
    finally:
        model.remove_all_hooks()
    return logits, cache


def block(name, iteration):
    """a readable label for a hook's block and pass.

    args:
        name: hook name.
        iteration: loop iteration of the hook, or None outside the core.
    returns:
        label such as "prelude", "core pass 3" or "coda".
    """
    part = name.split(".")[0]
    return f"core pass {iteration + 1}" if part == "core" else part


def ordered(cache, suffix):
    """the cache keys of one hook type, in forward order.

    args:
        cache: dict from eq_cache.
        suffix: end of the hook name.
    returns:
        list of (name, iteration).
    """
    rank = {"prelude": 0, "core": 1, "coda": 2}
    keys = [k for k in cache if k[0].endswith(suffix)]
    return sorted(keys, key=lambda k: (rank[k[0].split(".")[0]], -1 if k[1] is None else k[1]))


def attention(model, run, groups, device):
    """attention from = in every block, per head and hop count.

    args:
        model: LoopedTransformer.
        run: dict from analysis.load_run.
        groups: dict from hop count (str) to (tokens, answers).
        device: torch device string.
    returns:
        dict from block label to hop count to {"entropy", "max_gap", "on_eq"}, each a list over heads, and "mean", the
        weights averaged over chains, heads x tokens.
    """
    out = {}
    for h, (x, _) in groups.items():
        _, cache = eq_cache(model, x, run["T"], analysis.eval_generator(run, device),
                            lambda n: n.endswith("attn.hook_attn"))
        n_tok = int(h) + 2
        for name, it in ordered(cache, "attn.hook_attn"):
            a = cache[(name, it)].double()
            assert a.shape[-1] == n_tok, f"expected {n_tok} keys, got {tuple(a.shape)}"
            out.setdefault(block(name, it), {})[h] = {
                "entropy": (mech.entropy(a) / np.log(n_tok)).mean(0).tolist(),
                "max_gap": (a - 1 / n_tok).abs().amax(-1).mean(0).tolist(),
                "on_eq": a[..., -1].mean(0).tolist(),
                "mean": a.mean(0).tolist()}
    return out


def patterns(model, run, tokens, device):
    """the full attention pattern of every block, from every position, on one set of chains.

    args:
        model: LoopedTransformer.
        run: dict from analysis.load_run.
        tokens: token ids of shape (batch, pos).
        device: torch device string.
    returns:
        dict from block label to {"mean": the pattern averaged over chains, heads x queries x keys, and "gap": per
        head, the largest over query positions q of the mean over chains of max |weight - 1 / (q + 1)| over the
        q + 1 keys the query sees}.
    """
    _, cache = eq_cache(model, tokens, run["T"], analysis.eval_generator(run, device),
                        lambda n: n.endswith("attn.hook_attn"), eq_only=False)
    out = {}
    for name, it in ordered(cache, "attn.hook_attn"):
        a = cache[(name, it)].double()
        n = a.shape[-1]
        gaps = t.stack([(a[:, :, q, :q + 1] - 1 / (q + 1)).abs().amax(-1).mean(0) for q in range(n)], -1)
        out[block(name, it)] = {"mean": a.mean(0).tolist(), "gap": gaps.amax(-1).tolist()}
    return out


def dilution(model, run, groups, order, device):
    """the readout clock's fit and amplitude per hop count.

    args:
        model: LoopedTransformer.
        run: dict from analysis.load_run.
        groups: dict from hop count (str) to (tokens, answers).
        order: readout frequencies, largest first.
        device: torch device string.
    returns:
        dict from hop count to {"acc", and per k: "r2_k", "amp_k"}.
    """
    p = run["cfg"].p
    out = {}
    for h, (x, y) in groups.items():
        logits = model(x, run["T"], generator=analysis.eval_generator(run, device)).float()
        row = {"acc": float((logits.argmax(-1) == y).float().mean())}
        for k in KEYS:
            fitted, r2 = mech.clock_fit(logits, y, order[:k], p)
            row[f"r2_{k}"] = r2
            row[f"amp_{k}"] = float(fitted.std(-1).mean())
        out[h] = row
    return out


def embedding(model, order, p, basis):
    """how W_E's and W_U's norms spread over frequencies.

    args:
        model: LoopedTransformer.
        order: readout frequencies, largest first.
        p: modulus.
        basis: fourier basis.
    returns:
        dict with W_E's norm at each frequency 1..p // 2, its gini, the 10 largest frequencies, and the norm² share at
        the readout's top k for each k in KEYS; W_U's norms and gini; and the gini of a random gaussian matrix of
        W_U's shape (mean and standard deviation over GINI_DRAWS draws, seeded by SPLIT_SEED).
    """
    norms = mech.freq_norms(model.embed.W_E[:, :p].float(), basis)[1:]
    sq = norms.pow(2)
    out = {"norms": norms.tolist(), "gini": progress.gini(norms),
           "top": (norms.argsort(descending=True)[:10] + 1).tolist()}
    for k in KEYS:
        out[f"share_{k}"] = float(sq[[f - 1 for f in order[:k]]].sum() / sq.sum())
    W_U = model.unembed.W_U[:, :p].float()
    unembed = mech.freq_norms(W_U, basis)[1:]
    g = t.Generator().manual_seed(SPLIT_SEED)
    rand = [progress.gini(mech.freq_norms(t.randn(W_U.shape, generator=g).to(basis.device), basis)[1:])
            for _ in range(GINI_DRAWS)]
    out.update(unembed_norms=unembed.tolist(), unembed_gini=progress.gini(unembed),
               gini_chance=float(np.mean(rand)), gini_chance_sd=float(np.std(rand)))
    return out


def weighted_explained(acts, labels):
    """variance-weighted share of all columns' variance explained by grouping rows by label.

    args:
        acts: tensor of shape (n, k).
        labels: int tensor of shape (n,).
    returns:
        float.
    """
    var = acts.double().var(0)
    return float((mech.variance_explained_by(acts, labels) * var / var.sum()).sum())


def decodable(acts, labels, train, p):
    """accuracy of a linear readout of the answer: least squares onto one-hot answers with a bias, fitted on the
    train rows and scored on the rest.

    args:
        acts: tensor of shape (n, d).
        labels: int tensor of shape (n,).
        train: bool tensor of shape (n,).
        p: number of answers.
    returns:
        float.
    """
    X = t.cat([acts.double(), t.ones(len(acts), 1, dtype=t.float64, device=acts.device)], 1)
    Y = t.nn.functional.one_hot(labels, p).double()
    W = t.linalg.lstsq(X[train], Y[train]).solution
    return float(((X[~train] @ W).argmax(-1) == labels[~train]).double().mean())


def half_split(n, device):
    """a random half of n rows, the same for every run.

    args:
        n: number of rows.
        device: torch device.
    returns:
        bool tensor of shape (n,), true for the rows that fit a readout.
    """
    split = t.randperm(n, generator=t.Generator().manual_seed(SPLIT_SEED)).to(device)
    train = t.zeros(n, dtype=t.bool, device=device)
    train[split[:n // 2]] = True
    return train


def block_decoding(cache, labels, train, p):
    """linear decoding of the answer after each block's attention and after the whole block.

    args:
        cache: dict from eq_cache, holding hook_resid_mid and hook_resid_post.
        labels: int tensor of answers.
        train: bool tensor, true for the rows that fit the readout.
        p: number of answers.
    returns:
        dict from block label to {"decode_mid", "decode"}.
    """
    return {block(name, it): {"decode_mid": decodable(cache[(name.replace("hook_resid_post", "hook_resid_mid"), it)],
                                                      labels, train, p),
                              "decode": decodable(cache[(name, it)], labels, train, p)}
            for name, it in ordered(cache, "hook_resid_post")}


def decode_sets(run, hops):
    """chains to decode the answer from: the hop-1 grid for 1 hop, and uniformly sampled chains (as many as the grid
    has rows, seeded by SPLIT_SEED) for more hops.

    args:
        run: dict from analysis.load_run.
        hops: hop counts, in increasing order.
    returns:
        dict from hop count (str) to (tokens, answers).
    raises:
        AssertionError: if the run's val chains don't follow the answer rule used here.
    """
    p = run["cfg"].p
    g = t.Generator().manual_seed(SPLIT_SEED)
    sets = {}
    for h in hops:
        if h == 1:
            sets["1"] = run["data"]["grid"][:2]
            continue
        x, y = run["data"]["sets"]["val"][str(h)]
        assert t.equal(x[:, -1], t.full_like(y, p)) and t.equal(x[:, :-1].sum(1) % p, y)
        ops = t.randint(0, p, (p * p, h + 1), generator=g)
        sets[str(h)] = (t.cat([ops, t.full((p * p, 1), p)], 1).to(x.device), (ops.sum(1) % p).to(x.device))
    return sets


def decoding(model, run, sets, device, readout=False):
    """the model's accuracy and the answer's linear decodability per block, on each set of chains.

    args:
        model: LoopedTransformer.
        run: dict from analysis.load_run.
        sets: dict from decode_sets.
        device: torch device string.
        readout: also record the model's own accuracy after each pass 1..T.
    returns:
        dict from hop count (str) to {"acc", "sum": block label -> {"decode_mid", "decode"}, and "readout" if asked}.
    """
    p = run["cfg"].p
    out = {}
    for h, (x, y) in sets.items():
        logits, cache = eq_cache(model, x, run["T"], analysis.eval_generator(run, device),
                                 lambda name: name.endswith(("hook_resid_mid", "hook_resid_post")))
        out[h] = {"acc": float((logits.argmax(-1) == y).double().mean()),
                  "sum": block_decoding(cache, y, half_split(len(y), y.device), p)}
        if readout:
            out[h]["readout"] = [float((model(x, k, generator=analysis.eval_generator(run, device)).argmax(-1) == y)
                                       .double().mean()) for k in range(1, run["T"] + 1)]
    return out


def by_depth(model, run, sets, device):
    """per depth, the answer's linear decodability from the residual the unembedding reads, and the model's own
    accuracy, scored on the same held-out half.

    args:
        model: LoopedTransformer.
        run: dict from analysis.load_run.
        sets: dict from decode_sets.
        device: torch device string.
    returns:
        dict from hop count (str) to {"probe", "readout"}: lists over depths 1..T.
    """
    p = run["cfg"].p
    out = {}
    for h, (x, y) in sets.items():
        train = half_split(len(y), y.device)
        probe, readout = [], []
        for k in range(1, run["T"] + 1):
            logits, cache = eq_cache(model, x, k, analysis.eval_generator(run, device),
                                     lambda name: name.endswith("hook_resid_post"))
            last = ordered(cache, "hook_resid_post")[-1]
            probe.append(decodable(cache[last], y, train, p))
            readout.append(float((logits.argmax(-1) == y)[~train].double().mean()))
        out[h] = {"probe": probe, "readout": readout}
    return out


def choose(run, at):
    """the checkpoints at the listed steps.

    args:
        run: dict from analysis.load_run.
        at: comma-separated steps.
    returns:
        list of (step, path).
    raises:
        ValueError: if a listed step has no checkpoint.
    """
    steps = {int(s) for s in at.split(",")}
    chosen = [c for c in analysis.checkpoints(run) if c[0] in steps]
    if len(chosen) != len(steps):
        raise ValueError(f"{run['name']}: no checkpoint at {sorted(steps - {c[0] for c in chosen})}")
    return chosen


def grid_measures(model, run, order, basis, device):
    """neuron frequencies and where the total appears, on the hop-1 grid.

    args:
        model: LoopedTransformer.
        run: dict from analysis.load_run.
        order: readout frequencies, largest first.
        basis: fourier basis.
        device: torch device string.
    returns:
        dict with "neurons" (block label -> shares and common frequencies), "sum" (block label -> share explained by
        the answer and by the first operand, and linear decoding accuracy after the attention and after the block),
        and the chance levels of the grouping and of decoding.
    """
    p = run["cfg"].p
    tokens, labels, _ = run["data"]["grid"]
    a_idx = t.arange(p, device=tokens.device)
    # the grid is laid out a-major, so it reshapes to (a, b, ...)
    assert t.equal(tokens[:, 0], a_idx.repeat_interleave(p)) and t.equal(tokens[:, 1], a_idx.repeat(p))
    _, cache = eq_cache(model, tokens, run["T"], analysis.eval_generator(run, device),
                        lambda n: n.endswith(("mlp.hook_post", "hook_resid_mid", "hook_resid_post")))
    train = half_split(len(labels), labels.device)
    top20 = set(order[:20])
    neurons = {}
    for name, it in ordered(cache, "mlp.hook_post"):
        grid = cache[(name, it)].double().reshape(p, p, -1)
        live = grid.pow(2).sum((0, 1)) > 1e-8
        best, frac = mech.neuron_frequencies(mech.fourier_2d(grid, basis.double()))
        best, frac = best[live], frac[live]
        clean = frac > CUTS[0]
        counts = t.bincount(best[clean], minlength=p // 2 + 1)
        common = [int(f) for f in counts.argsort(descending=True)[:5] if counts[f] > 0]
        neurons[block(name, it)] = {
            "live": int(live.sum()),
            **{f"share_{c}": float((frac > c).float().mean()) for c in CUTS},
            "common": common,
            "in_readout_top20": float(np.mean([int(f) in top20 for f in best[clean].tolist()])) if clean.any() else 0.0}
    sums = block_decoding(cache, labels, train, p)
    for name, it in ordered(cache, "hook_resid_post"):
        sums[block(name, it)].update(answer=weighted_explained(cache[(name, it)], labels),
                                     first=weighted_explained(cache[(name, it)], tokens[:, 0]))
    return {"neurons": neurons, "sum": sums, "chance": (p - 1) / (len(labels) - 1), "decode_chance": 1 / p}


def summarize(name, res):
    """markdown lines for one run.

    args:
        name: run name.
        res: dict of this run's results.
    returns:
        list of markdown lines.
    """
    lines = [f"## {name}", ""]
    lines += ["attention from =: normalized entropy, min over heads / max gap to 1/(h+2), max over heads", "",
              "| block | " + " | ".join(f"h{h}" for h in HOPS) + " |", "|---|" + "---|" * len(HOPS)]
    for label, by_hop in res["attention"].items():
        lines.append(f"| {label} | " + " | ".join(
            f"{min(by_hop[h]['entropy']):.2f} / {max(by_hop[h]['max_gap']):.3f}" for h in HOPS) + " |")
    d = res["dilution"]
    lines += ["", "readout clock per hop: acc / r² / amplitude ratio to hop 1 (an even average gives 3/(h+2))", "",
              "| k | " + " | ".join(f"h{h}" for h in HOPS) + " |", "|---|" + "---|" * len(HOPS),
              "| even average | " + " | ".join(f"{3 / (int(h) + 2):.2f}" for h in HOPS) + " |"]
    for k in KEYS:
        lines.append(f"| {k} | " + " | ".join(
            f"{d[h]['acc']:.2f} / {d[h][f'r2_{k}']:.2f} / {d[h][f'amp_{k}'] / d['1'][f'amp_{k}']:.2f}"
            for h in HOPS) + " |")
    e = res["embedding"]
    lines += ["", f"embedding: gini {e['gini']:.2f}; top {e['top'][:6]}; norm² share at the readout's top 5 / 20: "
              f"{e['share_5']:.2f} / {e['share_20']:.2f} (spread out: {5 / 56:.2f} / {20 / 56:.2f})", ""]
    g = res["grid"]
    lines += [f"hop-1 grid (grouping chance {g['chance']:.3f}, decoding chance {g['decode_chance']:.3f}): neurons with "
              "one frequency > 0.5 / > 0.85, most common frequencies, share of those in the readout's top 20; residual "
              "at = explained by the answer / the first operand; answer decoded linearly after the attention / after "
              "the block", "", "| block | neurons | common | in readout top 20 | answer | first operand | decoded |",
              "|---|---|---|---|---|---|---|"]
    for label in g["sum"]:
        n = g["neurons"].get(label)
        cells = (f"{n['share_0.5']:.2f} / {n['share_0.85']:.2f} | {n['common']} | {n['in_readout_top20']:.2f}"
                 if n else "— | — | —")
        s = g["sum"][label]
        lines.append(f"| {label} | {cells} | {s['answer']:.2f} | {s['first']:.2f} | "
                     f"{s['decode_mid']:.2f} / {s['decode']:.2f} |")
    c = res["chains"]
    lines += ["", "random chains: answer decoded linearly after the attention / after the block (the model's own "
              "accuracy on them in the header)", "",
              "| block | " + " | ".join(f"h{h} (acc {c[h]['acc']:.2f})" for h in c) + " |", "|---|" + "---|" * len(c)]
    for label in g["sum"]:
        lines.append(f"| {label} | " + " | ".join(
            f"{c[h]['sum'][label]['decode_mid']:.2f} / {c[h]['sum'][label]['decode']:.2f}" for h in c) + " |")
    return lines + [""]


def summarize_at(name, res):
    """markdown lines for one run read at several checkpoints.

    args:
        name: run name.
        res: dict with "T" and "checkpoints".
    returns:
        list of markdown lines.
    """
    lines = [f"## {name}", "", "per checkpoint and hop count: the model's accuracy, its own readout after pass 1, and "
             "the answer decoded linearly after each block", ""]
    for row in res["checkpoints"]:
        d = row["decoding"]
        labels = list(d["1"]["sum"])
        lines += [f"step {row['step']}", "", "| hops | acc | readout after pass 1 | " + " | ".join(labels) + " |",
                  "|---|---|---|" + "---|" * len(labels)]
        for h, r in d.items():
            lines.append(f"| {h} | {r['acc']:.2f} | {r['readout'][0]:.2f} | "
                         + " | ".join(f"{r['sum'][b]['decode']:.2f}" for b in labels) + " |")
        lines.append("")
    return lines


def main(argv=None):
    """run the cli.

    args:
        argv: argument list, or None for sys.argv.
    returns:
        exit code.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--freqs", default="results/mechanism/frequencies", help="folder of frequencies.py outputs")
    ap.add_argument("--n", type=int, default=1000, help="chains per hop count")
    ap.add_argument("--at", help="comma-separated steps: record only the decoding (with the readout after each pass) "
                    "on 1-, 2- and 3-hop chains at these checkpoints")
    ap.add_argument("--out", default="runs/mech-internals")
    ap.add_argument("--device", default="cuda" if t.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    md = ["# inside the models", ""]
    with t.no_grad():
        for rd in args.runs:
            run = analysis.load_run(rd, device=args.device)
            if args.at:
                sets = decode_sets(run, (1,) + DECODE_HOPS)
                rows = [{"step": step, "decoding": decoding(analysis.load_model(run, path, args.device), run, sets,
                                                           args.device, readout=True)}
                        for step, path in choose(run, args.at)]
                res = {"T": run["T"], "checkpoints": rows}
                (out / f"{run['name']}.json").write_text(json.dumps(res))
                md += summarize_at(run["name"], res)
                print(f"done {run['name']}", flush=True)
                continue
            step, path = analysis.checkpoints(run)[-1]
            model = analysis.load_model(run, path, args.device)
            order = json.loads((Path(args.freqs) / f"{run['name']}.json").read_text())["order"]
            sets = run["data"]["sets"]
            groups = {h: (x[:args.n], y[:args.n]) for h, (x, y) in {**sets["val"], **sets["extrap_depth"]}.items()
                      if h in HOPS}
            basis = progress.fourier_basis(run["cfg"].p, args.device)
            res = {"step": step, "T": run["T"],
                   "attention": attention(model, run, groups, args.device),
                   "patterns": patterns(model, run, groups[PATTERN_HOPS][0], args.device),
                   "dilution": dilution(model, run, groups, order, args.device),
                   "embedding": embedding(model, order, run["cfg"].p, basis),
                   "grid": grid_measures(model, run, order, basis, args.device),
                   "chains": decoding(model, run, decode_sets(run, DECODE_HOPS), args.device),
                   "by_depth": by_depth(model, run, decode_sets(run, (1,) + DECODE_HOPS), args.device)}
            (out / f"{run['name']}.json").write_text(json.dumps(res))
            md += summarize(run["name"], res)
            print(f"done {run['name']}", flush=True)
    (out / "summary.md").write_text("\n".join(md))
    print(f"wrote {out / 'summary.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
