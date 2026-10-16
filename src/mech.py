"""mechanistic-analysis helpers: activation caches, fourier views of weights and neurons, and one-layer attention."""

import numpy as np
import torch as t

import progress


@t.no_grad()
def run_cache(model, tokens, n_iter=1, generator=None):
    """run a model and keep every hooked activation.

    args:
        model: LoopedTransformer.
        tokens: token ids of shape (batch, pos).
        n_iter: number of core passes.
        generator: torch generator for a random start, or None.
    returns:
        (logits, cache) with cache keyed by (hook name, iteration).
    """
    cache = {}
    model.cache_all(cache)
    try:
        logits = model(tokens, n_iter, generator=generator)
    finally:
        model.remove_all_hooks()
    return logits, cache


def freq_norms(W, basis):
    """norm of a matrix's fourier components, with each frequency's cos and sin combined.

    args:
        W: tensor of shape (d, p) whose last axis indexes numbers.
        basis: fourier basis from progress.fourier_basis.
    returns:
        tensor of shape (p//2 + 1,): the constant term, then frequencies 1..p//2.
    """
    sq = (W @ basis.T).pow(2).sum(0)
    return t.cat([sq[:1], sq[1::2] + sq[2::2]]).sqrt()


def fold(f, p):
    """map any integer frequency to its equivalent in 0..p//2.

    args:
        f: integer frequency.
        p: modulus.
    returns:
        folded frequency.
    """
    return min(f % p, (-f) % p)


def harmonics(freqs, k_max, p):
    """frequencies reachable as k * f for a key frequency f and 1 <= k <= k_max.

    args:
        freqs: key frequencies.
        k_max: largest multiple.
        p: modulus.
    returns:
        sorted list of folded frequencies, excluding 0.
    """
    return sorted({fold(k * f, p) for f in freqs for k in range(1, k_max + 1)} - {0})


def fourier_2d(grid, basis):
    """2-d fourier transform of activations laid out on the (a, b) grid.

    args:
        grid: tensor of shape (p, p, n), indexed by the two operands.
        basis: fourier basis.
    returns:
        coefficients of shape (p, p, n); entry (i, j) pairs basis row i over a with row j over b.
    """
    return t.einsum('ia,jb,abn->ijn', basis, basis, grid)


def _rows(f):
    """basis rows for the constant term and frequency f."""
    return [0] if f == 0 else [0, 2 * f - 1, 2 * f]


def neuron_frequencies(coef):
    """for each neuron, the single frequency whose terms explain most of its non-constant variance.

    args:
        coef: 2-d fourier coefficients of shape (p, p, n).
    returns:
        (best frequency per neuron, fraction of non-constant variance it explains), both shape (n,).
    """
    p = coef.shape[0]
    sq = coef.pow(2)
    total = sq.sum((0, 1)) - sq[0, 0]
    fracs = []
    for f in range(1, p // 2 + 1):
        r = _rows(f)
        fracs.append((sq[r][:, r].sum((0, 1)) - sq[0, 0]) / total.clamp_min(1e-12))
    fracs = t.stack(fracs)
    best = fracs.argmax(0)
    return best + 1, fracs.gather(0, best[None])[0]


def keep_frequencies_2d(grid, basis, freqs):
    """project grid activations onto the terms built only from the constant and the given frequencies.

    args:
        grid: tensor of shape (p, p, n).
        basis: fourier basis.
        freqs: frequencies to keep.
    returns:
        projected activations, same shape as grid.
    """
    rows = sorted({r for f in freqs for r in _rows(f)} | {0})
    coef = fourier_2d(grid, basis)
    kept = t.zeros_like(coef)
    idx = t.tensor(rows, device=grid.device)
    kept[idx[:, None], idx[None, :]] = coef[idx[:, None], idx[None, :]]
    return t.einsum('ia,jb,ijn->abn', basis, basis, kept)


def head_scores(attn, q_resid, k_resid):
    """attention scores of one query against many keys, per head, before the softmax.

    args:
        attn: Attention module.
        q_resid: residual stream at the query position, shape (d_model,).
        k_resid: residual streams at the key positions, shape (n, d_model).
    returns:
        tensor of shape (heads, n), scaled as in the forward pass.
    """
    q = t.einsum('ihd,d->ih', attn.W_Q, q_resid)
    k = t.einsum('ihd,nd->inh', attn.W_K, k_resid)
    return t.einsum('ih,inh->in', q, k) / np.sqrt(attn.d_head)


def head_values(attn, resid):
    """what each head writes to the residual stream for each source position, at full attention.

    args:
        attn: Attention module.
        resid: residual streams at the source positions, shape (n, d_model).
    returns:
        tensor of shape (heads, n, d_model).
    """
    v = t.einsum('ihd,nd->inh', attn.W_V, resid)
    W_O = attn.W_O.reshape(attn.W_O.shape[0], attn.W_V.shape[0], attn.d_head)
    return t.einsum('dih,inh->ind', W_O, v)


def variance_explained_by(acts, labels):
    """fraction of each column's variance explained by the mean within each label group.

    args:
        acts: tensor of shape (n, k).
        labels: int tensor of shape (n,).
    returns:
        tensor of shape (k,).
    """
    acts = acts.double()
    groups = labels.unique(return_inverse=True)[1]
    sums = t.zeros(int(groups.max()) + 1, acts.shape[1], dtype=acts.dtype, device=acts.device)
    sums.index_add_(0, groups, acts)
    counts = t.bincount(groups).to(acts.dtype)[:, None]
    fitted = (sums / counts)[groups]
    resid = (acts - fitted).pow(2).sum(0)
    total = (acts - acts.mean(0)).pow(2).sum(0)
    return 1 - resid / total.clamp_min(1e-12)


def clock_fit(logits, sums, freqs, p):
    """least-squares fit of the logits by cos and sin of w * (sum - c) at the given frequencies.

    args:
        logits: tensor of shape (n, p).
        sums: true answer (sum mod p) per row, shape (n,).
        freqs: frequencies to use.
        p: modulus.
    returns:
        (fitted logits of shape (n, p), fraction of row-centred logit variance explained).
    """
    y = (logits - logits.mean(-1, keepdim=True)).double()
    c = t.arange(p, device=logits.device)
    d = (sums[:, None] - c[None, :]).double()
    feats = t.stack([g(2 * np.pi * f * d / p) for f in freqs for g in (t.cos, t.sin)], -1)
    coef = t.linalg.lstsq(feats.reshape(-1, feats.shape[-1]), y.reshape(-1, 1)).solution
    fitted = (feats @ coef)[..., 0]
    r2 = 1 - (y - fitted).pow(2).sum() / y.pow(2).sum()
    return fitted.to(logits.dtype), float(r2)


def loss_and_acc(logits, labels):
    """float64 cross-entropy and accuracy.

    args:
        logits: tensor of shape (n, p).
        labels: int tensor of shape (n,).
    returns:
        (loss, accuracy) as floats.
    """
    return (float(progress.cross_entropy_high_precision(logits, labels)),
            float((logits.argmax(-1) == labels).float().mean()))


def entropy(probs):
    """entropy of each distribution along the last axis, in nats.

    args:
        probs: tensor whose last axis sums to 1.
    returns:
        tensor without the last axis.
    """
    return -(probs * probs.clamp_min(1e-30).log()).sum(-1)
