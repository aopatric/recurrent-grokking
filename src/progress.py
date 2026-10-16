"""nanda et al.'s grokking progress measures, computed on the full hop-1 grid."""

import numpy as np
import torch as t
import torch.nn.functional as F


def cross_entropy_high_precision(logits, labels):
    """mean cross-entropy computed in float64.

    args:
        logits: tensor of shape (batch, classes).
        labels: int tensor of shape (batch,).
    returns:
        scalar loss tensor (float64).
    """
    # float32 log-softmax underflows on very confident predictions, which causes loss spikes
    logprobs = F.log_softmax(logits.to(t.float64), dim=-1)
    return -logprobs.gather(-1, labels[:, None]).mean()


def fourier_basis(p, device=None):
    """build nanda's normalized fourier basis over [0, p).

    args:
        p: odd prime modulus.
        device: torch device.
    returns:
        tensor of shape (p, p); rows are const, cos 1, sin 1, ..., cos p//2, sin p//2.
    """
    x = t.arange(p, device=device, dtype=t.float32)
    rows = [t.ones(p, device=device)]
    for f in range(1, p // 2 + 1):
        rows += [t.cos(2 * np.pi * f * x / p), t.sin(2 * np.pi * f * x / p)]
    basis = t.stack(rows)
    return basis / basis.norm(dim=1, keepdim=True)


def xpy_directions(basis):
    """unit directions in the p*p input grid for cos(w(a+b)) and sin(w(a+b)) at every frequency.

    args:
        basis: fourier basis from fourier_basis.
    returns:
        tensor of shape (2 * (p//2), p*p); rows 2(f-1) and 2(f-1)+1 are cos and sin at frequency f.
    """
    p = basis.shape[0]
    rows = []
    for f in range(1, p // 2 + 1):
        c, s = basis[2 * f - 1], basis[2 * f]
        # cos(a+b) = cos a cos b - sin a sin b; sin(a+b) = sin a cos b + cos a sin b
        rows.append(((t.outer(c, c) - t.outer(s, s)) / np.sqrt(2)).flatten())
        rows.append(((t.outer(s, c) + t.outer(c, s)) / np.sqrt(2)).flatten())
    return t.stack(rows)


def component_norms(W, basis):
    """norm of each fourier component of a matrix whose last axis indexes numbers.

    args:
        W: tensor of shape (d, p).
        basis: fourier basis.
    returns:
        tensor of shape (p,), one norm per basis row.
    """
    return (W @ basis.T).norm(dim=0)


def gini(x):
    """gini coefficient of a non-negative vector; 0 is uniform, near 1 is sparse.

    args:
        x: 1-d tensor.
    returns:
        float.
    """
    x = x.flatten().sort().values.double()
    n = x.numel()
    idx = t.arange(1, n + 1, dtype=t.float64, device=x.device)
    return float((2 * (idx * x).sum()) / (n * x.sum()) - (n + 1) / n)


def key_component(logits, dirs, freqs):
    """the part of the grid logits along cos/sin(w(a+b)) for the given frequencies.

    args:
        logits: tensor of shape (p*p, p), rows in pair-index order.
        dirs: directions from xpy_directions.
        freqs: 1-based frequencies.
    returns:
        tensor of the same shape as logits.
    """
    rows = [2 * (f - 1) + j for f in freqs for j in (0, 1)]
    D = dirs[rows]
    return D.T @ (D @ logits)


@t.no_grad()
def progress_measures(model, grid_tokens, grid_labels, is_train, n_iter, n_keys=5, generator=None):
    """compute the hop-1 progress measures for one model state.

    args:
        model: LoopedTransformer.
        grid_tokens: token ids of all p*p hop-1 prompts in pair-index order.
        grid_labels: their answers.
        is_train: bool tensor marking train pairs.
        n_iter: depth to run the model at.
        n_keys: number of key frequencies.
        generator: torch generator for a random start, or None.
    returns:
        dict of json-ready progress measures.
    """
    p = model.cfg.p
    logits = model(grid_tokens, n_iter, generator=generator).float()
    basis = fourier_basis(p, logits.device)
    dirs = xpy_directions(basis)

    coef = dirs @ logits
    freq_norms = (coef[0::2].pow(2).sum(-1) + coef[1::2].pow(2).sum(-1)).sqrt()
    keys = (freq_norms.argsort(descending=True)[:n_keys] + 1).tolist()

    comp = key_component(logits, dirs, keys)
    # bias correction: add back the input-independent mean of what was removed
    restricted = comp + (logits - comp).mean(dim=0, keepdim=True)
    excluded = logits - comp

    ce = cross_entropy_high_precision
    W_E = model.embed.W_E[:, :p]
    W_U = model.unembed.W_U
    out = {
        # nanda's train/test split of the grid: test is every non-train pair
        "grid_train_loss": float(ce(logits[is_train], grid_labels[is_train])),
        "grid_test_loss": float(ce(logits[~is_train], grid_labels[~is_train])),
        "grid_test_acc": float((logits[~is_train].argmax(-1) == grid_labels[~is_train]).float().mean()),
        "key_freqs": keys,
        "logit_freq_norms": [round(float(v), 6) for v in freq_norms],
        "restricted_loss": float(ce(restricted, grid_labels)),
        "excluded_loss": float(ce(excluded[is_train], grid_labels[is_train])),
        "excluded_loss_per_freq": [float(ce((logits - key_component(logits, dirs, [f]))[is_train],
                                            grid_labels[is_train])) for f in keys],
        "gini_W_E": gini(component_norms(W_E, basis)),
        "gini_W_U": gini(component_norms(W_U, basis)),
        "sum_sq_weights": float(sum(prm.pow(2).sum() for prm in model.parameters())),
        # rms norm gains, reported apart: their scale can trade off against the next matrix's
        "sum_sq_gains": float(sum(prm.pow(2).sum() for n, prm in model.named_parameters() if "w_rms" in n)),
    }
    if model.cfg.n_coda == 0:
        # neuron-logit map of the last mlp; only defined when nothing sits between it and the unembedding
        W_out = model.core[-1][-1].mlp.W_out
        out["gini_W_L"] = gini(component_norms((W_U.T @ W_out).T, basis))
    return out
