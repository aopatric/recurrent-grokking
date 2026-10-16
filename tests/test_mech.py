"""tests for the mechanistic-analysis helpers."""

import numpy as np
import torch as t

import mech
import models
import progress

P = 11


def test_freq_norms_finds_a_pure_frequency():
    """a matrix built from frequency 3 has all its non-constant norm at frequency 3."""
    x = t.arange(P, dtype=t.float32)
    W = t.stack([t.cos(2 * np.pi * 3 * x / P), t.sin(2 * np.pi * 3 * x / P)])
    norms = mech.freq_norms(W, progress.fourier_basis(P))
    assert norms.argmax() == 3 and norms[3] ** 2 > 0.999 * norms.pow(2).sum()


def test_harmonics_fold():
    """multiples fold into 1..p//2."""
    # 4, 8 and 12 fold to 4, 3 and 1
    assert mech.harmonics([4], 3, P) == [1, 3, 4]
    assert mech.fold(12, P) == 1


def test_neuron_frequency_and_projection():
    """a neuron equal to cos(w(a+b)) is assigned frequency w and survives projection onto w unchanged."""
    a, b = t.meshgrid(t.arange(P), t.arange(P), indexing="ij")
    grid = t.cos(2 * np.pi * 2 * (a + b) / P)[..., None].float()
    basis = progress.fourier_basis(P)
    best, frac = mech.neuron_frequencies(mech.fourier_2d(grid, basis))
    assert best.item() == 2 and frac.item() > 0.999
    assert t.allclose(mech.keep_frequencies_2d(grid, basis, [2]), grid, atol=1e-4)
    assert mech.keep_frequencies_2d(grid, basis, [3]).abs().max() < 1e-4


def test_head_scores_match_the_forward_pass():
    """scores computed from the weights reproduce the hooked attention pattern of a one-layer nope model."""
    t.manual_seed(0)
    m = models.LoopedTransformer(models.ModelConfig(p=P, d_model=16, num_heads=2, d_head=8, d_mlp=32))
    x = t.tensor([[3, 7, 1, P]])
    _, cache = mech.run_cache(m, x)
    attn = cache[("core.0.0.attn.hook_attn", 0)][0, :, -1]
    s = mech.head_scores(m.core[0][0].attn, m.embed.W_E[:, P], m.embed.W_E[:, x[0]].T)
    assert t.allclose(s.softmax(-1), attn, atol=1e-5)


def test_variance_explained_and_clock_fit():
    """group means explain a function of the label fully, and a pure clock is fitted exactly."""
    labels = t.randint(0, P, (200,))
    acts = t.stack([labels.float() ** 2, t.randn(200)], 1)
    frac = mech.variance_explained_by(acts, labels)
    assert frac[0] > 0.999 and frac[1] < 0.2
    c = t.arange(P)
    logits = 5 * t.cos(2 * np.pi * 4 * (labels[:, None] - c[None, :]) / P)
    _, r2 = mech.clock_fit(logits, labels, [4], P)
    assert r2 > 0.999
