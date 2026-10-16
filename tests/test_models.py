"""tests for the model components."""

import pytest
import torch as t

import models


def _looped(hp, x, n_iter):
    """run x through a hook point n_iter times, tagging each pass, scaling so passes differ.

    args:
        hp: hook point.
        x: input tensor.
        n_iter: number of passes.
    returns:
        x times 2 ** n_iter.
    """
    for i in range(n_iter):
        models.set_iteration(hp, i)
        x = hp(x) * 2
    models.set_iteration(hp, None)
    return x


def test_forward_hooks_see_each_iteration():
    """forward hooks get the name and the iteration of every loop pass."""
    hp = models.HookPoint()
    hp.give_name("core.hook")
    seen = []
    hp.add_hook(lambda x, name, iteration: seen.append((name, iteration, x.item())))
    _looped(hp, t.tensor(1.0), 3)
    assert seen == [("core.hook", 0, 1.0), ("core.hook", 1, 2.0), ("core.hook", 2, 4.0)]


def test_backward_hooks_keep_their_iteration():
    """gradient hooks are tagged with the pass they came from, not the last one."""
    hp = models.HookPoint()
    hp.give_name("core.hook")
    seen = {}
    hp.add_hook(lambda g, name, iteration: seen.__setitem__(iteration, g.item()), dir="bwd")
    x = t.tensor(1.0, requires_grad=True)
    _looped(hp, x, 3).backward()
    # output = 8x, so d out / d (input to pass i) = 2^(3 - i)
    assert seen == {0: 8.0, 1: 4.0, 2: 2.0}


def test_remove_hooks():
    """removed hooks no longer fire in either direction."""
    hp = models.HookPoint()
    hp.give_name("h")
    calls = []
    hp.add_hook(lambda x, name, iteration: calls.append("fwd"))
    hp.add_hook(lambda g, name, iteration: calls.append("bwd"), dir="bwd")
    hp.remove_hooks("both")
    hp(t.tensor(1.0, requires_grad=True)).backward()
    assert calls == []


def test_rms_norm_unit_rms():
    """rms norm gives unit root-mean-square per position at the initial gain."""
    x = t.randn(4, 5, 16) * 7 + 3
    out = models.RMSNorm(16)(x)
    assert t.allclose(out.pow(2).mean(-1), t.ones(4, 5), atol=1e-4)


# looped transformer

SMALL = dict(p=7, d_model=16, num_heads=2, d_head=8, d_mlp=32)


def small_model(**over):
    """build a tiny model on cpu with fixed init.

    args:
        **over: ModelConfig fields to override.
    returns:
        LoopedTransformer.
    """
    t.manual_seed(0)
    return models.LoopedTransformer(models.ModelConfig(**{**SMALL, **over}))


def huginn(**over):
    """tiny huginn-style model: prelude, sandwich norm, concat injection, random start, coda.

    args:
        **over: ModelConfig fields to override.
    returns:
        LoopedTransformer.
    """
    return small_model(norm="rms_sandwich", n_prelude=1, injection="concat", state_init="random", n_coda=1, **over)


def tokens(batch, hops, p=7):
    """random prompts with hops + 1 operands and a final "=" (id p).

    args:
        batch: number of prompts.
        hops: hops per prompt.
        p: modulus.
    returns:
        int tensor of shape (batch, hops + 2).
    """
    g = t.Generator().manual_seed(1)
    x = t.randint(0, p, (batch, hops + 1), generator=g)
    return t.cat([x, t.full((batch, 1), p)], dim=1)


def test_anchor_matches_nanda_shapes():
    """the anchor has nanda's 1-layer parameter shapes at nanda's widths."""
    m = models.LoopedTransformer(models.ModelConfig(p=113, pos="learned", n_ctx=3))
    shapes = {n: tuple(v.shape) for n, v in m.named_parameters()}
    assert shapes == {
        "embed.W_E": (128, 114), "pos_embed.W_pos": (3, 128),
        "core.0.0.attn.W_K": (4, 32, 128), "core.0.0.attn.W_Q": (4, 32, 128),
        "core.0.0.attn.W_V": (4, 32, 128), "core.0.0.attn.W_O": (128, 128),
        "core.0.0.mlp.W_in": (512, 128), "core.0.0.mlp.b_in": (512,),
        "core.0.0.mlp.W_out": (128, 512), "core.0.0.mlp.b_out": (128,),
        "unembed.W_U": (128, 113),
    }


def test_core_hooks_tagged_every_iteration():
    """every hook in the loop fires once per pass with tags 0..T-1; prelude hooks report None."""
    m = huginn()
    seen = []
    for hp in m.hook_points():
        hp.add_hook(lambda x, name, iteration: seen.append((name, iteration)))
    m(tokens(2, 2), 3)
    core = [(n, i) for n, i in seen if n.startswith("core.") or n in ("hook_state", "hook_inject")]
    by_name = {}
    for n, i in core:
        by_name.setdefault(n, []).append(i)
    assert by_name and all(v == [0, 1, 2] for v in by_name.values())
    assert {i for n, i in seen if n.startswith("prelude.") or n == "hook_embed"} == {None}
    # the default readout reads the last pass
    assert {i for n, i in seen if n.startswith("coda.")} == {2}


def test_return_all_tags_coda_and_matches_default():
    """per-pass readouts have shape (batch, T, p), coda hooks carry each pass, and the last equals the default."""
    m = huginn()
    x = tokens(3, 2)
    seen = set()
    m.coda[0].hook_resid_post.add_hook(lambda a, name, iteration: seen.add(iteration))
    allr = m(x, 4, return_all=True, generator=t.Generator().manual_seed(5))
    last = m(x, 4, generator=t.Generator().manual_seed(5))
    assert allr.shape == (3, 4, 7)
    assert t.allclose(allr[:, -1], last)
    assert seen == {0, 1, 2, 3}


def test_unshared_core():
    """the unshared baseline holds T copies of the core and refuses more passes."""
    shared = small_model()
    unshared = small_model(core_shared=False, n_core_copies=4)
    count = lambda m: sum(v.numel() for n, v in m.named_parameters() if n.startswith("core."))  # noqa: E731
    assert count(unshared) == 4 * count(shared)
    unshared(tokens(2, 1), 4)
    with pytest.raises(ValueError):
        unshared(tokens(2, 1), 5)


def test_truncated_backprop():
    """with grad_iters = b, only the last b passes get gradients."""
    m = huginn()
    seen = []
    m.hook_state.add_hook(lambda g, name, iteration: seen.append(iteration), dir="bwd")
    m(tokens(2, 2), 5, grad_iters=2, generator=t.Generator().manual_seed(0)).sum().backward()
    assert sorted(seen) == [3, 4]
    assert m.embed.W_E.grad is not None


def test_nope_runs_any_length_and_is_causal():
    """without positions the model takes any prompt length, and later tokens never change earlier positions."""
    m = small_model()
    x = tokens(2, 9)
    cache = {}
    m.cache_all(cache)
    m(x, 2)
    first = cache[("hook_state", 1)][:, 0].clone()
    x2 = x.clone()
    x2[:, 3] = (x2[:, 3] + 1) % 7
    m(x2, 2)
    assert t.allclose(cache[("hook_state", 1)][:, 0], first)


def test_model_config_rejects_bad_settings():
    """invalid model settings are rejected."""
    with pytest.raises(ValueError):
        models.ModelConfig.from_dict({**SMALL, "norm": "layer"})
    with pytest.raises(ValueError):
        models.ModelConfig.from_dict({**SMALL, "state_init": "random"})
    with pytest.raises(ValueError):
        models.ModelConfig.from_dict({**SMALL, "colour": 1})


def test_state_norm_toggle():
    """state_norm adds an rms norm on the carried state, so it stays at unit rms however long the loop runs."""
    plain, normed = small_model(), small_model(state_norm=True)
    assert plain.core_norm is None and normed.core_norm is not None
    cache = {}
    normed.cache_all(cache)
    normed(tokens(2, 2), 6)
    rms = cache[("hook_state", 5)].pow(2).mean(-1)
    assert t.allclose(rms, t.ones_like(rms), atol=1e-3)
