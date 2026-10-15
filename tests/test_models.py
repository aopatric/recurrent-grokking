"""tests for the model components."""

import torch as t

import models


def _looped(hp, x, n_iter):
    """run x through a hook point n_iter times, tagging each pass, scaling so passes differ."""
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
