"""tests for the depth rules."""

import pytest

import depth as dp


def test_fixed_depth():
    """a fixed rule always returns T."""
    rule = dp.make_depth({"rule": "fixed", "T": 6})
    rng = dp.depth_rng(0)
    assert {rule.sample(rng) for _ in range(50)} == {6} and rule.eval_depth() == 6


def test_sampled_depth_mean_and_cap():
    """huginn's sampler has mean `mean + 1` when uncapped, and never exceeds the cap."""
    rng = dp.depth_rng(0)
    free = dp.SampledDepth(mean=8, sigma=0.5, cap=10_000)
    draws = [free.sample(rng) for _ in range(20_000)]
    assert min(draws) >= 1
    assert abs(sum(draws) / len(draws) - 9) < 0.2
    capped = dp.SampledDepth(mean=8, sigma=0.5, cap=10)
    assert max(capped.sample(rng) for _ in range(5_000)) == 10


def test_sampling_is_seeded():
    """the same seed gives the same depth sequence."""
    rule = dp.SampledDepth(mean=8, sigma=0.5, cap=24)
    a, b = dp.depth_rng(3), dp.depth_rng(3)
    assert [rule.sample(a) for _ in range(100)] == [rule.sample(b) for _ in range(100)]


@pytest.mark.parametrize("cfg", [
    {"rule": "learned"},
    {"rule": "fixed", "T": 4, "mean": 2},
    {"rule": "fixed", "T": 0},
    {"rule": "sampled", "mean": 8, "sigma": 0, "cap": 24},
])
def test_bad_depth_configs(cfg):
    """unknown rules, unknown keys and out-of-range values are rejected."""
    with pytest.raises(ValueError):
        dp.make_depth(cfg)


def test_min_depth_only_raises_shallow_draws():
    """a minimum depth lifts draws below it and leaves every other draw, and the random sequence, unchanged."""
    free = dp.make_depth({"rule": "sampled", "mean": 2, "sigma": 0.5, "cap": 24})
    floored = dp.make_depth({"rule": "sampled", "mean": 2, "sigma": 0.5, "cap": 24, "min_depth": 2})
    a, b = dp.depth_rng(0), dp.depth_rng(0)
    pairs = [(free.sample(a), floored.sample(b)) for _ in range(2_000)]
    assert any(x == 1 for x, _ in pairs)
    assert all(y == max(x, 2) for x, y in pairs)


def test_fixed_depth_accepts_grad_iters():
    """the fixed rule takes a truncated-backprop window like the sampled rule."""
    assert dp.make_depth({"rule": "fixed", "T": 8, "grad_iters": 4}).eval_depth() == 8


@pytest.mark.parametrize("cfg", [
    {"rule": "sampled", "mean": 8, "sigma": 0.5, "cap": 24, "min_depth": 0},
    {"rule": "sampled", "mean": 8, "sigma": 0.5, "cap": 24, "min_depth": 25},
    {"rule": "fixed", "T": 8, "min_depth": 2},
])
def test_bad_min_depth(cfg):
    """a minimum depth outside [1, cap], or on the fixed rule, is rejected."""
    with pytest.raises(ValueError):
        dp.make_depth(cfg)
