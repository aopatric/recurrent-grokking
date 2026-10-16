"""depth rules: how many core passes each training step uses."""

import math
import random


class FixedDepth:
    """always the same depth."""

    def __init__(self, T):
        """build the rule.

        args:
            T: number of core passes, >= 1.
        raises:
            ValueError: if T < 1.
        """
        if T < 1:
            raise ValueError("T must be >= 1")
        self.T = T

    def sample(self, rng):
        """pick the depth for one step.

        args:
            rng: random.Random (unused).
        returns:
            T.
        """
        return self.T

    def update(self, r, loss):
        """take feedback after a step; fixed depth ignores it.

        args:
            r: depth used.
            loss: loss at that depth.
        """

    def eval_depth(self):
        """depth used for routine evaluation.

        returns:
            T.
        """
        return self.T

    def state_dict(self):
        """state to save for resuming.

        returns:
            empty dict; nothing changes during training.
        """
        return {}

    def load_state_dict(self, state):
        """restore saved state.

        args:
            state: dict from state_dict.
        """


class SampledDepth:
    """huginn's log-normal poisson depth: tau ~ normal(log(mean) - sigma^2 / 2, sigma), r = poisson(e^tau) + 1."""

    def __init__(self, mean, sigma, cap, min_depth=1):
        """build the rule.

        args:
            mean: target mean depth.
            sigma: log-normal spread (huginn uses 0.5).
            cap: largest depth; bigger draws are clipped to it.
            min_depth: smallest depth; smaller draws are raised to it.
        raises:
            ValueError: if the settings are out of range.
        """
        if mean < 1 or sigma <= 0 or cap < 1:
            raise ValueError("need mean >= 1, sigma > 0 and cap >= 1")
        if not 1 <= min_depth <= cap:
            raise ValueError("need 1 <= min_depth <= cap")
        self.mean, self.sigma, self.cap, self.min_depth = mean, sigma, cap, min_depth

    def sample(self, rng):
        """pick the depth for one step.

        args:
            rng: random.Random.
        returns:
            depth in [min_depth, cap].
        """
        tau = rng.gauss(math.log(self.mean) - self.sigma ** 2 / 2, self.sigma)
        # clip rather than redraw, so the random sequence matches a run without a minimum
        return max(min(_poisson(rng, math.exp(tau)) + 1, self.cap), self.min_depth)

    def update(self, r, loss):
        """take feedback after a step; a fixed distribution ignores it.

        args:
            r: depth used.
            loss: loss at that depth.
        """

    def eval_depth(self):
        """depth used for routine evaluation.

        returns:
            the target mean, rounded and capped.
        """
        return min(round(self.mean), self.cap)

    def state_dict(self):
        """state to save for resuming.

        returns:
            empty dict; the distribution never changes.
        """
        return {}

    def load_state_dict(self, state):
        """restore saved state.

        args:
            state: dict from state_dict.
        """


def _poisson(rng, lam):
    """draw from a poisson distribution by counting exponential arrivals.

    args:
        rng: random.Random.
        lam: rate, > 0.
    returns:
        non-negative int.
    """
    # python's random has no poisson; arrivals in [0, lam) of a unit-rate process are poisson(lam)
    n, total = 0, rng.expovariate(1.0)
    while total < lam:
        n += 1
        total += rng.expovariate(1.0)
    return n


def make_depth(cfg):
    """build a depth rule from its config section.

    args:
        cfg: mapping with rule = fixed (T) or sampled (mean, sigma, cap, optional min_depth),
            plus optional grad_iters.
    returns:
        the depth rule.
    raises:
        ValueError: if the rule or its keys are unknown.
    """
    keys = {"fixed": {"rule", "T", "grad_iters"},
            "sampled": {"rule", "mean", "sigma", "cap", "min_depth", "grad_iters"}}
    rule = cfg.get("rule")
    if rule not in keys:
        raise ValueError(f"depth.rule must be one of {sorted(keys)}, got {rule!r}")
    if unknown := set(cfg) - keys[rule]:
        raise ValueError(f"unknown depth keys for {rule}: {sorted(unknown)}")
    if rule == "fixed":
        return FixedDepth(cfg["T"])
    return SampledDepth(cfg["mean"], cfg["sigma"], cfg["cap"], cfg.get("min_depth", 1))


def depth_rng(seed):
    """make the random source for depth sampling.

    args:
        seed: run seed.
    returns:
        random.Random seeded from the run seed.
    """
    return random.Random(f"{seed}:depth")
