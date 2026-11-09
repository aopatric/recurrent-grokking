"""tests for the mechanism probes in experiments/mechanism/."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch as t

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import analysis  # noqa: E402
import mech  # noqa: E402

DATA = REPO / "data" / "synthetic" / "p113_h3_x6"
RUN = REPO / "results" / "runs" / "t6_huginn_twophase_p113_h3_x6_s41"


def load_script(name):
    """import one of the hyphen-named mechanism scripts.

    args:
        name: script name without .py.
    returns:
        the module.
    """
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"),
                                                  REPO / "experiments" / "mechanism" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(not DATA.exists(), reason="needs the generated dataset (src/make_data.py)")
def test_probe_with_the_real_state_matches_the_model():
    """the state probe's own pass loop gives the model's forward-pass accuracy."""
    state_use = load_script("state-use")
    run = analysis.load_run(RUN)
    model = analysis.load_model(run, analysis.checkpoints(run)[-1][1])
    x, y = run["data"]["sets"]["val"]["2"]
    x, y = x[:200], y[:200]
    with t.no_grad():
        want = state_use.accuracy(model(x, run["T"], generator=analysis.eval_generator(run)), y)
    assert state_use.probe(model, run, x, y, run["T"], False, "cpu")["normal"] == want


@pytest.mark.skipif(not DATA.exists(), reason="needs the generated dataset (src/make_data.py)")
def test_eq_cache_matches_the_full_cache():
    """the internals script's partial cache holds the = slice of every activation the full cache holds."""
    internals = load_script("internals")
    run = analysis.load_run(RUN)
    model = analysis.load_model(run, analysis.checkpoints(run)[-1][1])
    x = run["data"]["sets"]["val"]["3"][0][:16]
    with t.no_grad():
        _, full = mech.run_cache(model, x, run["T"], generator=analysis.eval_generator(run))
        _, part = internals.eq_cache(model, x, run["T"], analysis.eval_generator(run),
                                     lambda n: n.endswith(("attn.hook_attn", "mlp.hook_post", "hook_resid_post")))
    assert len(part) == 3 * (2 + run["T"])
    for (name, it), got in part.items():
        want = full[(name, it)][:, :, -1] if name.endswith("hook_attn") else full[(name, it)][:, -1]
        assert t.equal(got, want), (name, it)


def test_past_depth_reads_the_requested_checkpoint():
    """past-depth reads the last checkpoint by default, the --at step when given, and refuses a missing step."""
    past_depth = load_script("past-depth")
    run_dir = REPO / "results" / "runs" / "t6_huginn_twophase_p113_h3_x6_s41"
    run = {"name": run_dir.name, "run_dir": run_dir, "meta": json.loads((run_dir / "meta.json").read_text())}
    cks = analysis.checkpoints(run)
    assert past_depth.pick(run, None) == cks[-1]
    assert past_depth.pick(run, cks[0][0]) == cks[0]
    with pytest.raises(ValueError):
        past_depth.pick(run, cks[0][0] + 1)
