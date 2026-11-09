"""tests that the write-up's figures build from the committed results alone."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "figures"))

import ablations  # noqa: E402
import collapse  # noqa: E402
import common  # noqa: E402
import mechanism  # noqa: E402


@pytest.mark.parametrize("module, names", [
    (ablations, ["grokking_outcomes"]),
    (collapse, ["fixes"]),
    (mechanism, ["attention", "clocks", "compute_early"]),
])
def test_figures_build(module, names, tmp_path):
    """each figure script writes its figures and their notes from results/."""
    assert module.main(["--out", str(tmp_path)]) == 0
    for name in names:
        assert (tmp_path / f"{name}.png").stat().st_size > 0
        assert (tmp_path / f"{name}.md").stat().st_size > 0


@pytest.mark.parametrize("module, names", [
    (ablations, ["looping", "normalization", "injection", "weight_decay"]),
    (collapse, ["instability", "collapse", "two_phase"]),
])
def test_supporting_dynamics_figures_build(module, names, tmp_path):
    """--supporting also writes the training figures that aren't in the write-up."""
    assert module.main(["--out", str(tmp_path), "--supporting"]) == 0
    for name in names:
        assert (tmp_path / "supporting" / f"{name}.png").stat().st_size > 0


def test_supporting_mechanism_figures_build(tmp_path):
    """--supporting also writes the mechanism figures that aren't in the write-up."""
    assert mechanism.main(["--out", str(tmp_path), "--supporting"]) == 0
    for name in ["passes", "past_depth", "frequencies", "state_use", "dilution", "decoding"]:
        assert (tmp_path / "supporting" / f"{name}.png").stat().st_size > 0


def test_seed_table_has_both_seeds(tmp_path):
    """the seed table reports every section at both seeds."""
    mechanism.write_seed_table(str(common.RESULTS / "mechanism"), str(common.RESULTS / "runs"),
                               str(tmp_path / "seeds.md"))
    text = (tmp_path / "seeds.md").read_text()
    assert text.count("| | seed 41 | seed 42 | mean |") == len(mechanism.TABLE)


def test_branch_high_loss_counts():
    """the branch arms' high-loss step totals over 4000 steps match the recorded experiment."""
    expected = {"wd0": 212, "wd05": 705, "control": 1159, "wd2": 1535, "wd2_lr05": 1090, "nowd_embed": 984,
                "clip": 63, "lrdecay": 516, "clip_lrdecay": 27}
    for arm, total in expected.items():
        s = common.load_steps(common.run_dir(f"{collapse.SOURCE}_from{collapse.BRANCH_STEP}_{arm}"))
        _, counts = collapse.high_loss_counts(s["step"], s["loss"], collapse.BRANCH_LEN)
        assert counts.sum() == total


def test_median_keeps_steps_and_drops_spikes():
    """a rolling median keeps a clean drop and removes a one-point spike."""
    ys = [5, 5, 5, 5, 1, 1, 1, 9, 1, 1]
    out = common.median(ys, 3)
    assert list(out[:4]) == [5, 5, 5, 5] and out[7] == 1
