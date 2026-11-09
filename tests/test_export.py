"""tests for the export plan: which runs the figures need and which gpu each trained on."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import export  # noqa: E402


def test_gpu_groups_match_the_committed_runs():
    """every planned training run's gpu group matches the gpu its committed meta.json records."""
    runs, _, _ = export.plan()
    for name in runs:
        if "_from" in name:
            continue
        device = json.loads((ROOT / "results" / "runs" / name / "meta.json").read_text())["device"]
        assert export.gpu_of(name) in device, name


def test_gpu_groups_cover_the_plan():
    """the two groups split the planned training runs: 20 from the seed-41 sweep, 15 later runs."""
    runs, _, _ = export.plan()
    on_5090, on_4090 = export.runs_on("5090"), export.runs_on("4090")
    assert (len(on_5090), len(on_4090)) == (20, 15)
    assert sorted(on_5090 + on_4090) == sorted(r for r in runs if "_from" not in r)
