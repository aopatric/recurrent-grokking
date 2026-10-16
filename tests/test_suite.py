"""tests for the run-list runner and the verdict summaries."""

import json
import os
import signal
import threading
import time

import pytest
import torch as t
import yaml

import suite
import train
import verdicts
from test_train import config, data_root, run_entry  # noqa: F401


def write_config(tmp_path, data_root, *entries):
    """write a training config file whose runs go under tmp_path.

    args:
        tmp_path: directory for the config and run outputs.
        data_root: directory holding the generated dataset.
        *entries: run entries.
    returns:
        (path to the config file, config dict).
    """
    cfg = config(tmp_path, data_root, *entries)
    path = tmp_path / "train.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path, cfg


def test_plan_starts_resumes_and_skips(tmp_path, data_root):
    """new runs start, stopped ones resume, finished or far-enough ones are skipped, unknown names fail."""
    entries = [run_entry(n) for n in ("new", "stopped", "done")]
    _, cfg = write_config(tmp_path, data_root, *entries)
    train.train(entries[1], cfg, "cpu", until=2, log=lambda _: None)
    train.train(entries[2], cfg, "cpu", log=lambda _: None)
    assert dict(suite.plan(["stopped"], cfg))["stopped"].endswith("probably still training")
    got = dict(suite.plan(["new", "stopped", "done"], cfg, active_seconds=0))
    assert got["new"] == "start" and got["stopped"] == "resume" and got["done"].startswith("skip: complete")
    assert dict(suite.plan(["stopped"], cfg, until=2, active_seconds=0))["stopped"].startswith("skip: already at step 2")
    with pytest.raises(train.RunConfigError, match="no runs named"):
        suite.plan(["missing"], cfg)
    with pytest.raises(train.RunConfigError, match="twice"):
        suite.plan(["new", "new"], cfg)


def test_command_pins_one_gpu_per_job():
    """a gpu job sees only its own gpu; a cpu job sees none."""
    cmd, env = suite.command("c.yaml", "r", "resume", "3", until=100)
    assert env["CUDA_VISIBLE_DEVICES"] == "3"
    assert "--resume" in cmd and cmd[cmd.index("--until") + 1] == "100" and cmd[cmd.index("--device") + 1] == "cuda"
    cmd, env = suite.command("c.yaml", "r", "start", "cpu")
    assert env["CUDA_VISIBLE_DEVICES"] == "" and "--resume" not in cmd and "--until" not in cmd


def test_run_all_until_then_finish(tmp_path, data_root):
    """two runs share one device slot, stop at until, and a second call finishes them."""
    path, cfg = write_config(tmp_path, data_root, run_entry("a"), run_entry("b"))
    first = suite.run_all(path, ["a", "b"], ["cpu"], until=4, poll=0.1, log=lambda _: None)
    assert first == {"a": "stopped at step 4", "b": "stopped at step 4"}
    second = suite.run_all(path, ["a", "b"], ["cpu", "cpu"], poll=0.1, active_seconds=0, log=lambda _: None)
    assert second == {"a": "complete", "b": "complete"}
    assert "resumed a at step 4" in (tmp_path / "runs" / "a.log").read_text()
    assert t.load(tmp_path / "runs" / "a" / "latest.pt", weights_only=False)["step"] == 6


def test_load_suite(tmp_path):
    """a suite is read by name, and a missing one is an error."""
    path = tmp_path / "suites.yaml"
    path.write_text(yaml.safe_dump({"pair": ["a", "b"]}))
    assert suite.load_suite(path, "pair") == ["a", "b"]
    with pytest.raises(train.RunConfigError):
        suite.load_suite(path, "other")


def evals(vals, trains=None, restricted=None, every=1000):
    """fake eval rows with the given mean val accuracies.

    args:
        vals: val accuracy per eval.
        trains: train accuracy per eval, default 1.
        restricted: restricted loss per eval, default 4.
        every: steps between evals.
    returns:
        list of metrics rows.
    """
    n = len(vals)
    trains = trains or [1.0] * n
    restricted = restricted or [4.0] * n
    return [{"kind": "eval", "step": i * every, "val": {"1": {"acc": v}, "2": {"acc": v}},
             "train": {"1": {"acc": tr}}, "progress": {"restricted_loss": r}}
            for i, (v, tr, r) in enumerate(zip(vals, trains, restricted))]


def test_verdict_groks_and_stability():
    """t_grok is the first eval at 0.9, and stability counts the window after it."""
    vals = [0.0] * 5 + [0.95] + [0.95, 0.5] * 10
    v = verdicts.verdict(evals(vals))
    assert v["verdict"] == "groks" and v["t_grok"] == 5000
    # window 6k..15k holds 10 evals, alternating 0.95 and 0.5
    assert v["stable"] == 0.5 and v["window_end"] == 15000


def test_verdict_slow_versus_fails():
    """a run that hasn't grokked is slow while restricted loss keeps falling, and fails once it is flat."""
    falling = verdicts.verdict(evals([0.05] * 21, restricted=[4.0 - 0.1 * i for i in range(21)]))
    flat = verdicts.verdict(evals([0.05] * 21, trains=[0.6] * 21))
    assert falling["verdict"] == "slow" and falling["fits_train"]
    assert flat["verdict"] == "fails" and not flat["fits_train"]


def test_verdict_judges_at_the_given_step():
    """grokking after the judging step doesn't count."""
    rows = evals([0.05] * 25 + [0.95])
    assert verdicts.verdict(rows, at=20000)["verdict"] != "groks"
    assert verdicts.verdict(rows, at=30000)["t_grok"] == 25000


def test_fixed_point():
    """a sweep is a fixed point when pass 16 matches the eval depth at every hop."""
    flat = {"kind": "sweep", "depth": 8, "by_T": {"val": {"1": [0.5] * 7 + [0.99] * 25}}}
    peaked = {"kind": "sweep", "depth": 8, "by_T": {"val": {"1": [0.5] * 7 + [0.99] + [0.3] * 24}}}
    assert verdicts.fixed_point(flat) is True and verdicts.fixed_point(peaked) is False
    assert verdicts.fixed_point({"kind": "sweep", "depth": 1, "by_T": {}}) is None


def test_verdicts_cli_on_a_real_run(tmp_path, data_root, capsys):
    """the cli prints a table row for a trained run."""
    entry = run_entry()
    run = train.train(entry, config(tmp_path, data_root, entry), "cpu", log=lambda _: None)
    assert verdicts.main([str(run)]) == 0
    assert "| `r` |" in capsys.readouterr().out
    assert verdicts.main(["--json", str(run)]) == 0
    assert set(json.loads(capsys.readouterr().out)["r"]) >= {"verdict", "fixed_point"}


def test_stop_pauses_running_jobs(tmp_path, data_root):
    """a sigterm to the runner pauses its job at a resumable step and launches nothing else."""
    slow = {**run_entry()["train"], "steps": 100_000, "eval_every": 50, "sweep_every": 100_000}
    path, _ = write_config(tmp_path, data_root, run_entry("a", train=slow), run_entry("b"))
    log_file = tmp_path / "runs" / "a.log"

    def stop_once_training():
        """signal the runner once the job has logged a training step."""
        while "step     50" not in (log_file.read_text() if log_file.exists() else ""):
            time.sleep(0.05)
        os.kill(os.getpid(), signal.SIGTERM)

    threading.Thread(target=stop_once_training, daemon=True).start()
    out = suite.run_all(path, ["a", "b"], ["cpu"], poll=0.1, log=lambda _: None)
    assert out["a"].startswith("paused at step") and "b" not in out
    assert not (tmp_path / "runs" / "b").exists()
    assert signal.getsignal(signal.SIGTERM) == signal.SIG_DFL
