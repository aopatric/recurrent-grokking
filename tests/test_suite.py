"""tests for the run-list runner and the verdict summaries."""

import json
import os
import signal
import socket
import subprocess
import threading
import time

import pytest
import torch as t
import yaml

import smoke
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


def claim_for(run_dir, pid):
    """write a training claim on a run for some process, as train.py does.

    args:
        run_dir: run directory.
        pid: process id to record.
    """
    (run_dir / "train.pid").write_text(json.dumps({"pid": pid, "host": socket.gethostname()}))


def dead_pid():
    """a process id that no longer exists.

    returns:
        pid of a finished, reaped child.
    """
    proc = subprocess.Popen(["true"])
    proc.wait()
    return proc.pid


def test_plan_starts_resumes_and_skips(tmp_path, data_root):
    """new runs start, stopped ones resume, finished, claimed, diverged or far-enough ones are skipped."""
    entries = [run_entry(n) for n in ("new", "stopped", "done")]
    _, cfg = write_config(tmp_path, data_root, *entries)
    train.train(entries[1], cfg, "cpu", until=2, log=lambda _: None)
    train.train(entries[2], cfg, "cpu", log=lambda _: None)
    # a run stopped a moment ago is resumable straight away
    got = dict(suite.plan(["new", "stopped", "done"], cfg))
    assert got["new"] == "start" and got["stopped"] == "resume" and got["done"].startswith("skip: complete")
    assert dict(suite.plan(["stopped"], cfg, until=2))["stopped"].startswith("skip: already at step 2")
    stopped = tmp_path / "runs" / "stopped"
    claim_for(stopped, os.getppid())
    assert dict(suite.plan(["stopped"], cfg))["stopped"] == f"skip: being trained by process {os.getppid()}"
    claim_for(stopped, dead_pid())
    assert dict(suite.plan(["stopped"], cfg))["stopped"] == "resume"
    meta = json.loads((stopped / "meta.json").read_text())
    (stopped / "meta.json").write_text(json.dumps({**meta, "diverged_at": 2}))
    assert dict(suite.plan(["stopped"], cfg))["stopped"] == "skip: diverged at step 2"
    with pytest.raises(train.RunConfigError, match="no runs named"):
        suite.plan(["missing"], cfg)
    with pytest.raises(train.RunConfigError, match="twice"):
        suite.plan(["new", "new"], cfg)


def test_command_pins_one_gpu_per_job(monkeypatch):
    """a gpu job sees only its own gpu, counted among the already-visible ones; a cpu job sees none."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    cmd, env = suite.command("c.yaml", "r", "resume", "3", until=100)
    assert env["CUDA_VISIBLE_DEVICES"] == "3"
    assert "--resume" in cmd and cmd[cmd.index("--until") + 1] == "100" and cmd[cmd.index("--device") + 1] == "cuda"
    cmd, env = suite.command("c.yaml", "r", "start", "cpu")
    assert env["CUDA_VISIBLE_DEVICES"] == "" and "--resume" not in cmd and "--until" not in cmd
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,6")
    assert suite.command("c.yaml", "r", "start", "1")[1]["CUDA_VISIBLE_DEVICES"] == "6"


def test_run_all_until_then_finish(tmp_path, data_root):
    """two runs share one device slot, stop at until, and a second call finishes them."""
    path, cfg = write_config(tmp_path, data_root, run_entry("a"), run_entry("b"))
    first = suite.run_all(path, ["a", "b"], ["cpu"], until=4, poll=0.1, log=lambda _: None)
    assert first == {"a": "stopped at step 4", "b": "stopped at step 4"}
    second = suite.run_all(path, ["a", "b"], ["cpu", "cpu"], poll=0.1, log=lambda _: None)
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


def test_verdict_diverged_and_seed_summary():
    """a run that diverged before grokking says so, and seeds of one config combine into one row."""
    assert verdicts.verdict(evals([0.05] * 5), diverged_at=4500)["verdict"] == "diverged"
    results = {"x_s41": verdicts.verdict(evals([0.0] * 5 + [0.95] * 15)),
               "x_s42": verdicts.verdict(evals([0.0] * 9 + [0.95] * 11)),
               "x_s43": verdicts.verdict(evals([0.05] * 21, trains=[0.6] * 21)),
               "y_s41": verdicts.verdict(evals([0.05] * 5), diverged_at=4500)}
    summary = verdicts.summarize(results)
    assert summary["x"]["groks"] == "2/3" and summary["x"]["t_grok_median"] == 7000 and summary["x"]["fails"] == 1
    assert summary["y"]["diverged"] == 1 and summary["y"]["groks"] == "0/1"
    assert "| `x` |" in verdicts.table(summary, "config")


def test_load_rows_drops_a_cut_off_last_line(tmp_path):
    """a metrics file copied mid-write loses only its unfinished last line; earlier damage is an error."""
    rows = evals([0.1, 0.2])
    text = "".join(json.dumps(r) + "\n" for r in rows)
    (tmp_path / "metrics.jsonl").write_text(text + '{"kind": "ev')
    assert verdicts.load_rows(tmp_path) == rows
    (tmp_path / "metrics.jsonl").write_text('{"kind": "ev\n' + text)
    with pytest.raises(ValueError):
        verdicts.load_rows(tmp_path)


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
    out = suite.run_all(path, ["a", "b"], ["cpu"], poll=0.1, grace=0.5, log=lambda _: None)
    assert out["a"].startswith("paused at step") and "b" not in out
    assert not (tmp_path / "runs" / "b").exists()
    assert signal.getsignal(signal.SIGTERM) == signal.SIG_DFL


def test_stop_right_after_launch_still_pauses(tmp_path, data_root):
    """a stop that arrives as a job starts waits out the grace period, so the job pauses instead of dying."""
    slow = {**run_entry()["train"], "steps": 100_000, "eval_every": 50, "sweep_every": 100_000}
    path, _ = write_config(tmp_path, data_root, run_entry("a", train=slow))

    def stop_on_launch(line):
        """signal the runner as soon as it reports the launch."""
        if "start on cpu" in line:
            os.kill(os.getpid(), signal.SIGTERM)

    out = suite.run_all(path, ["a"], ["cpu"], poll=0.1, grace=8, log=stop_on_launch)
    assert out["a"].startswith("paused at step")
    assert dict(suite.plan(["a"], yaml.safe_load(path.read_text())))["a"] == "resume"


def test_smoke_tries_each_config_once(tmp_path, data_root):
    """runs that differ only in seed share one smoke test, which reports timings without leaving files."""
    entries = [run_entry("a"), {**run_entry("b"), "seed": 1}, run_entry("c", depth={"rule": "fixed", "T": 3})]
    path, cfg = write_config(tmp_path, data_root, *entries)
    assert [e["name"] for e in smoke.distinct(["a", "b", "c"], cfg)] == ["a", "c"]
    r = smoke.smoke(entries[0], cfg, "cpu", steps=3)
    assert r["peak_gb"] is None and r["step_s"] >= 0 and r["hours_20k"] > 0
    assert not (tmp_path / "runs").exists() or not any((tmp_path / "runs").iterdir())
    assert smoke.main(["--config", str(path), "--runs", "a", "b", "--steps", "2", "--device", "cpu"]) == 0


def test_claims_ignore_a_reused_pid(tmp_path):
    """a claim whose pid now belongs to a process started at another time is stale."""
    path = tmp_path / "train.pid"
    parent = os.getppid()
    path.write_text(json.dumps({"pid": parent, "host": socket.gethostname(), "start": train._start_time(parent)}))
    assert train.live_claim(path) == parent
    path.write_text(json.dumps({"pid": parent, "host": socket.gethostname(), "start": "1"}))
    assert train.live_claim(path) is None
    train.write_claim(path)
    assert train.live_claim(path) is None
    train.release_claim(path)
    assert not path.exists()


def test_one_runner_per_output_root_and_cli_controls(tmp_path, data_root, capsys):
    """a second runner on the same output root is refused, and --running / --stop report and signal it."""
    path, cfg = write_config(tmp_path, data_root, run_entry("a"))
    (tmp_path / "runs").mkdir()
    claim_for_runner = tmp_path / "runs" / "suite.pid"
    claim_for_runner.write_text(json.dumps({"pid": os.getppid(), "host": socket.gethostname()}))
    with pytest.raises(train.RunConfigError, match="already going"):
        suite.run_all(path, ["a"], ["cpu"], poll=0.1, log=lambda _: None)
    assert suite.main(["--config", str(path), "--running"]) == 0
    assert f"runner: process {os.getppid()}" in capsys.readouterr().out
    claim_for_runner.unlink()
    assert suite.main(["--config", str(path), "--running"]) == 1
    assert suite.main(["--config", str(path), "--stop"]) == 1
    # a live job is listed with its latest eval
    (tmp_path / "runs" / "y").mkdir()
    claim_for(tmp_path / "runs" / "y", os.getppid())
    (tmp_path / "runs" / "y" / "metrics.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in evals([0.1, 0.25])) + '{"kind": "sw')
    assert suite.main(["--config", str(path), "--running"]) == 0
    assert f"training: y (process {os.getppid()}): step 1000, val 0.250" in capsys.readouterr().out
    (tmp_path / "runs" / "y" / "train.pid").unlink()
    # a job whose runner is gone is stopped directly
    orphan = subprocess.Popen(["sleep", "30"])
    (tmp_path / "runs" / "x").mkdir()
    claim_for(tmp_path / "runs" / "x", orphan.pid)
    assert suite.main(["--config", str(path), "--stop"]) == 0
    assert orphan.wait(timeout=10) == -signal.SIGTERM
    (tmp_path / "runs" / "x" / "train.pid").unlink()
    assert suite.run_all(path, ["a"], ["cpu"], poll=0.1, log=lambda _: None) == {"a": "complete"}
    assert not claim_for_runner.exists()
