"""tests for the training loop, run directory and progress measures."""

import json
import os
import signal
import socket

import pytest
import torch as t
import yaml

import analysis
import make_data
import models
import progress
import plot_progress
import tb_log
import train


TINY = {"d_model": 16, "num_heads": 2, "d_head": 8, "d_mlp": 32}


@pytest.fixture(scope="module")
def data_root(tmp_path_factory):
    """generate a small p=11 dataset with hops 1..2 trained and 3 held out."""
    root = tmp_path_factory.mktemp("data")
    entry = {
        "name": "t_p11", "p": 11, "ops": ["+"], "max_train_hops": 2, "max_test_hops": 3, "seed": 0,
        "hop1": {"train_frac": 0.3, "val_frac": 0.1},
        "multihop": {"train_per_hop": 20, "val_per_hop": 5, "max_space_frac": 0.5},
        "extrap": {"per_hop": 10}, "systematic": {"per_hop": 6},
    }
    cfg = root / "data.yaml"
    cfg.write_text(yaml.safe_dump({"output_root": str(root / "out"), "datasets": [entry]}))
    assert make_data.main(["--config", str(cfg), "--name", "t_p11"]) == 0
    return root / "out"


def run_entry(name="r", **over):
    """build a tiny run entry.

    args:
        name: run name.
        **over: top-level keys to override.
    returns:
        run entry dict.
    """
    e = {
        "name": name, "data": "t_p11", "seed": 0,
        "model": {"pos": "none", "norm": "none", "n_prelude": 0, "k": 1, "core_shared": True,
                  "injection": "none", "state_init": "embed", "n_coda": 0, **TINY},
        "depth": {"rule": "fixed", "T": 2},
        "train": {"steps": 6, "eval_every": 2, "sweep_every": 3, "eval_T_max": 3,
                  "n_checkpoints": 3, "chunk_size": 7, "resume_every": 3},
    }
    e.update(over)
    return e


def config(tmp_path, data_root, *entries):
    """make a config dict whose runs go under tmp_path.

    args:
        tmp_path: directory for run outputs.
        data_root: directory holding the generated dataset.
        *entries: run entries.
    returns:
        config dict.
    """
    cfg = {"output_root": str(tmp_path / "runs"), "data_root": str(data_root), "runs": list(entries)}
    train.validate_config(cfg)
    return cfg


def quiet(_):
    """discard log lines."""


def test_run_writes_every_artifact(tmp_path, data_root):
    """a short run writes config, meta, metrics, checkpoints and the resume file."""
    cfg = config(tmp_path, data_root, run_entry())
    run = train.train(cfg["runs"][0], cfg, "cpu", log=quiet)
    assert (run / "config.yaml").exists() and (run / "latest.pt").exists()
    assert (run / "checkpoints" / "final.pt").exists()
    meta = json.loads((run / "meta.json").read_text())
    assert meta["data"]["name"] == "t_p11" and "train.jsonl" in meta["data"]["sha256"]
    rows = [json.loads(ln) for ln in (run / "metrics.jsonl").read_text().splitlines()]
    evals = [r for r in rows if r["kind"] == "eval"]
    assert [r["step"] for r in evals] == [0, 2, 4, 6]
    assert set(evals[-1]["val"]) == {"1", "2"} and "restricted_loss" in evals[-1]["progress"]
    sweep = [r for r in rows if r["kind"] == "sweep"][-1]
    assert set(sweep["at_eval_depth"]) == {"extrap_depth", "systematic", "systematic_extrap"}
    assert len(sweep["by_T"]["extrap_depth"]["3"]) == 3


def test_demo_runs_live_apart(tmp_path, data_root):
    """a --max-steps demo writes under demos/ and leaves the real run name free."""
    cfg = config(tmp_path, data_root, run_entry())
    demo = train.train(cfg["runs"][0], cfg, "cpu", max_steps=2, log=quiet)
    assert demo == tmp_path / "runs" / "demos" / "r"
    assert json.loads((demo / "meta.json").read_text())["max_steps"] == 2
    real = train.train(cfg["runs"][0], cfg, "cpu", log=quiet)
    assert real == tmp_path / "runs" / "r"


def test_refuses_existing_run(tmp_path, data_root):
    """a second run with the same name fails without touching the first."""
    cfg = config(tmp_path, data_root, run_entry())
    train.train(cfg["runs"][0], cfg, "cpu", log=quiet)
    with pytest.raises(train.RunConfigError):
        train.train(cfg["runs"][0], cfg, "cpu", log=quiet)


def test_deterministic(tmp_path, data_root):
    """two runs of the same entry give identical metrics and weights."""
    outs = []
    for sub in ("a", "b"):
        cfg = config(tmp_path / sub, data_root, run_entry())
        outs.append(train.train(cfg["runs"][0], cfg, "cpu", log=quiet))
    assert (outs[0] / "metrics.jsonl").read_text() == (outs[1] / "metrics.jsonl").read_text()


class _Stop(Exception):
    """simulated interruption."""


def test_resume_matches_uninterrupted(tmp_path, data_root):
    """a run interrupted after step 4 and resumed ends with the same weights and metrics."""
    entry = run_entry(model={**run_entry()["model"], "norm": "rms_sandwich", "n_prelude": 1,
                             "injection": "concat", "state_init": "random", "n_coda": 1},
                      depth={"rule": "sampled", "mean": 2, "sigma": 0.5, "cap": 4, "grad_iters": 2})
    full_cfg = config(tmp_path / "full", data_root, entry)
    full = train.train(entry, full_cfg, "cpu", log=quiet)

    def stop_at_4(line):
        """interrupt once step 4 is logged."""
        if line.startswith("step      4"):
            raise _Stop

    cut_cfg = config(tmp_path / "cut", data_root, entry)
    with pytest.raises(_Stop):
        train.train(entry, cut_cfg, "cpu", log=stop_at_4)
    cut = train.train(entry, cut_cfg, "cpu", resume=True, log=quiet)

    a = t.load(full / "checkpoints" / "final.pt")
    b = t.load(cut / "checkpoints" / "final.pt")
    assert a.keys() == b.keys() and all(t.equal(a[k], b[k]) for k in a)
    assert (full / "metrics.jsonl").read_text() == (cut / "metrics.jsonl").read_text()


def test_chunked_gradient_matches_full_batch(data_root):
    """accumulating over small chunks gives the full-batch gradient."""
    data = train.load_data(data_root / "t_p11", "cpu")
    grads = []
    for chunk in (5, 10_000):
        t.manual_seed(0)
        m = models.LoopedTransformer(models.ModelConfig(p=11, **TINY))
        train.accumulate_grads(m, data["sets"]["train"], 2, None, chunk)
        grads.append({n: v.grad.clone() for n, v in m.named_parameters()})
    assert all(t.allclose(grads[0][n], grads[1][n], atol=1e-6) for n in grads[0])


@pytest.mark.parametrize("over, msg", [
    ({"colour": 1}, "unknown keys"),
    ({"depth": {"rule": "sampled", "mean": 2, "sigma": 0.5, "cap": 4},
      "model": {**run_entry()["model"], "core_shared": False}}, "unshared"),
    ({"model": {**run_entry()["model"], "n_ctx": 5}}, "set from the data"),
    ({"depth": {"rule": "sampled", "mean": 2, "sigma": 0.5, "cap": 4, "grad_iters": 9}}, "grad_iters"),
    ({"depth": {"rule": "fixed", "T": 2, "grad_iters": 3}}, "grad_iters"),
    ({"optim": {"no_decay": "w_rms"}}, "no_decay"),
])
def test_bad_run_configs(over, msg):
    """invalid run entries are rejected with a message naming the problem."""
    with pytest.raises(train.RunConfigError, match=msg):
        train.validate_entry(run_entry(**over))


def test_checkpoint_steps():
    """checkpoints are log-spaced, unique and end at the final step."""
    s = train.checkpoint_steps(40_000, 25)
    assert s[0] == 100 and s[-1] == 40_000 and len(s) == len(set(s)) <= 26


def test_restricted_loss_on_a_pure_fourier_model():
    """logits built only from key-frequency (a+b) terms have restricted loss equal to the full loss."""
    p = 11
    a, b = t.meshgrid(t.arange(p), t.arange(p), indexing="ij")
    s = (a + b).flatten()
    c = t.arange(p)
    w = 2 * t.pi * 3 / p
    logits = 20 * t.cos(w * (s[:, None] - c[None, :]))
    dirs = progress.xpy_directions(progress.fourier_basis(p))
    comp = progress.key_component(logits, dirs, [3])
    assert t.allclose(comp, logits, atol=1e-3)
    assert t.allclose(logits - comp, t.zeros_like(logits), atol=1e-3)


def test_progress_measures_on_anchor(data_root):
    """progress measures report every paper quantity, including the neuron-logit gini for coda-free models."""
    data = train.load_data(data_root / "t_p11", "cpu")
    t.manual_seed(0)
    m = models.LoopedTransformer(models.ModelConfig(p=11, **TINY))
    out = progress.progress_measures(m, *data["grid"], 1)
    assert {"grid_train_loss", "grid_test_loss", "restricted_loss", "excluded_loss",
            "gini_W_E", "gini_W_L", "sum_sq_weights", "sum_sq_gains"} <= set(out)
    assert out["sum_sq_gains"] == 0
    assert len(out["excluded_loss_per_freq"]) == len(out["key_freqs"]) == 5


def test_plot_progress_writes_figure(tmp_path, data_root):
    """the progress plot renders from a short run's metrics."""
    cfg = config(tmp_path, data_root, run_entry())
    run = train.train(cfg["runs"][0], cfg, "cpu", log=quiet)
    out = plot_progress.plot_run(run, phases=[2, 4])
    assert out.exists() and out.stat().st_size > 10_000


def test_plot_progress_splits_out_norm_gains(tmp_path, data_root):
    """a model with norm gains logs their squared norm, and the progress plot still renders."""
    model = {**run_entry()["model"], "norm": "rms_sandwich", "n_prelude": 1, "injection": "concat",
             "state_init": "random", "n_coda": 1}
    entry = run_entry(model=model)
    run = train.train(entry, config(tmp_path, data_root, entry), "cpu", log=quiet)
    prog = [json.loads(ln)["progress"] for ln in (run / "metrics.jsonl").read_text().splitlines()
            if json.loads(ln)["kind"] == "eval"]
    assert all(0 < r["sum_sq_gains"] < r["sum_sq_weights"] for r in prog)
    assert plot_progress.plot_run(run).stat().st_size > 10_000


def test_plot_depth_sweep_writes_figure(tmp_path, data_root):
    """the depth-sweep plot renders from a short run with a shared core."""
    cfg = config(tmp_path, data_root, run_entry())
    run = train.train(cfg["runs"][0], cfg, "cpu", log=quiet)
    out = plot_progress.plot_depth_sweep(run, train_T=2)
    assert out.exists() and out.stat().st_size > 10_000


def test_tensorboard_events_match_metrics(tmp_path, data_root):
    """a run writes event files live, and a backfill from metrics.jsonl reproduces the same tags."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    cfg = config(tmp_path, data_root, run_entry())
    run = train.train(cfg["runs"][0], cfg, "cpu", log=quiet)
    live = EventAccumulator(str(run / "tb")).Reload().Tags()["scalars"]
    assert {"val_acc/h1", "progress/restricted_loss", "systematic_acc/h2.k1"} <= set(live)
    assert tb_log.backfill(run) > 0
    assert set(EventAccumulator(str(run / "tb")).Reload().Tags()["scalars"]) == set(live)


def test_recompute_matches_logged_metrics(tmp_path, data_root):
    """measures recomputed from a run's checkpoints equal the ones logged during training."""
    entry = run_entry(model={**run_entry()["model"], "norm": "rms_sandwich", "n_prelude": 1,
                             "injection": "concat", "state_init": "random", "n_coda": 1})
    cfg = config(tmp_path, data_root, entry)
    run_dir = train.train(entry, cfg, "cpu", log=quiet)
    run = analysis.load_run(run_dir, data_root)
    assert run["data_changed"] == [] and run["T"] == 2
    logged = {r["step"]: r for r in plot_progress.load_evals(run_dir)}
    rows = analysis.recompute(run)
    assert [r["step"] for r in rows] == [s for s, _ in analysis.checkpoints(run)]
    for r in rows:
        assert r["val"] == logged[r["step"]]["val"] and r["progress"] == logged[r["step"]]["progress"]
    final = analysis.load_model(run, run_dir / "checkpoints" / "final.pt")
    latest = analysis.load_model(run, run_dir / "latest.pt")
    assert all(t.equal(a, b) for a, b in zip(final.state_dict().values(), latest.state_dict().values()))
    sweep = analysis.depth_sweep(run, analysis.checkpoints(run)[-1][1], 3)
    assert len(sweep["extrap_depth"]["3"]) == 3


def test_plot_compare_writes_figure(tmp_path, data_root):
    """the comparison plot overlays runs with different hop ranges."""
    cfg = config(tmp_path, data_root, run_entry("a"), run_entry("b", depth={"rule": "fixed", "T": 1}))
    runs = [(e["name"], plot_progress.load_evals(train.train(e, cfg, "cpu", log=quiet))) for e in cfg["runs"]]
    out = plot_progress.plot_compare(runs, tmp_path / "cmp" / "c.png")
    assert out.exists() and out.stat().st_size > 10_000


def test_pause_and_resume_matches_uninterrupted(tmp_path, data_root):
    """a sigterm mid-run saves at the current step and stops; resuming gives the uninterrupted result."""
    entry = run_entry(train={**run_entry()["train"], "resume_every": 100})
    full = train.train(entry, config(tmp_path / "full", data_root, entry), "cpu", log=quiet)

    def pause_at_2(line):
        """send sigterm to this process once step 2 is logged."""
        if line.startswith("step      2"):
            os.kill(os.getpid(), signal.SIGTERM)

    cut_cfg = config(tmp_path / "cut", data_root, entry)
    cut = train.train(entry, cut_cfg, "cpu", log=pause_at_2)
    assert not (cut / "checkpoints" / "final.pt").exists()
    assert t.load(cut / "latest.pt", weights_only=False)["step"] == 2
    train.train(entry, cut_cfg, "cpu", resume=True, log=quiet)
    a = t.load(full / "checkpoints" / "final.pt")
    b = t.load(cut / "checkpoints" / "final.pt")
    assert all(t.equal(a[k], b[k]) for k in a)
    assert (full / "metrics.jsonl").read_text() == (cut / "metrics.jsonl").read_text()
    assert (full / "steps.jsonl").read_text() == (cut / "steps.jsonl").read_text()
    assert signal.getsignal(signal.SIGTERM) == signal.SIG_DFL


def test_step_log_and_no_decay_group(tmp_path, data_root):
    """every step is logged with its depth and group norms, and no_decay parameters are left undecayed."""
    model = {**run_entry()["model"], "norm": "rms_sandwich", "n_prelude": 1, "injection": "concat",
             "state_init": "random", "n_coda": 1}
    depth = {"rule": "sampled", "mean": 2, "sigma": 0.5, "cap": 4, "grad_iters": 2}
    entry = run_entry(model=model, depth=depth, optim={"no_decay": ["w_rms"]})
    run = train.train(entry, config(tmp_path, data_root, entry), "cpu", log=quiet)
    rows = [json.loads(ln) for ln in (run / "steps.jsonl").read_text().splitlines()]
    assert [r["step"] for r in rows] == list(range(1, 7))
    assert all(1 <= r["depth"] <= 4 and r["grad_norm"] > 0 for r in rows)
    assert {"gains", "core", "prelude", "coda", "W_A"} <= set(rows[0]["groups"])
    # warmup starts at lr 0, so the first step moves nothing and later ones do
    assert rows[0]["groups"]["core"]["update"] == 0 and all(r["groups"]["core"]["update"] > 0 for r in rows[1:])
    cfg = models.ModelConfig(p=11, **{**{k: v for k, v in model.items()}, **TINY})
    groups = train.param_groups(models.LoopedTransformer(cfg), 1.0, ["w_rms"])
    assert groups[1]["weight_decay"] == 0 and groups[1]["params"] and all(g.dim() == 1 for g in groups[1]["params"])
    with pytest.raises(train.RunConfigError, match="match no parameter"):
        train.param_groups(models.LoopedTransformer(cfg), 1.0, ["nothing_like_this"])


def test_until_then_resume_matches_uninterrupted(tmp_path, data_root):
    """stopping at a step with until and resuming gives the uninterrupted result."""
    entry = run_entry(depth={"rule": "fixed", "T": 2, "grad_iters": 1},
                      train={**run_entry()["train"], "resume_every": 100})
    full = train.train(entry, config(tmp_path / "full", data_root, entry), "cpu", log=quiet)
    cut_cfg = config(tmp_path / "cut", data_root, entry)
    cut = train.train(entry, cut_cfg, "cpu", until=4, log=quiet)
    assert not (cut / "checkpoints" / "final.pt").exists()
    assert t.load(cut / "latest.pt", weights_only=False)["step"] == 4
    with pytest.raises(train.RunConfigError, match="until"):
        train.train(entry, cut_cfg, "cpu", resume=True, until=4, log=quiet)
    train.train(entry, cut_cfg, "cpu", resume=True, log=quiet)
    a = t.load(full / "checkpoints" / "final.pt")
    b = t.load(cut / "checkpoints" / "final.pt")
    assert all(t.equal(a[k], b[k]) for k in a)
    assert (full / "metrics.jsonl").read_text() == (cut / "metrics.jsonl").read_text()
    assert (full / "steps.jsonl").read_text() == (cut / "steps.jsonl").read_text()


def test_resume_refuses_a_live_claim_or_a_changed_config(tmp_path, data_root):
    """resuming is refused while another live process trains the run, or once its config entry has changed."""
    entry = run_entry()
    cfg = config(tmp_path, data_root, entry)
    run = train.train(entry, cfg, "cpu", until=2, log=quiet)
    assert not (run / "train.pid").exists()
    (run / "train.pid").write_text(json.dumps({"pid": os.getppid(), "host": socket.gethostname()}))
    with pytest.raises(train.RunConfigError, match="being trained"):
        train.train(entry, cfg, "cpu", resume=True, log=quiet)
    (run / "train.pid").write_text(json.dumps({"pid": os.getppid(), "host": "another-machine"}))
    changed = {**entry, "optim": {"weight_decay": 0.5}}
    with pytest.raises(train.RunConfigError, match="changed"):
        train.train(changed, cfg, "cpu", resume=True, log=quiet)
    assert train.train(entry, cfg, "cpu", resume=True, log=quiet) == run
    assert (run / "checkpoints" / "final.pt").exists()


def test_until_keeps_its_weights_as_a_checkpoint(tmp_path, data_root):
    """stopping early saves a checkpoint at that step, which a later resume doesn't overwrite."""
    entry = run_entry(train={**run_entry()["train"], "n_checkpoints": 1})
    cfg = config(tmp_path, data_root, entry)
    run = train.train(entry, cfg, "cpu", until=3, log=quiet)
    kept = t.load(run / "checkpoints" / "step_0000003.pt")
    train.train(entry, cfg, "cpu", resume=True, log=quiet)
    assert all(t.equal(kept[k], v) for k, v in t.load(run / "checkpoints" / "step_0000003.pt").items())
    assert t.load(run / "latest.pt", weights_only=False)["step"] == 6


def test_interrupted_before_the_first_save_resumes_from_step_0(tmp_path, data_root):
    """a fresh run is resumable from step 0, and resuming there matches the uninterrupted run."""
    entry = run_entry(train={**run_entry()["train"], "resume_every": 100})
    full = train.train(entry, config(tmp_path / "full", data_root, entry), "cpu", log=quiet)

    def stop_at_2(line):
        """interrupt once step 2 is logged, before any periodic save."""
        if line.startswith("step      2"):
            raise _Stop

    cut_cfg = config(tmp_path / "cut", data_root, entry)
    with pytest.raises(_Stop):
        train.train(entry, cut_cfg, "cpu", log=stop_at_2)
    cut = tmp_path / "cut" / "runs" / "r"
    assert t.load(cut / "latest.pt", weights_only=False)["step"] == 0
    train.train(entry, cut_cfg, "cpu", resume=True, log=quiet)
    a, b = t.load(full / "checkpoints" / "final.pt"), t.load(cut / "checkpoints" / "final.pt")
    assert all(t.equal(a[k], b[k]) for k in a)
    assert (full / "metrics.jsonl").read_text() == (cut / "metrics.jsonl").read_text()


def test_resumed_tensorboard_events_match_metrics(tmp_path, data_root):
    """after a resume, the event files hold each eval step exactly once, including the resume step."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    entry = run_entry()
    cfg = config(tmp_path, data_root, entry)
    run = train.train(entry, cfg, "cpu", until=4, log=quiet)
    train.train(entry, cfg, "cpu", resume=True, log=quiet)
    logged = [json.loads(ln)["step"] for ln in (run / "metrics.jsonl").read_text().splitlines()
              if json.loads(ln)["kind"] == "eval"]
    events = [e.step for e in EventAccumulator(str(run / "tb")).Reload().Scalars("val_acc/h1")]
    assert sorted(events) == logged == [0, 2, 4, 6]


def test_divergence_stops_and_is_recorded(tmp_path, data_root, monkeypatch):
    """a non-finite loss stops the run without updating, records the step, and blocks resuming."""
    real, calls = train.accumulate_grads, []

    def nan_on_third(*args, **kwargs):
        """the real gradient, but a nan loss on the third step."""
        calls.append(1)
        total = real(*args, **kwargs)
        return float("nan") if len(calls) == 3 else total

    monkeypatch.setattr(train, "accumulate_grads", nan_on_third)
    entry = run_entry()
    cfg = config(tmp_path, data_root, entry)
    run = train.train(entry, cfg, "cpu", log=quiet)
    assert json.loads((run / "meta.json").read_text())["diverged_at"] == 3
    assert not (run / "checkpoints" / "final.pt").exists()
    assert [json.loads(ln)["step"] for ln in (run / "steps.jsonl").read_text().splitlines()] == [1, 2]
    with pytest.raises(train.RunConfigError, match="diverged"):
        train.train(entry, cfg, "cpu", resume=True, log=quiet)


def test_pause_during_setup_still_saves(tmp_path, data_root, monkeypatch):
    """a sigterm while the data is loading pauses the run after its first step, resumable."""
    real = train.load_data

    def load_then_signal(*args, **kwargs):
        """load the data, then deliver a sigterm as if sent during setup."""
        os.kill(os.getpid(), signal.SIGTERM)
        return real(*args, **kwargs)

    monkeypatch.setattr(train, "load_data", load_then_signal)
    entry = run_entry(train={**run_entry()["train"], "resume_every": 100})
    run = train.train(entry, config(tmp_path, data_root, entry), "cpu", log=quiet)
    assert t.load(run / "latest.pt", weights_only=False)["step"] == 1
    assert signal.getsignal(signal.SIGTERM) == signal.SIG_DFL


def test_metrics_truncation_drops_a_cut_off_last_line(tmp_path):
    """resuming after a write was cut off drops the partial line instead of failing."""
    path = tmp_path / "metrics.jsonl"
    path.write_text('{"step": 1}\n{"step": 2}\n{"step": 3, "kind"')
    train._truncate_metrics(path, 5)
    assert path.read_text() == '{"step": 1}\n{"step": 2}\n'
