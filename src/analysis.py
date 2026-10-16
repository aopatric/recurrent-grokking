"""reload a finished run from its directory and recompute its measures from the saved checkpoints."""

import json
from pathlib import Path

import torch as t
import yaml

import depth as dp
import models
import progress
import train


def load_run(run_dir, data_root="data/synthetic", device="cpu"):
    """load everything needed to analyse a run, using only what the run directory recorded.

    args:
        run_dir: run directory.
        data_root: directory holding the generated datasets.
        device: torch device string.
    returns:
        dict with run_dir, name, entry, meta, cfg (ModelConfig), T (eval depth), seed, chunk_size, data and
        data_changed (files whose sha256 differs from what the run trained on).
    """
    run_dir = Path(run_dir)
    entry = yaml.safe_load((run_dir / "config.yaml").read_text())
    meta = json.loads((run_dir / "meta.json").read_text())
    data = train.load_data(Path(data_root) / meta["data"]["name"], device)
    tr = train.resolve(entry, data["meta"])[3]
    changed = sorted(f for f, h in meta["data"]["sha256"].items() if data["sha256"].get(f) != h)
    return {
        "run_dir": run_dir,
        "name": entry["name"],
        "entry": entry,
        "meta": meta,
        "cfg": models.ModelConfig.from_dict(meta["model"]),
        "T": dp.make_depth(entry["depth"]).eval_depth(),
        "seed": entry["seed"],
        # random starts are drawn per chunk, so matching the logged numbers needs the run's chunk size
        "chunk_size": tr["chunk_size"],
        "data": data,
        "data_changed": changed,
    }


def checkpoints(run):
    """the run's saved weights in step order.

    args:
        run: dict from load_run.
    returns:
        list of (step, path); final.pt is placed at the run's last step.
    """
    ck = run["run_dir"] / "checkpoints"
    out = [(int(p.stem.split("_")[1]), p) for p in ck.glob("step_*.pt")]
    if (ck / "final.pt").exists():
        out.append((run["meta"]["steps"], ck / "final.pt"))
    return sorted(out)


def load_model(run, path, device="cpu"):
    """build the run's model and load one checkpoint into it.

    args:
        run: dict from load_run.
        path: a checkpoint file, or the run's latest.pt resume file.
        device: torch device string.
    returns:
        LoopedTransformer in eval mode.
    """
    model = models.LoopedTransformer(run["cfg"]).to(device)
    state = t.load(path, map_location=device, weights_only=False)
    # resume files hold the weights next to optimizer and sampler state
    model.load_state_dict(state["model"] if "optim" in state else state)
    return model.eval()


def eval_generator(run, device="cpu"):
    """the generator training used for evaluation, so random starts match the logged numbers.

    args:
        run: dict from load_run.
        device: torch device string.
    returns:
        torch generator.
    """
    return t.Generator(device=device).manual_seed(run["seed"] + 1)


def recompute(run, device="cpu", T=None, log=None):
    """recompute progress measures and per-hop accuracy at every checkpoint.

    args:
        run: dict from load_run.
        device: torch device string.
        T: depth to run at, or None for the run's eval depth.
        log: optional function for one line per checkpoint.
    returns:
        list of rows shaped like metrics.jsonl "eval" rows (step, depth, train, val, progress).
    """
    T = T or run["T"]
    data, chunk_size = run["data"], run["chunk_size"]
    rows = []
    for step, path in checkpoints(run):
        model = load_model(run, path, device)
        row = {
            "kind": "eval", "step": step, "depth": T,
            "train": train.evaluate(model, data["sets"]["train"], T, chunk_size, eval_generator(run, device)),
            "val": train.evaluate(model, data["sets"]["val"], T, chunk_size, eval_generator(run, device)),
            "progress": progress.progress_measures(model, *data["grid"], T, generator=eval_generator(run, device)),
        }
        rows.append(row)
        if log:
            accs = " ".join(f"h{h}={v['acc']:.3f}" for h, v in row["val"].items())
            log(f"step {step:>6}  val acc {accs}  restricted {row['progress']['restricted_loss']:.3f}")
    return rows


def depth_sweep(run, path, T_max, device="cpu"):
    """accuracy after every pass 1..T_max on every evaluation set, for one checkpoint.

    args:
        run: dict from load_run; the model must have a shared core.
        path: checkpoint file.
        T_max: deepest pass.
        device: torch device string.
    returns:
        dict from set name to {label: list of T_max accuracies}, as in a metrics.jsonl sweep's by_T.
    raises:
        ValueError: if the core is not shared.
    """
    if not run["cfg"].core_shared:
        raise ValueError("an unshared core cannot run past its depth")
    model = load_model(run, path, device)
    sets = {k: v for k, v in run["data"]["sets"].items() if k != "train"}
    return {name: train.accuracy_by_depth(model, groups, T_max, run["chunk_size"], eval_generator(run, device))
            for name, groups in sets.items()}
