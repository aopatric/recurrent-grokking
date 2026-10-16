"""train a model from configs/train.yaml and write its run directory.

    uv run python src/train.py --name anchor_p113_h1 [--max-steps N] [--until N] [--resume]

--max-steps makes a demo run, written under <output_root>/demos/<name>.
--until stops a real run early at step N, ready for --resume to carry on toward train.steps.
ctrl-c (or kill) pauses a run: it saves latest.pt at the current step and exits; --resume continues it.

paths in the config are relative to the working directory.
"""

import argparse
import dataclasses
import hashlib
import json
import os
import platform
import signal
import sys
from pathlib import Path

import numpy as np
import torch as t
import yaml

import depth as dp
import models
import progress
import tb_log
from make_data import git_info

_RUN_KEYS = {"name", "data", "seed", "model", "depth", "train"}
_OPTIONAL_RUN_KEYS = {"optim"}
_TRAIN_REQUIRED = {"steps", "eval_every", "sweep_every", "eval_T_max"}
_TRAIN_DEFAULTS = {"n_checkpoints": 25, "chunk_size": 8192, "resume_every": 1000, "log_steps": True}
# nanda's settings, from his progress-measures training code
_OPTIM_DEFAULTS = {"lr": 1e-3, "weight_decay": 1.0, "betas": [0.9, 0.98], "warmup_steps": 10, "no_decay": []}
_MODEL_DERIVED = {"p", "n_ctx", "n_core_copies"}
_EVAL_FILES = ("extrap_depth", "systematic", "systematic_extrap")


class RunConfigError(ValueError):
    """invalid run config, or a run directory in the wrong state."""


# config

def validate_config(cfg):
    """check the top-level shape of a training config and every run entry.

    args:
        cfg: parsed yaml.
    raises:
        RunConfigError: on the first invalid entry.
    """
    if not isinstance(cfg, dict) or not {"output_root", "data_root", "runs"} <= set(cfg):
        raise RunConfigError("config must have output_root, data_root and runs")
    names = [r.get("name") for r in cfg["runs"]]
    if dupes := sorted({n for n in names if names.count(n) > 1}):
        raise RunConfigError(f"duplicate run names: {dupes}")
    for entry in cfg["runs"]:
        validate_entry(entry)


def validate_entry(entry):
    """check one run entry's keys and settings, without touching the data.

    args:
        entry: run entry from the config.
    raises:
        RunConfigError: listing every violated rule.
    """
    name = entry.get("name", "<unnamed>")
    errs = []
    keys = set(entry)
    if missing := _RUN_KEYS - keys:
        errs.append(f"missing keys: {sorted(missing)}")
    if unknown := keys - _RUN_KEYS - _OPTIONAL_RUN_KEYS:
        errs.append(f"unknown keys: {sorted(unknown)}")
    if errs:
        raise RunConfigError(f"[{name}] " + "; ".join(errs))

    tr = entry["train"]
    if missing := _TRAIN_REQUIRED - set(tr):
        errs.append(f"train: missing keys {sorted(missing)}")
    if unknown := set(tr) - _TRAIN_REQUIRED - set(_TRAIN_DEFAULTS):
        errs.append(f"train: unknown keys {sorted(unknown)}")
    if unknown := set(entry.get("optim", {})) - set(_OPTIM_DEFAULTS):
        errs.append(f"optim: unknown keys {sorted(unknown)}")
    nd = entry.get("optim", {}).get("no_decay", [])
    if not isinstance(nd, list) or not all(isinstance(x, str) and x for x in nd):
        errs.append("optim.no_decay must be a list of parameter-name substrings")
    if derived := set(entry["model"]) & _MODEL_DERIVED:
        errs.append(f"model: {sorted(derived)} are set from the data and depth, not the config")
    try:
        rule = dp.make_depth(entry["depth"])
    except (ValueError, KeyError) as e:
        errs.append(f"depth: {e}")
        rule = None
    if not entry["model"].get("core_shared", True) and not isinstance(rule, dp.FixedDepth):
        errs.append("an unshared core needs depth.rule = fixed")
    gi = entry["depth"].get("grad_iters")
    if gi is not None and isinstance(rule, dp.SampledDepth) and not 1 <= gi <= rule.cap:
        errs.append("depth.grad_iters must be in [1, cap]")
    if gi is not None and isinstance(rule, dp.FixedDepth) and not 1 <= gi <= rule.T:
        errs.append("depth.grad_iters must be in [1, T]")
    if errs:
        raise RunConfigError(f"[{name}] " + "; ".join(errs))


def resolve(entry, data_meta):
    """turn a validated entry into concrete settings using the dataset's metadata.

    args:
        entry: run entry.
        data_meta: the dataset's meta.json.
    returns:
        (ModelConfig, depth rule, grad_iters, train settings, optim settings).
    raises:
        RunConfigError: if the model settings are invalid.
    """
    rule = dp.make_depth(entry["depth"])
    m = dict(entry["model"])
    m["p"] = data_meta["config"]["p"]
    if m.get("pos") == "learned":
        # the longest prompt is max_test_hops + 1 operands plus "="
        m["n_ctx"] = data_meta["config"]["max_test_hops"] + 2
    if not m.get("core_shared", True):
        # validation guarantees a fixed rule here, so its eval depth is T
        m["n_core_copies"] = rule.eval_depth()
    try:
        mcfg = models.ModelConfig.from_dict(m)
    except (ValueError, TypeError) as e:
        raise RunConfigError(f"[{entry['name']}] model: {e}") from e
    train = {**_TRAIN_DEFAULTS, **entry["train"]}
    optim = {**_OPTIM_DEFAULTS, **entry.get("optim", {})}
    return mcfg, rule, entry["depth"].get("grad_iters"), train, optim


# data

def _read_jsonl(path):
    """read a json lines file.

    args:
        path: file to read.
    returns:
        list of row dicts.
    """
    with open(path) as f:
        return [json.loads(line) for line in f]


def _group(rows, vocab, device, by_step=False):
    """stack rows into token and answer tensors, one group per hop (and novel step).

    args:
        rows: dataset rows.
        vocab: token to id mapping.
        device: torch device.
        by_step: also split by novel_step, for systematic files.
    returns:
        dict from label ("3", or "3.k2" with by_step) to (tokens, answers).
    """
    groups = {}
    for r in rows:
        label = f"{r['hop']}.k{r['novel_step']}" if by_step else str(r["hop"])
        groups.setdefault(label, []).append(r)
    return {
        label: (t.tensor([[vocab[tok] for tok in r["tokens"]] for r in rs], device=device),
                t.tensor([r["answer"] for r in rs], device=device))
        for label, rs in sorted(groups.items(), key=lambda kv: kv[0])
    }


def load_data(data_dir, device):
    """load a generated dataset into per-hop tensors.

    args:
        data_dir: dataset directory from make_data.py.
        device: torch device.
    returns:
        dict with meta, sets (name -> groups), grid (tokens, labels, is_train) and sha256 per file.
    """
    data_dir = Path(data_dir)
    meta = json.loads((data_dir / "meta.json").read_text())
    vocab = json.loads((data_dir / "vocab.json").read_text())
    sets = {"train": _group(_read_jsonl(data_dir / "train.jsonl"), vocab, device),
            "val": _group(_read_jsonl(data_dir / "val.jsonl"), vocab, device)}
    for name in _EVAL_FILES:
        if (data_dir / f"{name}.jsonl").exists():
            sets[name] = _group(_read_jsonl(data_dir / f"{name}.jsonl"), vocab, device,
                                by_step=name.startswith("systematic"))
    grid = _read_jsonl(data_dir / "grid_hop1.jsonl")
    grid_tokens = t.tensor([[vocab[tok] for tok in r["tokens"]] for r in grid], device=device)
    grid_labels = t.tensor([r["answer"] for r in grid], device=device)
    is_train = t.tensor([r["split"] == "train" for r in grid], device=device)
    sha = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(data_dir.iterdir()) if f.is_file()}
    return {"meta": meta, "sets": sets, "grid": (grid_tokens, grid_labels, is_train), "sha256": sha}


# evaluation

def _chunks(n, size):
    """split range(n) into consecutive slices.

    args:
        n: total length.
        size: largest slice length.
    returns:
        list of slice objects.
    """
    return [slice(i, min(i + size, n)) for i in range(0, n, size)]


@t.no_grad()
def evaluate(model, groups, n_iter, chunk_size, generator=None):
    """loss and accuracy per group at one depth.

    args:
        model: LoopedTransformer.
        groups: dict from label to (tokens, answers).
        n_iter: depth.
        chunk_size: rows per forward pass.
        generator: torch generator for a random start, or None.
    returns:
        dict from label to {"loss", "acc"}.
    """
    out = {}
    for label, (x, y) in groups.items():
        loss, correct = 0.0, 0
        for sl in _chunks(len(y), chunk_size):
            logits = model(x[sl], n_iter, generator=generator)
            loss += float(progress.cross_entropy_high_precision(logits, y[sl])) * (sl.stop - sl.start)
            correct += int((logits.argmax(-1) == y[sl]).sum())
        out[label] = {"loss": loss / len(y), "acc": correct / len(y)}
    return out


@t.no_grad()
def accuracy_by_depth(model, groups, T_max, chunk_size, generator=None):
    """accuracy per group after every pass 1..T_max, from one run per chunk.

    args:
        model: LoopedTransformer with a shared core.
        groups: dict from label to (tokens, answers).
        T_max: deepest pass.
        chunk_size: rows per forward pass.
        generator: torch generator for a random start, or None.
    returns:
        dict from label to a list of T_max accuracies.
    """
    out = {}
    for label, (x, y) in groups.items():
        correct = t.zeros(T_max, device=y.device)
        for sl in _chunks(len(y), chunk_size):
            logits = model(x[sl], T_max, return_all=True, generator=generator)
            correct += (logits.argmax(-1) == y[sl, None]).sum(0)
        out[label] = [round(float(c) / len(y), 6) for c in correct]
    return out


def accumulate_grads(model, train_sets, n_iter, grad_iters, chunk_size, generator=None):
    """backprop the full-batch loss: the mean over hops of each hop's mean cross-entropy.

    args:
        model: LoopedTransformer.
        train_sets: dict from hop label to (tokens, answers).
        n_iter: depth for this step.
        grad_iters: truncated-backprop window, or None.
        chunk_size: rows per forward pass.
        generator: torch generator for a random start, or None.
    returns:
        the loss as a float; gradients are accumulated into the model.
    """
    hop_weight = 1 / len(train_sets)
    total = 0.0
    for x, y in train_sets.values():
        for sl in _chunks(len(y), chunk_size):
            logits = model(x[sl], n_iter, grad_iters=grad_iters, generator=generator)
            # each chunk's share of its hop's mean, so chunking gives exactly the full-batch gradient
            share = (sl.stop - sl.start) / len(y) * hop_weight
            loss = progress.cross_entropy_high_precision(logits, y[sl]) * share
            loss.backward()
            total += float(loss.detach())
    return total


# checkpoints and run directory

class _PauseRequest:
    """turns the first ctrl-c or sigterm into a request to save and stop after the current step."""

    def __init__(self):
        """install the handlers; outside the main thread signals can't be caught, so pausing is off."""
        self.requested = False
        self._old = {}
        try:
            for sig in (signal.SIGINT, signal.SIGTERM):
                self._old[sig] = signal.signal(sig, self._handle)
        except ValueError:
            self._old = {}

    def _handle(self, signum, frame):
        """record the request; a second signal stops immediately.

        args:
            signum: signal number.
            frame: current stack frame (unused).
        """
        if self.requested:
            raise KeyboardInterrupt
        self.requested = True

    def restore(self):
        """put back the previous handlers."""
        for sig, handler in self._old.items():
            signal.signal(sig, handler)


def checkpoint_steps(steps, n):
    """log-spaced checkpoint steps from 100 to steps, always including steps.

    args:
        steps: final step.
        n: target number of checkpoints.
    returns:
        sorted list of unique step numbers.
    """
    lo = min(100, steps)
    pts = np.unique(np.round(np.logspace(np.log10(lo), np.log10(steps), n)).astype(int))
    return sorted(set(int(s) for s in pts) | {steps})


def _atomic_save(obj, path):
    """torch.save through a temp file so an interruption never leaves a corrupt file.

    args:
        obj: object to save.
        path: destination.
    """
    tmp = path.with_name(path.name + ".tmp")
    t.save(obj, tmp)
    os.replace(tmp, path)


def _truncate_metrics(path, step):
    """drop metrics lines logged after a step, before resuming from it.

    args:
        path: metrics.jsonl.
        step: last step kept.
    """
    if not path.exists():
        return
    keep = [ln for ln in path.read_text().splitlines() if json.loads(ln)["step"] <= step]
    path.write_text("".join(ln + "\n" for ln in keep))


def param_groups(model, weight_decay, no_decay):
    """split parameters into a decayed group and an undecayed one.

    args:
        model: the model.
        weight_decay: decoupled weight decay for the decayed group.
        no_decay: parameter-name substrings that are not decayed.
    returns:
        list of optimizer parameter groups; a single group when no_decay is empty.
    raises:
        RunConfigError: if a no_decay substring matches no parameter.
    """
    named = list(model.named_parameters())
    if not no_decay:
        return [{"params": [prm for _, prm in named], "weight_decay": weight_decay}]
    if unused := [x for x in no_decay if not any(x in n for n, _ in named)]:
        raise RunConfigError(f"optim.no_decay entries match no parameter: {unused}")
    skip = [any(x in n for x in no_decay) for n, _ in named]
    return [{"params": [prm for (_, prm), sk in zip(named, skip) if not sk], "weight_decay": weight_decay},
            {"params": [prm for (_, prm), sk in zip(named, skip) if sk], "weight_decay": 0.0}]


def _module_group(name):
    """the per-step logging group of a parameter: norm gains together, everything else by top-level module.

    args:
        name: parameter name.
    returns:
        group name.
    """
    return "gains" if "w_rms" in name else name.split(".")[0]


def step_stats(model, before):
    """per-step weight, gradient and update norms for each parameter group.

    args:
        model: the model, after the optimizer step, with gradients still set.
        before: dict from parameter name to its value before the step.
    returns:
        (global gradient norm, dict from group to {"w", "grad", "update"}), where update is ||dw|| / ||w_before||.
    """
    sums = {}
    for n, prm in model.named_parameters():
        g = sums.setdefault(_module_group(n), [0.0, 0.0, 0.0, 0.0])
        g[0] += float(prm.detach().pow(2).sum())
        g[1] += float(prm.grad.pow(2).sum()) if prm.grad is not None else 0.0
        g[2] += float((prm.detach() - before[n]).pow(2).sum())
        g[3] += float(before[n].pow(2).sum())
    grad_norm = sum(v[1] for v in sums.values()) ** 0.5
    return grad_norm, {k: {"w": v[0] ** 0.5, "grad": v[1] ** 0.5, "update": (v[2] / max(v[3], 1e-30)) ** 0.5}
                       for k, v in sorted(sums.items())}


def _device_name(device):
    """name of the device a run trains on.

    args:
        device: torch device string.
    returns:
        gpu model name, or "cpu".
    """
    return t.cuda.get_device_name(t.device(device)) if device.startswith("cuda") else "cpu"


def setup_precision():
    """force true fp32 matmuls (no tf32)."""
    t.backends.cuda.matmul.allow_tf32 = False
    t.backends.cudnn.allow_tf32 = False
    t.set_float32_matmul_precision("highest")


# training

def train(entry, cfg, device, max_steps=None, resume=False, until=None, log=print):
    """train one run and write its directory.

    args:
        entry: validated run entry.
        cfg: whole config (for output_root and data_root).
        device: torch device string.
        max_steps: stop after this many steps instead of train.steps; the run goes under output_root/demos.
        resume: continue from latest.pt instead of starting fresh.
        until: stop after this step, saving latest.pt for a later resume; None trains to the end.
        log: function for progress lines.
    returns:
        path to the run directory; the run is complete only if checkpoints/final.pt exists.
    raises:
        RunConfigError: if the run directory is in the wrong state or the settings are invalid.
    """
    # demos live apart so a short run never blocks the real one
    root = Path(cfg["output_root"]) / ("demos" if max_steps else "")
    run_dir = root / entry["name"]
    latest = run_dir / "latest.pt"
    if resume and not latest.exists():
        raise RunConfigError(f"{run_dir} has no latest.pt to resume from")
    if not resume and run_dir.exists():
        raise RunConfigError(f"{run_dir} exists; use --resume or add a new run entry")
    if (run_dir / "checkpoints" / "final.pt").exists():
        raise RunConfigError(f"{run_dir} is already complete")

    setup_precision()
    data = load_data(Path(cfg["data_root"]) / entry["data"], device)
    mcfg, rule, grad_iters, tr, op = resolve(entry, data["meta"])
    steps = max_steps or tr["steps"]
    seed = entry["seed"]

    # init on cpu so the starting weights don't depend on the device
    t.manual_seed(seed)
    model = models.LoopedTransformer(mcfg).to(device)
    opt = t.optim.AdamW(param_groups(model, op["weight_decay"], op["no_decay"]), lr=op["lr"],
                        betas=tuple(op["betas"]))
    warm = op["warmup_steps"]
    sched = t.optim.lr_scheduler.LambdaLR(opt, lambda s: min(s / warm, 1) if warm else 1.0)
    rng = dp.depth_rng(seed)
    state_gen = t.Generator(device=device).manual_seed(seed)
    start = 0

    metrics_path = run_dir / "metrics.jsonl"
    if resume:
        ck = t.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optim"])
        sched.load_state_dict(ck["sched"])
        rule.load_state_dict(ck["depth"])
        rng.setstate(ck["depth_rng"])
        state_gen.set_state(ck["state_gen"].cpu())
        start = ck["step"]
        if until is not None and until <= start:
            raise RunConfigError(f"{run_dir} is already at step {start}; --until must be later")
        _truncate_metrics(metrics_path, start)
        _truncate_metrics(run_dir / "steps.jsonl", start)
        meta = json.loads((run_dir / "meta.json").read_text())
        meta.setdefault("resumes", []).append({"from_step": start, "max_steps": max_steps, "until": until,
                                               "device": _device_name(device), "git": git_info()})
        (run_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        log(f"resumed {entry['name']} at step {start}")
    else:
        (run_dir / "checkpoints").mkdir(parents=True)
        (run_dir / "config.yaml").write_text(yaml.safe_dump(entry, sort_keys=False))
        meta = {
            "git": git_info(),
            "python": platform.python_version(),
            "torch": t.__version__,
            "device": _device_name(device),
            "data": {"name": entry["data"], "generator_version": data["meta"]["generator_version"],
                     "sha256": data["sha256"]},
            "model": dataclasses.asdict(mcfg),
            "steps": steps,
            "max_steps": max_steps,
        }
        (run_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    train_sets = data["sets"]["train"]
    grid_tokens, grid_labels, grid_train = data["grid"]
    ckpts = set(checkpoint_steps(steps, tr["n_checkpoints"]))
    eval_T = rule.eval_depth()
    can_sweep = mcfg.core_shared and tr["eval_T_max"] > 1

    stop = steps if until is None else min(until, steps)

    def eval_gen():
        """fresh generator so every evaluation sees the same random starts."""
        return t.Generator(device=device).manual_seed(seed + 1)

    def log_eval(step, train_loss):
        """write the routine evaluation line."""
        model.eval()
        row = {"kind": "eval", "step": step, "lr": sched.get_last_lr()[0], "depth": eval_T,
               "train_step_loss": train_loss,
               "train": evaluate(model, train_sets, eval_T, tr["chunk_size"], eval_gen()),
               "val": evaluate(model, data["sets"]["val"], eval_T, tr["chunk_size"], eval_gen()),
               "progress": progress.progress_measures(model, grid_tokens, grid_labels, grid_train, eval_T,
                                                       generator=eval_gen())}
        model.train()
        with open(metrics_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        tb_log.write_row(writer, row)
        accs = " ".join(f"h{h}={v['acc']:.3f}" for h, v in row["val"].items())
        shown = "-" if train_loss is None else f"{train_loss:.3e}"
        log(f"step {step:>6}  train loss {shown}"
            f"  val acc {accs}  restricted {row['progress']['restricted_loss']:.3f}")

    def log_sweep(step):
        """write the held-out and depth-sweep line."""
        model.eval()
        row = {"kind": "sweep", "step": step, "depth": eval_T, "at_eval_depth": {}, "by_T": {}}
        for name in ("val",) + _EVAL_FILES:
            if name not in data["sets"]:
                continue
            if name != "val":
                row["at_eval_depth"][name] = evaluate(model, data["sets"][name], eval_T, tr["chunk_size"], eval_gen())
            if can_sweep:
                row["by_T"][name] = accuracy_by_depth(model, data["sets"][name], tr["eval_T_max"],
                                                      tr["chunk_size"], eval_gen())
        model.train()
        with open(metrics_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        tb_log.write_row(writer, row)

    def save_latest(step):
        """write the resume checkpoint."""
        _atomic_save({"model": model.state_dict(), "optim": opt.state_dict(), "sched": sched.state_dict(),
                      "depth": rule.state_dict(), "depth_rng": rng.getstate(),
                      "state_gen": state_gen.get_state(), "step": step}, latest)

    writer = tb_log.open_writer(run_dir, start)
    pause = _PauseRequest()
    try:
        if start == 0:
            log_eval(0, None)
        model.train()
        for step in range(start, steps):
            r = rule.sample(rng)
            opt.zero_grad(set_to_none=True)
            total = accumulate_grads(model, train_sets, r, grad_iters, tr["chunk_size"], state_gen)
            if tr["log_steps"]:
                before = {n: prm.detach().clone() for n, prm in model.named_parameters()}
            opt.step()
            sched.step()
            rule.update(r, total)

            done = step + 1
            if tr["log_steps"]:
                grad_norm, groups = step_stats(model, before)
                row = {"step": done, "depth": r, "loss": total, "grad_norm": grad_norm, "groups": groups}
                with open(run_dir / "steps.jsonl", "a") as f:
                    f.write(json.dumps(row) + "\n")
                tb_log.write_step(writer, row)
            if done % tr["eval_every"] == 0 or done == steps:
                log_eval(done, total)
            if done % tr["sweep_every"] == 0 or done in ckpts:
                log_sweep(done)
            if done in ckpts:
                name = "final.pt" if done == steps else f"step_{done:07d}.pt"
                _atomic_save(model.state_dict(), run_dir / "checkpoints" / name)
            if (done == steps or done == stop or done % tr["resume_every"] == 0
                    or (pause.requested and done < steps)):
                save_latest(done)
            if pause.requested and done < steps:
                log(f"paused {entry['name']} at step {done}; continue with --resume")
                break
            if done == stop < steps:
                log(f"stopped {entry['name']} at step {done} (--until); continue with --resume")
                break
    finally:
        pause.restore()
        writer.close()
    return run_dir


def main(argv=None):
    """run the cli.

    args:
        argv: argument list, or None for sys.argv.
    returns:
        exit code: 0 ok, 2 bad config or run directory state.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/train.yaml")
    ap.add_argument("--name", required=True)
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--until", type=int, help="stop early at this step, ready to resume")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", default="cuda" if t.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)
    try:
        cfg = yaml.safe_load(Path(args.config).read_text())
        validate_config(cfg)
        entries = [r for r in cfg["runs"] if r["name"] == args.name]
        if not entries:
            raise RunConfigError(f"no run named {args.name!r} in {args.config}")
        run_dir = train(entries[0], cfg, args.device, args.max_steps, args.resume, args.until,
                        log=lambda s: print(s, flush=True))
    except (OSError, yaml.YAMLError, RunConfigError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if (run_dir / "checkpoints" / "final.pt").exists():
        print(f"wrote {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
