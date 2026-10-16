"""run a list of training runs, one per device at a time, resuming any that were stopped.

    uv run python src/suite.py --runs NAME [NAME ...] [--devices 0,1] [--until N] [--dry-run]
    uv run python src/suite.py --suite NAME [NAME ...] [--suites configs/suites.yaml] ...
    uv run python src/suite.py --running      # list the live runner and training jobs
    uv run python src/suite.py --stop         # pause the live runner's jobs (or orphaned jobs directly)

each run is a src/train.py process with its output in <output_root>/<name>.log. finished runs are skipped,
runs with a latest.pt are resumed, and new runs start fresh. ctrl-c (or sigterm) pauses every running job
at its next step and stops launching new ones; rerunning the same command carries on. only one runner
works on an output_root at a time.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import torch as t
import yaml

import train as tr

_TRAIN_SCRIPT = Path(__file__).resolve().parent / "train.py"
# a job this young may not have installed its pause handler yet, so a stop waits before signalling it
_GRACE_SECONDS = 20


def load_suite(path, name):
    """read one named run list from a suites file.

    args:
        path: yaml file mapping suite names to lists of run names.
        name: suite to read.
    returns:
        list of run names.
    raises:
        RunConfigError: if the suite is missing or not a list of names.
    """
    suites = yaml.safe_load(Path(path).read_text()) or {}
    runs = suites.get(name)
    if not isinstance(runs, list) or not all(isinstance(r, str) for r in runs):
        raise tr.RunConfigError(f"no suite {name!r} with a list of run names in {path}")
    return runs


def plan(names, cfg, until=None):
    """decide what to do with each run.

    args:
        names: run names, in launch order.
        cfg: validated training config.
        until: step to stop each run at, or None to train to the end.
    returns:
        list of (name, action), where action is "start", "resume", or a reason to skip.
    raises:
        RunConfigError: if a name is unknown or repeated.
    """
    known = {r["name"] for r in cfg["runs"]}
    if unknown := [n for n in names if n not in known]:
        raise tr.RunConfigError(f"no runs named {unknown} in the config")
    if dupes := sorted({n for n in names if names.count(n) > 1}):
        raise tr.RunConfigError(f"runs listed twice: {dupes}")
    out = []
    for name in names:
        run_dir = Path(cfg["output_root"]) / name
        latest = run_dir / "latest.pt"
        if (run_dir / "checkpoints" / "final.pt").exists():
            out.append((name, "skip: complete"))
        elif pid := tr.training_pid(run_dir):
            out.append((name, f"skip: being trained by process {pid}"))
        elif (div := _diverged_at(run_dir)) is not None:
            out.append((name, f"skip: diverged at step {div}"))
        elif latest.exists():
            step = t.load(latest, map_location="cpu", weights_only=False)["step"]
            if until is not None and step >= until:
                out.append((name, f"skip: already at step {step}"))
            else:
                out.append((name, "resume"))
        elif run_dir.exists():
            out.append((name, "skip: run directory exists with nothing to resume from"))
        else:
            out.append((name, "start"))
    return out


def _diverged_at(run_dir):
    """the step at which a run's loss stopped being finite.

    args:
        run_dir: run directory.
    returns:
        that step, or None.
    """
    try:
        return json.loads((Path(run_dir) / "meta.json").read_text()).get("diverged_at")
    except (OSError, ValueError):
        return None


def command(config_path, name, action, device, until=None):
    """build the training command and environment for one run.

    args:
        config_path: training config file.
        name: run name.
        action: "start" or "resume".
        device: position of the gpu among the visible ones, as a string, or "cpu".
        until: step to stop at, or None.
    returns:
        (argument list, environment dict).
    """
    cmd = [sys.executable, str(_TRAIN_SCRIPT), "--config", str(config_path), "--name", name,
           "--device", "cpu" if device == "cpu" else "cuda"]
    if action == "resume":
        cmd.append("--resume")
    if until is not None:
        cmd += ["--until", str(until)]
    # each job sees only its own gpu, so it always trains on cuda:0; if this process was already limited to
    # some gpus, device counts among those
    visible = [d.strip() for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if d.strip()]
    gpu = visible[int(device)] if visible and device.isdigit() and int(device) < len(visible) else device
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "" if device == "cpu" else gpu}
    return cmd, env


class _Stop:
    """turns ctrl-c or sigterm into a request to pause the running jobs and launch no more."""

    def __init__(self):
        """install the handlers."""
        self.requested = False
        self._old = {sig: signal.signal(sig, self._handle) for sig in (signal.SIGINT, signal.SIGTERM)}

    def _handle(self, signum, frame):
        """record the request.

        args:
            signum: signal number.
            frame: current stack frame (unused).
        """
        self.requested = True

    def restore(self):
        """put back the previous handlers."""
        for sig, handler in self._old.items():
            signal.signal(sig, handler)


def run_all(config_path, names, devices, until=None, poll=2.0, grace=_GRACE_SECONDS, log=print):
    """train runs with at most one job per device, until every run is done or a stop is requested.

    args:
        config_path: training config file.
        names: run names, in launch order.
        devices: list of gpu indices as strings, or ["cpu", ...].
        until: step to stop each run at, or None.
        poll: seconds between checks on the running jobs.
        grace: seconds a new job runs before a stop signals it.
        log: function for progress lines; each gets a timestamp.
    returns:
        dict from run name to its outcome: the skip reason, "complete", "stopped at step N",
        "paused at step N", "diverged at step N" or "failed (exit N)".
    """
    raw_log = log

    def log(line):
        """log one line with the time."""
        raw_log(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {line}")

    cfg = yaml.safe_load(Path(config_path).read_text())
    tr.validate_config(cfg)
    claim = runner_claim(cfg)
    if pid := tr.live_claim(claim):
        raise tr.RunConfigError(f"a runner is already going (process {pid})")
    queue = plan(names, cfg, until)
    outcome = {n: a for n, a in queue if a.startswith("skip")}
    queue = [(n, a) for n, a in queue if not a.startswith("skip")]
    for name, reason in outcome.items():
        log(f"{name}: {reason}")
    Path(cfg["output_root"]).mkdir(parents=True, exist_ok=True)
    tr.write_claim(claim)
    # name -> [process, device, log file, signalled, start time]
    free, jobs, announced = list(devices), {}, False
    stop = _Stop()
    try:
        while queue or jobs:
            if stop.requested:
                if not announced:
                    log("stop requested: pausing running jobs, launching no more")
                    announced, queue = True, []
                for job in jobs.values():
                    if not job[3] and time.monotonic() - job[4] >= grace:
                        job[0].send_signal(signal.SIGTERM)
                        job[3] = True
            while queue and free and not stop.requested:
                (name, action), device = queue.pop(0), free.pop(0)
                cmd, env = command(config_path, name, action, device, until)
                out = open(Path(cfg["output_root"]) / f"{name}.log", "a")
                # own session, so a terminal ctrl-c reaches the jobs only through this runner
                proc = subprocess.Popen(cmd, env=env, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
                jobs[name] = [proc, device, out, False, time.monotonic()]
                log(f"{name}: {action} on {device}")
            time.sleep(poll)
            for name in [n for n, j in jobs.items() if j[0].poll() is not None]:
                proc, device, out, signalled, _ = jobs.pop(name)
                out.close()
                free.append(device)
                outcome[name] = _describe(Path(cfg["output_root"]) / name, proc.returncode, signalled)
                log(f"{name}: {outcome[name]}")
    finally:
        stop.restore()
        for proc, _, out, _, _ in jobs.values():
            proc.wait()
            out.close()
        tr.release_claim(claim)
    return outcome


def runner_claim(cfg):
    """where the live runner records itself.

    args:
        cfg: training config.
    returns:
        path of the claim file.
    """
    return Path(cfg["output_root"]) / "suite.pid"


def progress_note(run_dir):
    """the step and mean val accuracy of a run's latest eval, read from the end of its metrics.

    args:
        run_dir: run directory.
    returns:
        text like "step 1200, val 0.034", or "" if there is no eval yet.
    """
    path = Path(run_dir) / "metrics.jsonl"
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 200_000))
            lines = f.read().decode(errors="ignore").splitlines()
    except OSError:
        return ""
    for ln in reversed(lines):
        try:
            row = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if row.get("kind") == "eval":
            val = sum(v["acc"] for v in row["val"].values()) / len(row["val"])
            return f"step {row['step']}, val {val:.3f}"
    return ""


def running(cfg):
    """the live runner and training jobs under the config's output_root.

    args:
        cfg: training config.
    returns:
        (runner pid or None, dict from run name to the pid training it).
    """
    root = Path(cfg["output_root"])
    jobs = {d.name: pid for d in sorted(root.glob("*/")) if (pid := tr.training_pid(d))}
    return tr.live_claim(runner_claim(cfg)), jobs


def _describe(run_dir, code, signalled):
    """say where a finished job left its run.

    args:
        run_dir: the run's directory.
        code: the job's exit code.
        signalled: whether the runner asked the job to pause.
    returns:
        outcome string.
    """
    if code == 3 and (div := _diverged_at(run_dir)) is not None:
        return f"diverged at step {div}"
    if code != 0:
        return f"failed (exit {code})"
    if (run_dir / "checkpoints" / "final.pt").exists():
        return "complete"
    step = t.load(run_dir / "latest.pt", map_location="cpu", weights_only=False)["step"]
    return f"{'paused' if signalled else 'stopped'} at step {step}"


def default_devices():
    """every visible gpu, or the cpu if there is none.

    returns:
        list of device names for run_all.
    """
    n = t.cuda.device_count()
    return [str(i) for i in range(n)] if n else ["cpu"]


def main(argv=None):
    """run the cli.

    args:
        argv: argument list, or None for sys.argv.
    returns:
        exit code: 0 ok, 1 a run failed, 2 bad arguments or config.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/train.yaml")
    which = ap.add_mutually_exclusive_group(required=True)
    which.add_argument("--runs", nargs="+")
    which.add_argument("--suite", nargs="+", help="one or more suite names, run in the order given")
    which.add_argument("--running", action="store_true", help="list live jobs; exit 0 if any, 1 if none")
    which.add_argument("--stop", action="store_true",
                       help="ask the live runner to pause its jobs, or the jobs themselves if the runner is gone")
    ap.add_argument("--suites", default="configs/suites.yaml")
    ap.add_argument("--devices", help="comma-separated gpu indices, or cpu; default every visible gpu")
    ap.add_argument("--until", type=int, help="stop each run at this step, ready to resume")
    ap.add_argument("--dry-run", action="store_true", help="print what would run and exit")
    args = ap.parse_args(argv)
    if args.running or args.stop:
        try:
            cfg = yaml.safe_load(Path(args.config).read_text())
        except (OSError, yaml.YAMLError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        runner, jobs = running(cfg)
        if args.stop:
            if runner is not None:
                os.kill(runner, signal.SIGTERM)
                print(f"asked runner {runner} to pause its jobs; each saves at its next step")
                return 0
            # jobs whose runner died still pause cleanly on their own sigterm
            for name, pid in jobs.items():
                os.kill(pid, signal.SIGTERM)
                print(f"asked {name} (process {pid}) to pause; it saves at its next step")
            if not jobs:
                print("nothing running")
            return 0 if jobs else 1
        if runner:
            print(f"runner: process {runner}")
        for name, pid in jobs.items():
            note = progress_note(Path(cfg["output_root"]) / name)
            print(f"training: {name} (process {pid})" + (f": {note}" if note else ""))
        if not runner and not jobs:
            print("nothing running")
        return 0 if runner or jobs else 1
    try:
        names = args.runs or [n for s in args.suite for n in load_suite(args.suites, s)]
        devices = args.devices.split(",") if args.devices else default_devices()
        if args.dry_run:
            cfg = yaml.safe_load(Path(args.config).read_text())
            tr.validate_config(cfg)
            for name, action in plan(names, cfg, args.until):
                print(f"{name}: {action}")
            print(f"devices: {', '.join(devices)}")
            return 0
        outcome = run_all(args.config, names, devices, args.until, log=lambda s: print(s, flush=True))
    except (OSError, yaml.YAMLError, tr.RunConfigError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 1 if any(v.startswith("failed") for v in outcome.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
