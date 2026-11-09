"""training-set logit margins at every checkpoint, to check for barron et al. 2026 softmax collapse.

    uv run python experiments/stability/check-margins.py RUN_DIR [RUN_DIR ...] [--device cuda]

the margin is the target logit minus the log-sum-exp of the other logits. the collapse needs margins past a
precision cliff: 16.6 in fp32, 36.7 in fp64.
"""

import argparse
import sys
from pathlib import Path

import torch as t

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import analysis  # noqa: E402

CHUNK = 8192


def margins(run, path, device):
    """training-set margins of one checkpoint.

    args:
        run: dict from analysis.load_run.
        path: checkpoint path.
        device: torch device string.
    returns:
        1-d tensor of margins, one per training example.
    """
    m = analysis.load_model(run, path, device)
    out = []
    with t.no_grad():
        for x, y in run["data"]["sets"]["train"].values():
            for i in range(0, len(x), CHUNK):
                z = m(x[i:i + CHUNK], run["T"], generator=analysis.eval_generator(run, device))
                z = z[0] if isinstance(z, tuple) else z
                if z.dim() == 3:
                    z = z[:, -1]
                z, yy = z.double(), y[i:i + CHUNK]
                target = z.gather(-1, yy[:, None])[:, 0]
                others = z.scatter(-1, yy[:, None], float("-inf"))
                out.append(target - t.logsumexp(others, -1))
    return t.cat(out)


def main(argv=None):
    """run the cli.

    args:
        argv: argument list, or None for sys.argv.
    returns:
        exit code: 0 ok.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run directories")
    ap.add_argument("--device", default="cuda" if t.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)
    for rd in args.runs:
        run = analysis.load_run(rd, device=args.device)
        print("==", run["name"], "T", run["T"])
        for step, path in analysis.checkpoints(run):
            mm = margins(run, path, args.device)
            q = t.quantile(mm.float(), t.tensor([0.5, 0.99], device=mm.device))
            print(f"step {step:>6}  median {q[0]:6.1f}  p99 {q[1]:6.1f}  max {mm.max():6.1f}  "
                  f"frac>16.6 {(mm > 16.6).float().mean():.3f}  frac>36.7 {(mm > 36.7).float().mean():.3f}",
                  flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
