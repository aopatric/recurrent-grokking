"""draw the three-panel blog figures from the seed-41 runs: embedding sparsity, hop-6 loss and held-out hop-3 loss.

    uv run python scripts/blog-plots.py [--root vast-runs] [--seed 41] [--out vast-runs/blog-figures]

hop-6 loss is on chains longer than any seen in training, built only from training pairs; held-out hop-3 loss is on
training-length chains that contain at least one held-out pair. both come from the sweeps on the regular 1000-step
grid; the extra log-spaced checkpoint sweeps are dropped so the rolling mean has even spacing.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from matplotlib.ticker import FuncFormatter, LogLocator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import plot_progress as pp  # noqa: E402
import verdicts  # noqa: E402

plt = pp.plt
RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
DASHED = (0, (4, 2))
# rolling window over grid sweeps (1000 steps apart)
SMOOTH_SWEEPS = 3
LAST_STEP = 20000

# each figure: file name, title, and its runs as (config, label, colour, dashed)
FIGURES = [
    ("looping", "Looping without normalization",
     [("t1_anchor_nope_ref", "1 layer, 1 pass (Nanda's model)", pp.SERIES[0], False),
      ("t2_ut_T2", "1 layer looped 2×", pp.SERIES[1], False),
      ("t1_ut_ref", "1 layer looped 8×", pp.SERIES[2], False),
      ("t1_unshared_L8", "8 distinct layers", pp.SERIES[3], False)]),
    ("normalization", "Normalization",
     [("t1_ut_ref", "UT, no norm", pp.SERIES[0], False),
      ("t2_ut_statenorm_ref", "UT, norm on the carried state", pp.SERIES[1], False),
      ("t1_ut_sandwich", "UT, sandwich norms", pp.SERIES[2], False),
      ("t2_huginn_nosandwich", "Huginn, norm on the carried state", pp.SERIES[1], True),
      ("t1_huginn_ref", "Huginn, sandwich norms (reference)", pp.SERIES[2], True)]),
    ("injection", "Input injection and prelude/coda (Huginn)",
     [("t1_huginn_ref", "Huginn (reference)", pp.SERIES[0], False),
      ("t1_huginn_embedstart", "embedding start instead of noise", pp.SERIES[2], False),
      ("t1_huginn_noinject", "no input injection", pp.SERIES[1], False),
      ("t1_huginn_noprecoda", "no prelude/coda", pp.SERIES[3], False)]),
    ("weight_decay", "Weight decay strength (Huginn)",
     [("t2_huginn_wd0", "λ = 0", RAMP[0], False),
      ("t2_huginn_wd05", "λ = 0.5", RAMP[1], False),
      ("t1_huginn_ref", "λ = 1 (reference)", RAMP[2], False),
      ("t3_huginn_wd2", "λ = 2", RAMP[3], False)]),
]

PANELS = [
    ("gini", "steps", "Fourier sparsity of W_E (Gini)", "gini coefficient", False),
    ("hop6", "sweep_steps", "loss, 6-hop chains, seen pairs", "cross-entropy", True),
    ("sys3", "sweep_steps", "loss, 3-hop chains, held-out pair", "cross-entropy", True),
]


def smooth(ys, n):
    """centred rolling mean, shrinking the window at the ends.

    args:
        ys: values.
        n: window length.
    returns:
        array of the same length.
    """
    ys = np.asarray(ys, dtype=float)
    lo, hi = n // 2, n - n // 2
    return np.array([ys[max(0, i - lo):i + hi].mean() for i in range(len(ys))])


def load_curves(run_dir):
    """read the three blog measures from one run, up to the judged step.

    args:
        run_dir: run directory.
    returns:
        dict with eval steps, sweep steps, and one array per measure.
    """
    rows = [r for r in verdicts.load_rows(run_dir) if r["step"] <= LAST_STEP]
    evals = [r for r in rows if r["kind"] == "eval"]
    sweeps = [r for r in rows if r["kind"] == "sweep" and r["step"] % 1000 == 0]
    sys3 = []
    for r in sweeps:
        groups = {k: v for k, v in r["at_eval_depth"]["systematic"].items() if k.startswith("3.")}
        sys3.append(np.mean([g["loss"] for g in groups.values()]))
    return {
        "steps": np.array([r["step"] for r in evals]),
        "gini": np.array([r["progress"]["gini_W_E"] for r in evals]),
        "sweep_steps": np.array([r["step"] for r in sweeps]),
        "hop6": smooth([r["at_eval_depth"]["extrap_depth"]["6"]["loss"] for r in sweeps], SMOOTH_SWEEPS),
        "sys3": smooth(sys3, SMOOTH_SWEEPS),
    }


def plot_figure(title, runs, out):
    """draw one three-panel figure.

    args:
        title: figure title.
        runs: list of (label, colour, dashed, curves).
        out: output path.
    returns:
        path to the written figure.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), facecolor=pp.SURFACE)
    for ax, (key, xkey, panel, ylabel, log_y) in zip(axes, PANELS):
        pp.style_axis(ax, panel, ylabel, log_y=log_y)
        for label, colour, dashed, c in runs:
            ax.plot(c[xkey], c[key], color=colour, linewidth=1.6, linestyle=DASHED if dashed else "-", label=label)
        if log_y:
            # ln 113: the loss of a uniform guess
            ax.axhline(np.log(113), color=pp.INK_2, linewidth=0.8, linestyle=(0, (1, 2)))
            ax.yaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
            ax.yaxis.set_minor_locator(LogLocator(subs=()))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
        ax.set_xlim(0, LAST_STEP)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 0.94), ncol=min(len(runs), 3),
               frameon=False, fontsize=9.5, labelcolor=pp.INK, handlelength=3)
    fig.suptitle(title, x=0.01, y=0.99, ha="left", fontsize=14, color=pp.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.84 if len(runs) > 3 else 0.86))
    fig.savefig(out, dpi=160, facecolor=pp.SURFACE)
    plt.close(fig)
    return out


def main(argv=None):
    """run the cli.

    args:
        argv: argument list, or None for sys.argv.
    returns:
        exit code: 0 ok.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="vast-runs")
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--out", help="output directory (default: <root>/blog-figures)")
    args = ap.parse_args(argv)
    root = Path(args.root)
    out_dir = Path(args.out) if args.out else root / "blog-figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = {}
    for name, title, spec in FIGURES:
        runs = []
        for config, label, colour, dashed in spec:
            run_dir = root / f"{config}_p113_h3_x6_s{args.seed}"
            if config not in cache:
                cache[config] = load_curves(run_dir)
            runs.append((label, colour, dashed, cache[config]))
        print(plot_figure(title, runs, out_dir / f"{name}.png"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
