"""plot a run's grokking progress measures from its metrics.jsonl.

    uv run python src/plot_progress.py runs/anchor_p113_h1 [--phases 1400,9400] [--out fig.png]

writes progress.png into the run directory by default.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# reference categorical palette (validated: adjacent cvd and normal-vision gates pass on the light surface);
# slots 3 and 4 sit below 3:1 contrast, so every line also gets a direct label
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"


def load_evals(run_dir):
    """read the routine evaluation lines of a run.

    args:
        run_dir: run directory.
    returns:
        list of eval row dicts, in step order.
    """
    with open(Path(run_dir) / "metrics.jsonl") as f:
        rows = [json.loads(line) for line in f]
    return [r for r in rows if r["kind"] == "eval"]


def style_axis(ax, title, ylabel, log_y=False):
    """apply the shared recessive axis style.

    args:
        ax: matplotlib axis.
        title: panel title.
        ylabel: y-axis label.
        log_y: use a log y scale.
    """
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=8)
    ax.set_ylabel(ylabel, fontsize=9, color=INK_2)
    ax.set_xlabel("step", fontsize=9, color=INK_2)
    if log_y:
        ax.set_yscale("log")
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8, length=0)


def spread(ys, gap):
    """push sorted-by-value positions apart so neighbours are at least gap apart.

    args:
        ys: positions in axes fraction.
        gap: minimum separation.
    returns:
        adjusted positions, in the input order.
    """
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    out = list(ys)
    for a, b in zip(order, order[1:]):
        out[b] = max(out[b], out[a] + gap)
    # if the stack ran past the top, push it back down from there
    out[order[-1]] = min(out[order[-1]], 0.98)
    for a, b in zip(order[::-1], order[-2::-1]):
        out[b] = min(out[b], out[a] - gap)
    return out


def draw_lines(ax, steps, series, ylim=None, end_labels=True, legend=True):
    """draw several series on one axis, each with a legend entry and a direct end label.

    args:
        ax: matplotlib axis.
        steps: x values, shared by every series, or a list of x values per series.
        series: list of (label, y values), drawn in palette order.
        ylim: optional fixed y limits, applied before labels are placed.
        end_labels: label each line at its right end.
        legend: add a legend when there are several series.
    """
    xs = steps if steps and isinstance(steps[0], (list, tuple)) else [steps] * len(series)
    for i, ((label, ys), x) in enumerate(zip(series, xs)):
        ax.plot(x, ys, color=SERIES[i], linewidth=1.5, label=label)
    if ylim:
        ax.set_ylim(*ylim)
    else:
        # settle the limits now; otherwise they update at draw time and the labels drift off their lines
        ax.relim()
        ax.autoscale_view()
    if end_labels:
        _end_labels(ax, xs, series)
    if legend and len(series) > 1:
        ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="best")


def _end_labels(ax, xs, series):
    """label each line at its last point, spaced so labels don't overlap.

    args:
        ax: matplotlib axis with settled limits.
        xs: x values per series.
        series: list of (label, y values).
    """
    # place end labels in axes fraction so they can be spaced apart without overlapping
    to_frac = ax.transData + ax.transAxes.inverted()
    ends = [to_frac.transform((x[-1], ys[-1]))[1] for x, (_, ys) in zip(xs, series)]
    for (label, _), x, y in zip(series, xs, spread(ends, 0.045)):
        ann = ax.annotate(label, (x[-1], y), xycoords=("data", "axes fraction"), xytext=(4, 0),
                          textcoords="offset points", fontsize=8, color=INK_2, va="center")
        ann.set_in_layout(False)


def mark_phases(axes, phases):
    """draw phase boundaries as thin vertical lines on every panel.

    args:
        axes: matplotlib axes.
        phases: step numbers.
    """
    for ax in axes:
        for x in phases:
            ax.axvline(x, color=INK_2, linewidth=0.8, linestyle=(0, (3, 3)))


def plot_run(run_dir, out=None, phases=()):
    """draw the progress-measure figure for one run from its logged metrics.

    args:
        run_dir: run directory.
        out: output path, or None for run_dir/progress.png.
        phases: optional phase-boundary steps to mark.
    returns:
        path to the written figure.
    """
    run_dir = Path(run_dir)
    return plot_rows(load_evals(run_dir), run_dir.name, out or run_dir / "progress.png", phases)


def plot_rows(rows, title, out, phases=()):
    """draw the progress-measure figure from eval rows, logged or recomputed.

    args:
        rows: eval rows in step order, each with step, val and progress.
        title: figure title.
        out: output path.
        phases: optional phase-boundary steps to mark.
    returns:
        path to the written figure.
    """
    steps = [r["step"] for r in rows]
    prog = [r["progress"] for r in rows]
    get = lambda key: [p[key] for p in prog]  # noqa: E731
    hops = sorted(rows[-1]["val"], key=int)

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8), facecolor=SURFACE)
    ax_loss, ax_acc, ax_gini, ax_w = axes.flat

    style_axis(ax_loss, "hop-1 loss and progress measures", "cross-entropy", log_y=True)
    draw_lines(ax_loss, steps, [("train", get("grid_train_loss")), ("test", get("grid_test_loss")),
                                ("restricted", get("restricted_loss")), ("excluded", get("excluded_loss"))])

    style_axis(ax_acc, "val accuracy by hop", "accuracy")
    draw_lines(ax_acc, steps, [(f"hop {h}", [r["val"][h]["acc"] for r in rows]) for h in hops], ylim=(0, 1.02))

    style_axis(ax_gini, "fourier sparsity (gini)", "gini coefficient")
    gini = [("W_E", get("gini_W_E"))]
    gini.append(("W_L", get("gini_W_L")) if "gini_W_L" in prog[-1] else ("W_U", get("gini_W_U")))
    draw_lines(ax_gini, steps, gini)

    style_axis(ax_w, "total squared weight norm", "sum of squared weights")
    weights = [("weights", get("sum_sq_weights"))]
    # runs logged before the gains were reported separately keep the single total
    if all("sum_sq_gains" in r for r in prog) and any(r["sum_sq_gains"] for r in prog):
        weights = [("without norm gains", [w - g for w, g in zip(get("sum_sq_weights"), get("sum_sq_gains"))]),
                   ("norm gains", get("sum_sq_gains"))]
    draw_lines(ax_w, steps, weights)

    mark_phases(axes.flat, phases)
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, color=INK)
    # leave a right margin for the end labels, which are kept out of the layout
    fig.tight_layout(rect=(0, 0, 0.97, 1))
    out = Path(out)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out


def plot_depth_sweep(run_dir, out=None, train_T=None):
    """plot accuracy against the number of core passes at the run's last sweep, one line per hop.

    args:
        run_dir: run directory of a model with a shared core.
        out: output path, or None for run_dir/depth_sweep.png.
        train_T: training depth to mark, or None.
    returns:
        path to the written figure.
    raises:
        ValueError: if the run logged no per-depth sweep.
    """
    run_dir = Path(run_dir)
    with open(run_dir / "metrics.jsonl") as f:
        sweeps = [r for r in map(json.loads, f) if r["kind"] == "sweep"]
    if not sweeps or not sweeps[-1]["by_T"]:
        raise ValueError(f"{run_dir} has no per-depth sweep")
    last = sweeps[-1]
    title = f"{run_dir.name}: accuracy by depth at step {last['step']}"
    return plot_sweep(last["by_T"], title, out or run_dir / "depth_sweep.png", train_T)


def plot_sweep(by_T, title, out, train_T=None):
    """plot accuracy against the number of core passes, one line per hop.

    args:
        by_T: dict from set name to {label: accuracies for T = 1, 2, ...}.
        title: figure title.
        out: output path.
        train_T: training depth to mark, or None.
    returns:
        path to the written figure.
    """
    def mean_over_steps(groups):
        """average the systematic novel-step groups ("3.k1", "3.k2", ...) into one curve per hop."""
        hops = {}
        for label, accs in groups.items():
            hops.setdefault(label.split(".")[0], []).append(accs)
        return {h: [sum(col) / len(col) for col in zip(*rows)] for h, rows in hops.items()}

    panels = [("closed chains (val, then extrapolation)", {**by_T.get("val", {}), **by_T.get("extrap_depth", {})})]
    systematic = {**mean_over_steps(by_T.get("systematic", {})), **mean_over_steps(by_T.get("systematic_extrap", {}))}
    if systematic:
        panels.append(("one held-out step (mean over its position)", systematic))

    fig, axes = plt.subplots(1, len(panels), figsize=(6.25 * len(panels), 4.5), facecolor=SURFACE, squeeze=False)
    for ax, (title, groups) in zip(axes[0], panels):
        style_axis(ax, title, "accuracy")
        ax.set_xlabel("core passes T", fontsize=9, color=INK_2)
        hops = sorted(groups, key=int)
        Ts = list(range(1, len(groups[hops[0]]) + 1))
        draw_lines(ax, Ts, [(f"hop {h}", groups[h]) for h in hops], ylim=(0, 1.02))
        if train_T:
            ax.axvline(train_T, color=INK_2, linewidth=0.8, linestyle=(0, (3, 3)))
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, color=INK)
    fig.tight_layout(rect=(0, 0, 0.96, 1))
    out = Path(out)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out


def plot_compare(runs, out):
    """overlay the key curves of several runs, one colour per run in the given order.

    args:
        runs: list of (name, eval rows) pairs, at most one per palette colour.
        out: output path.
    returns:
        path to the written figure.
    raises:
        ValueError: if there are more runs than palette colours.
    """
    if len(runs) > len(SERIES):
        raise ValueError(f"at most {len(SERIES)} runs per comparison; split them into several figures")
    xs = [[r["step"] for r in rows] for _, rows in runs]
    get = lambda rows, f: [f(r) for r in rows]  # noqa: E731
    deepest = [max(rows[-1]["val"], key=int) for _, rows in runs]

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), facecolor=SURFACE)
    panels = [
        ("hop-1 test loss (grid)", "cross-entropy", True, None,
         lambda rows, h: get(rows, lambda r: r["progress"]["grid_test_loss"])),
        ("restricted loss", "cross-entropy", True, None,
         lambda rows, h: get(rows, lambda r: r["progress"]["restricted_loss"])),
        ("val accuracy, hop 1", "accuracy", False, (0, 1.02),
         lambda rows, h: get(rows, lambda r: r["val"]["1"]["acc"])),
        ("val accuracy, deepest trained hop", "accuracy", False, (0, 1.02),
         lambda rows, h: get(rows, lambda r: r["val"][h]["acc"])),
    ]
    for ax, (title, ylabel, log_y, ylim, f) in zip(axes.flat, panels):
        style_axis(ax, title, ylabel, log_y=log_y)
        series = [(f"{name} (hop {h})" if "deepest" in title else name, f(rows, h))
                  for (name, rows), h in zip(runs, deepest)]
        draw_lines(ax, xs, series, ylim=ylim, end_labels=False, legend="deepest" in title)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False, fontsize=9, labelcolor=INK_2,
               ncol=min(3, len(runs)))
    fig.suptitle("run comparison", x=0.01, ha="left", fontsize=13, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out


def main(argv=None):
    """run the cli.

    args:
        argv: argument list, or None for sys.argv.
    returns:
        exit code: 0 ok, 2 missing metrics.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--out")
    ap.add_argument("--phases", default="", help="comma-separated phase-boundary steps")
    args = ap.parse_args(argv)
    phases = [int(x) for x in args.phases.split(",") if x]
    try:
        out = plot_run(args.run_dir, args.out, phases)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
