"""shared loading, smoothing and styling for the write-up's figure scripts."""

import csv
import sys
from pathlib import Path

import numpy as np
from matplotlib.ticker import FuncFormatter, LogLocator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import plot_progress as pp  # noqa: E402
import verdicts  # noqa: E402

plt = pp.plt
REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
OUT = REPO / "figures" / "out"
# the step every training figure is judged and cut at
LAST_STEP = 20000
# loss of a uniform guess over the 113 answers
UNIFORM_LOSS = float(np.log(113))
# ordinal blue ramp for ordered settings such as weight decay strength
RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
DASHED = (0, (4, 2))
RAW_ALPHA = 0.25
# the write-up's text sizes: figure title, panel title, axis label, tick label, legend, small notes in a panel
TITLE_SIZE = 16
PANEL_SIZE = 12
LABEL_SIZE = 10
TICK_SIZE = 9.5
LEGEND_SIZE = 11
NOTE_SIZE = 9.5


def text_sizes(ax):
    """set an axis's tick labels and axis labels to the write-up's sizes.

    args:
        ax: matplotlib axis.
    """
    ax.tick_params(labelsize=TICK_SIZE)
    ax.xaxis.label.set_size(LABEL_SIZE)
    ax.yaxis.label.set_size(LABEL_SIZE)


# mean val accuracy at which a run counts as grokked
GROK_ACC = 0.9
CHANCE_ACC = 1 / 113


def val_series(path, last=LAST_STEP):
    """a run's eval steps and mean val accuracy over 1-3 hops, up to a step.

    args:
        path: run directory.
        last: last step kept.
    returns:
        (steps, accuracy) arrays.
    """
    _, evals, _ = load_rows(path, last)
    return np.array([r["step"] for r in evals]), np.array([mean_acc(r["val"]) for r in evals])


def run_dir(name, root=None):
    """path of a committed run.

    args:
        name: run name.
        root: folder holding run directories (default results/runs).
    returns:
        path to the run directory.
    """
    return Path(root or RESULTS / "runs") / name


def mean_acc(split):
    """mean accuracy over the hop groups of one eval split.

    args:
        split: dict from hop count to {"loss", "acc"}.
    returns:
        float.
    """
    return float(np.mean([g["acc"] for g in split.values()]))


def mean_loss(split):
    """mean loss over the hop groups of one eval split.

    args:
        split: dict from hop count to {"loss", "acc"}.
    returns:
        float.
    """
    return float(np.mean([g["loss"] for g in split.values()]))


def load_rows(path, last=LAST_STEP):
    """a run's metrics rows up to a step, split into evals and regular sweeps.

    args:
        path: run directory.
        last: last step kept.
    returns:
        (rows, evals, sweeps): all rows up to last, eval rows, and sweep rows on the 1000-step grid (the extra
        log-spaced checkpoint sweeps are dropped so the spacing is even).
    """
    rows = [r for r in verdicts.load_rows(path) if r["step"] <= last]
    evals = [r for r in rows if r["kind"] == "eval"]
    sweeps = [r for r in rows if r["kind"] == "sweep" and r["step"] % 1000 == 0]
    return rows, evals, sweeps


def grok_step(rows, at=LAST_STEP):
    """the first eval step with mean val accuracy at the grokking threshold, or None.

    args:
        rows: metrics rows.
        at: last step judged.
    returns:
        int or None.
    """
    return verdicts.verdict(rows, at=at)["t_grok"]


def load_steps(path):
    """a run's per-step training log from steps.csv.

    args:
        path: run directory.
    returns:
        dict of arrays: step, depth, loss, grad_norm.
    """
    with open(Path(path) / "steps.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) if r[k] != "" else np.nan for r in rows]) for k in rows[0]}


def median(ys, n):
    """centred rolling median, shrinking the window at the ends and skipping nans.

    args:
        ys: values.
        n: window length; 1 or less returns the values unchanged.
    returns:
        array of the same length.
    """
    ys = np.asarray(ys, dtype=float)
    if n <= 1:
        return ys
    lo, hi = n // 2, n - n // 2
    return np.array([np.nanmedian(ys[max(0, i - lo):i + hi]) for i in range(len(ys))])


def draw(ax, xs, ys, colour, label=None, dashed=False, window=1):
    """draw one series, as a median line over its faint raw curve when window > 1.

    args:
        ax: matplotlib axis.
        xs, ys: data.
        colour: line colour.
        label: legend label.
        dashed: dashed line style.
        window: rolling median window, in points.
    """
    style = DASHED if dashed else "-"
    if window > 1:
        ax.plot(xs, ys, color=colour, linewidth=0.8, linestyle=style, alpha=RAW_ALPHA)
        ys = median(ys, window)
    ax.plot(xs, ys, color=colour, linewidth=1.6, linestyle=style, label=label)


def loss_axis(ax):
    """log y axis with plain tick labels and the uniform-guess line.

    args:
        ax: matplotlib axis with a log y scale.
    """
    ax.axhline(UNIFORM_LOSS, color=pp.INK_2, linewidth=0.8, linestyle=(0, (1, 2)))
    ax.yaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
    ax.yaxis.set_minor_locator(LogLocator(subs=()))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))


def mark_step(ax, step, colour, hollow=False):
    """a small triangle on the top edge of a panel, at one step.

    args:
        ax: matplotlib axis.
        step: step to mark, or None for no mark.
        colour: marker colour.
        hollow: draw an outline only (used for dashed runs).
    """
    if step is not None:
        ax.plot([step], [1.0], transform=ax.get_xaxis_transform(), marker="v", markersize=6, color=colour,
                markerfacecolor="none" if hollow else colour, clip_on=False, linestyle="none")


def tidy_log_ticks(ax):
    """thin a log axis's tick labels to one per decade when it spans more than four decades.

    args:
        ax: matplotlib axis with a log y scale and its limits set.
    """
    lo, hi = ax.get_ylim()
    if np.log10(hi / lo) > 4:
        ax.yaxis.set_major_locator(LogLocator(subs=(1,)))


def log_limits(series, pad=1.4):
    """y limits for a log axis covering every series.

    args:
        series: iterable of value arrays.
        pad: factor added below and above.
    returns:
        (low, high).
    """
    ys = np.concatenate([np.asarray(s, dtype=float) for s in series])
    ys = ys[np.isfinite(ys) & (ys > 0)]
    return ys.min() / pad, ys.max() * pad


def legend(fig, ax, n, y=0.95):
    """one figure-level legend from an axis's labelled lines.

    args:
        fig: matplotlib figure.
        ax: axis whose labelled lines go in the legend.
        n: number of entries, to pick the column count.
        y: top of the legend, in figure coordinates.
    """
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, y), ncol=min(n, 3), frameon=False,
               fontsize=9.5, labelcolor=pp.INK, handlelength=3)


def save(fig, out):
    """write a figure and close it.

    args:
        fig: matplotlib figure.
        out: output path.
    returns:
        the output path.
    """
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, facecolor=pp.SURFACE)
    plt.close(fig)
    return out


# dot-plot figures (instability, outcomes): one row per run, grouped, with the panels centred and a legend box to
# their right. seed 41 is drawn filled, seed 42 hollow
SEEDS = (41, 42)
# timeline line widths: at or above the grokking threshold (the goal, thick) and below it (thin)
HELD_WIDTH = 5.5
LOST_WIDTH = 1.2
# their grid, a step darker than the default so the dotted chance and grokked lines don't stand out over it
DOT_GRID = "#cfcdc8"


def k_steps(ax):
    """step axis with tick labels in thousands.

    args:
        ax: matplotlib axis.
    """
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1000:g}k" if x else "0"))


def seed_dot(colour, seed, size=8):
    """dot style for one run: filled for seed 41, hollow for seed 42.

    args:
        colour: family colour.
        seed: seed.
        size: marker size.
    returns:
        dict of matplotlib line keyword arguments.
    """
    return dict(marker="o", markersize=size, markeredgecolor=colour, markeredgewidth=1.6, linestyle="none",
                markerfacecolor=colour if seed == SEEDS[0] else pp.SURFACE, clip_on=False)


def below_spans(steps, acc, start, gap):
    """stretches of evals below the grokking threshold from a step on, each eval covering half the eval spacing on
    either side, clipped to [start, LAST_STEP].

    args:
        steps, acc: eval steps and mean val accuracy, from start on.
        start: first step covered.
        gap: eval spacing.
    returns:
        list of [start, end] steps.
    """
    spans = []
    for s, a in zip(steps, acc):
        if a < GROK_ACC:
            lo, hi = max(s - gap / 2, start), min(s + gap / 2, LAST_STEP)
            if spans and lo <= spans[-1][1]:
                spans[-1][1] = hi
            else:
                spans.append([lo, hi])
    return spans


def dot_figure(size, widths):
    """a figure of side-by-side dot-plot panels sharing one row axis.

    args:
        size: (width, height) in inches.
        widths: relative panel widths.
    returns:
        (figure, axes).
    """
    fig = plt.figure(figsize=size, facecolor=pp.SURFACE, layout="constrained")
    fig.get_layout_engine().set(rect=(0.02, 0.02, 0.96, 0.92), w_pad=0.15, h_pad=0.1)
    return fig, fig.subplots(1, len(widths), sharey=True, gridspec_kw={"width_ratios": widths})


def row_positions(groups):
    """y positions for rows grouped under headers, top down.

    args:
        groups: each row's group, in order.
    returns:
        (row ys, [(header y, group)], bottom y).
    """
    ys, heads, y, prev = [], [], 0.0, None
    for group in groups:
        if group != prev:
            y -= 0.5 if prev else 0
            heads.append((y, group))
            y -= 0.95
            prev = group
        ys.append(y)
        y -= 1
    return ys, heads, y


def label_rows(ax, labels, ys, heads, bottom):
    """row labels and bold group headers on a dot plot's first panel.

    args:
        ax: first panel.
        labels: row labels.
        ys, heads, bottom: from row_positions.
    """
    ax.set_yticks(ys)
    ax.set_yticklabels(labels)
    for yy, group in heads:
        ax.text(-0.03, yy, group, transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=LABEL_SIZE + 0.5, color=pp.INK, fontweight="bold")
    ax.set_ylim(bottom + 0.4, 0.5)


def timeline(ax, yy, colour, start, spans):
    """a run's line from a step to LAST_STEP: thin throughout, thick between the stretches below the threshold.

    args:
        ax: matplotlib axis.
        yy: row y.
        colour: family colour.
        start: first step drawn.
        spans: stretches below the threshold, from below_spans.
    """
    ax.plot([start, LAST_STEP], [yy, yy], color=colour, linewidth=LOST_WIDTH, solid_capstyle="butt")
    edges = [start] + [e for span in spans for e in span] + [LAST_STEP]
    for lo, hi in zip(edges[::2], edges[1::2]):
        if hi > lo:
            ax.plot([lo, hi], [yy, yy], color=colour, linewidth=HELD_WIDTH, solid_capstyle="butt")


def style_dot_panels(axes, specs):
    """titles, axis labels, limits and grid for dot-plot panels; the first is a step axis.

    args:
        axes: panels.
        specs: (title, x label, x limits) per panel.
    """
    for ax, (title, xlabel, lim) in zip(axes, specs):
        pp.style_axis(ax, "", "")
        # room under each title for the first panel's line key
        ax.set_title(title, loc="center", fontsize=PANEL_SIZE, color=pp.INK, pad=34)
        ax.set_xlabel(xlabel, color=pp.INK_2)
        ax.set_xlim(*lim)
        ax.set_axisbelow(True)
        ax.grid(True, color=DOT_GRID, linewidth=0.7)
        text_sizes(ax)
    axes[0].tick_params(axis="y", labelsize=LABEL_SIZE, labelcolor=pp.INK_2)
    k_steps(axes[0])
    axes[0].set_xticks(np.arange(0, LAST_STEP + 1, 5000))


def accuracy_marks(ax, bottom):
    """dotted chance and grokked lines on an accuracy panel, labelled at its foot.

    args:
        ax: matplotlib axis.
        bottom: bottom y, from row_positions.
    """
    for x, label in [(CHANCE_ACC, "chance"), (GROK_ACC, "grokked")]:
        ax.axvline(x, color=pp.INK_2, linewidth=0.9, linestyle=(0, (1, 2)))
        ax.text(x + (0.015 if x < 0.5 else -0.015), bottom + 0.55, label, fontsize=NOTE_SIZE, color=pp.INK_2,
                ha="left" if x < 0.5 else "right", va="bottom")


def panel_key(ax, handles, labels, handlelength=2.5, spacing=2.5, textpad=None):
    """a one-line key under a panel's title, for what applies to that panel only.

    args:
        ax: the panel.
        handles, labels: key entries.
        handlelength: legend handle length.
        spacing: space between entries.
        textpad: space between a handle and its label (None for matplotlib's default).
    """
    ax.legend(handles=handles, labels=labels, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(handles),
              frameon=False, fontsize=NOTE_SIZE, labelcolor=pp.INK_2, columnspacing=spacing, handlelength=handlelength,
              handletextpad=textpad, borderaxespad=0.3)


def line_key(ax):
    """the timeline's key, under the panel's title.

    args:
        ax: the timeline panel.
    """
    lines = [plt.Line2D([], [], color=pp.INK_2, linewidth=w) for w in (HELD_WIDTH, LOST_WIDTH)]
    panel_key(ax, lines, [f"val ≥ {GROK_ACC:g}", f"val < {GROK_ACC:g}"])


def finish_dot_figure(fig, axes, title, handles, labels):
    """the title, the panels centred in the image (the row labels hang off the left, so the layout narrows on the
    right by as much), and the legend box in the space that leaves on the right.

    args:
        fig: figure from dot_figure.
        axes: its panels.
        title: figure title.
        handles, labels: legend entries.
    """
    fig.suptitle(title, x=0.02, y=0.975, ha="left", fontsize=TITLE_SIZE, color=pp.INK)
    engine = fig.get_layout_engine()
    # twice, since the panels' tick labels shift a little as they narrow
    for _ in range(2):
        fig.draw_without_rendering()
        left, right = axes[0].get_position().x0, axes[-1].get_position().x1
        x0, y0, w, h = engine.get()["rect"]
        engine.set(rect=(x0, y0, w - (left - (1 - right)), h))
    fig.draw_without_rendering()
    box = axes[-1].get_position()
    fig.legend(handles=handles, labels=labels, loc="upper left", bbox_to_anchor=(box.x1 + 0.02, box.y1), ncol=1,
               frameon=True, facecolor=pp.SURFACE, edgecolor=DOT_GRID, framealpha=1, fancybox=False,
               fontsize=LEGEND_SIZE, labelcolor=pp.INK, labelspacing=1.0, handletextpad=0.8, borderpad=1.1)
