"""draw one comparison figure per ablated component from the seed-41 runs, with a summary table per figure.

    uv run python scripts/first-seed-plots.py [--root vast-runs] [--seed 41] [--out vast-runs/figures]

each figure has six panels: excluded loss, restricted loss, fourier sparsity of the embedding, smoothed test
accuracy, depth extrapolation accuracy and held-out extrapolation accuracy. runs that are missing are skipped.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import plot_progress as pp  # noqa: E402
import verdicts  # noqa: E402

plt = pp.plt
# ordinal blue ramp for ordered settings such as weight decay strength (validated: light end clears 2:1)
RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
SMOOTH_EVALS = 10
# loss panels set their y range from this step on
LOSS_FROM = 1000
# the step runs are judged at
LAST_STEP = 20000
DASHED = (0, (4, 2))

# each figure: file name, title, what the line styles mean, and its runs as (config, label, colour, dashed)
COMPARISONS = [
    ("1_looping", "Looping without normalization: where grokking breaks",
     "no normalization anywhere; every model has one layer's width",
     [("t1_anchor_nope_ref", "1 layer, 1 pass (Nanda's model, no positions)", pp.SERIES[0], False),
      ("t2_ut_T2", "1 layer looped 2×", pp.SERIES[1], False),
      ("t1_ut_ref", "1 layer looped 8×", pp.SERIES[2], False),
      ("t1_unshared_L8", "8 distinct layers, 1 pass each", pp.SERIES[3], False)]),
    ("2_normalization", "Normalization",
     "colour: which norm · solid: one layer looped 8× · dashed: the Huginn recipe",
     [("t1_ut_ref", "looped layer, no norm", pp.SERIES[0], False),
      ("t2_ut_statenorm_ref", "looped layer, norm on the carried state", pp.SERIES[1], False),
      ("t1_ut_sandwich", "looped layer, sandwich norms", pp.SERIES[2], False),
      ("t2_huginn_nosandwich", "Huginn, norm on the carried state", pp.SERIES[1], True),
      ("t1_huginn_ref", "Huginn, sandwich norms (reference)", pp.SERIES[2], True)]),
    ("3_prelude_coda", "Prelude and coda layers",
     "colour: with or without one plain layer before and after the loop · solid: looped layer, no norms · "
     "dashed: Huginn (sandwich norms)",
     [("t1_ut_ref", "looped layer", pp.SERIES[0], False),
      ("t2_ut_precoda", "looped layer + prelude and coda", pp.SERIES[1], False),
      ("t1_huginn_noprecoda", "Huginn without prelude and coda", pp.SERIES[0], True),
      ("t1_huginn_ref", "Huginn (reference)", pp.SERIES[1], True)]),
    ("4_injection_start", "Input injection and the loop's starting state",
     "colour: how the input enters the loop · solid: looped layer, no norms · dashed: Huginn",
     [("t1_ut_ref", "looped layer, no injection (starts from the embedding)", pp.SERIES[0], False),
      ("t1_ut_inject", "looped layer, injection, starts from the embedding", pp.SERIES[1], False),
      ("t1_huginn_noinject", "Huginn, no injection (starts from the embedding)", pp.SERIES[0], True),
      ("t1_huginn_embedstart", "Huginn, injection, starts from the embedding", pp.SERIES[1], True),
      ("t1_huginn_ref", "Huginn, injection, starts from noise (reference)", pp.SERIES[2], True)]),
    ("5_depth_schedule", "Number of passes during training: fixed or sampled",
     "colour: depth schedule · solid: looped layer with a norm on the carried state · dashed: Huginn",
     [("t2_ut_statenorm_ref", "state-norm loop, fixed 8 passes", pp.SERIES[0], False),
      ("t2_ut_statenorm_sampled", "state-norm loop, sampled passes (mean 8)", pp.SERIES[1], False),
      ("t1_huginn_fixedT8", "Huginn, fixed 8 passes", pp.SERIES[0], True),
      ("t1_huginn_ref", "Huginn, sampled passes (reference)", pp.SERIES[1], True),
      ("t1_huginn_mindepth2", "Huginn, sampled passes, never fewer than 2", pp.SERIES[2], True)]),
    ("6_backprop", "Backprop through the last 4 passes or through all of them",
     "both are the Huginn recipe with sampled depth",
     [("t1_huginn_ref", "last 4 passes (reference)", pp.SERIES[0], False),
      ("t2_huginn_fullbackprop", "all passes", pp.SERIES[1], False)]),
    ("7_weight_decay", "Weight decay strength (Huginn)",
     "darker is stronger decay; λ = 1 is Nanda's setting and the reference",
     [("t2_huginn_wd0", "λ = 0", RAMP[0], False),
      ("t2_huginn_wd05", "λ = 0.5", RAMP[1], False),
      ("t1_huginn_ref", "λ = 1 (reference)", RAMP[2], False),
      ("t3_huginn_wd2", "λ = 2", RAMP[3], False)]),
    ("8_decay_on_gains", "Which weights are decayed (Huginn)",
     "λ = 1 throughout; only the norm gains' decay changes",
     [("t1_huginn_ref", "decay on every weight (reference)", pp.SERIES[0], False),
      ("t2_huginn_innergainsnowd", "no decay on the inner norm gains", pp.SERIES[1], False),
      ("t2_huginn_gainsnowd", "no decay on any norm gain", pp.SERIES[2], False)]),
    ("9_positions", "Position embeddings in the one-pass model",
     "Nanda's one-layer model, one pass, multi-hop data",
     [("t1_anchor_nope_ref", "no positions", pp.SERIES[0], False),
      ("t3_anchor_learnedpos_ref", "learned positions", pp.SERIES[1], False),
      ("t3_anchor_learnedpos_wd2", "learned positions, λ = 2", pp.SERIES[2], False)]),
]


def mean_acc(groups):
    """mean accuracy over every group of a split.

    args:
        groups: dict from group label to {"loss", "acc"}.
    returns:
        float.
    """
    return float(np.mean([g["acc"] for g in groups.values()]))


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
    """read the curves a comparison figure needs from one run.

    args:
        run_dir: run directory.
    returns:
        dict with eval steps, sweep steps, and one array per measure.
    """
    rows = verdicts.load_rows(run_dir)
    evals = [r for r in rows if r["kind"] == "eval"]
    sweeps = [r for r in rows if r["kind"] == "sweep"]
    prog = [r["progress"] for r in evals]
    out = {
        "steps": np.array([r["step"] for r in evals]),
        "excluded": np.array([p["excluded_loss"] for p in prog]),
        "restricted": np.array([p["restricted_loss"] for p in prog]),
        "gini": np.array([p["gini_W_E"] for p in prog]),
        "test_raw": np.array([mean_acc(r["val"]) for r in evals]),
        "sweep_steps": np.array([r["step"] for r in sweeps]),
        "depth_extrap": np.array([mean_acc(r["at_eval_depth"]["extrap_depth"]) for r in sweeps]),
        "heldout_extrap": np.array([mean_acc(r["at_eval_depth"]["systematic_extrap"]) for r in sweeps]),
        "verdict": verdicts.judge(run_dir),
    }
    out["test"] = smooth(out["test_raw"], SMOOTH_EVALS)
    return out


PANELS = [
    ("excluded", "steps", "excluded loss (hop 1, train pairs)", "cross-entropy", True),
    ("restricted", "steps", "restricted loss (hop 1, all pairs)", "cross-entropy", True),
    ("gini", "steps", "fourier sparsity of the embedding (gini of W_E)", "gini coefficient", False),
    ("test", "steps", f"test accuracy, hops 1–3 (smoothed: mean of {SMOOTH_EVALS} evals)", "accuracy", False),
    ("depth_extrap", "sweep_steps", "depth extrapolation: hops 4–6", "accuracy", False),
    ("heldout_extrap", "sweep_steps", "held-out extrapolation: hops 4–6, one held-out pair", "accuracy", False),
]


def plot_comparison(title, note, runs, out, seed):
    """draw one six-panel comparison figure.

    args:
        title: figure title.
        note: one line on what the colours and line styles mean.
        runs: list of (label, colour, dashed, curves).
        out: output path.
        seed: the runs' seed, for the footnote.
    returns:
        path to the written figure.
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), facecolor=pp.SURFACE)
    for ax, (key, xkey, panel, ylabel, log_y) in zip(axes.flat, PANELS):
        pp.style_axis(ax, panel, ylabel, log_y=log_y)
        ax.title.set_fontsize(10)
        for label, colour, dashed, c in runs:
            ax.plot(c[xkey], c[key], color=colour, linewidth=1.4, linestyle=DASHED if dashed else "-", label=label)
        if ylabel == "accuracy":
            ax.set_ylim(0, 1.02)
        if log_y:
            # an unnormalized loop's first evals can sit orders of magnitude above everything after
            later = [y for _, _, _, c in runs for x, y in zip(c[xkey], c[key]) if x >= LOSS_FROM]
            ax.set_ylim(top=1.5 * max(later))
        if key == "test":
            ax.axhline(verdicts.GROK, color=pp.INK_2, linewidth=0.8, linestyle=(0, (1, 2)))
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 0.935), ncol=len(runs) if len(runs) <= 4 else 3,
               frameon=False, fontsize=9, labelcolor=pp.INK, handlelength=3)
    fig.suptitle(title, x=0.01, y=0.99, ha="left", fontsize=14, color=pp.INK)
    fig.text(0.01, 0.948, note, ha="left", fontsize=9, color=pp.INK_2)
    fig.text(0.01, 0.008, f"seed {seed}, one run per line. test accuracy is a rolling mean over {SMOOTH_EVALS} evals "
             "(1k steps), and its dotted line is the 0.9 grokking threshold; every other panel is unsmoothed.\n"
             f"the loss panels' y range is set from step {LOSS_FROM} on, so a larger loss at initialization runs off "
             "the top. extrapolation is measured at the training depth, on chains longer than any seen in training.",
             ha="left", va="bottom", fontsize=8, color=pp.INK_2, linespacing=1.5)
    fig.tight_layout(rect=(0, 0.045, 1, 0.905))
    fig.savefig(out, dpi=150, facecolor=pp.SURFACE)
    plt.close(fig)
    return out


def summary_table(title, runs):
    """the figure's runs as a markdown table of verdicts, late test accuracy and final extrapolation accuracy.

    args:
        title: figure title.
        runs: list of (label, colour, dashed, curves).
    returns:
        markdown string.
    """
    lines = [f"## {title}", "", "| run | verdict | t_grok | stable | test acc at 20k | mean test acc, last 5k | "
             "depth extrap | held-out extrap |", "|---|---|---|---|---|---|---|---|"]
    for label, _, _, c in runs:
        v = c["verdict"]
        late = c["test_raw"][(c["steps"] > LAST_STEP - 5000) & (c["steps"] <= LAST_STEP)]
        cells = [v["verdict"], v["t_grok"], v["stable"], v["val"], round(float(late.mean()), 3),
                 round(float(c["depth_extrap"][-1]), 3),
                 round(float(c["heldout_extrap"][-1]), 3)]
        lines.append(f"| {label} | " + " | ".join("—" if x is None else str(x) for x in cells) + " |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    """run the cli.

    args:
        argv: argument list, or None for sys.argv.
    returns:
        exit code: 0 ok, 2 no figure could be drawn.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="vast-runs")
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--out", help="output directory (default: <root>/figures)")
    args = ap.parse_args(argv)
    root = Path(args.root)
    out_dir = Path(args.out) if args.out else root / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache, tables, drawn = {}, [], 0
    for name, title, note, spec in COMPARISONS:
        runs = []
        for config, label, colour, dashed in spec:
            run_dir = root / f"{config}_p113_h3_x6_s{args.seed}"
            if not (run_dir / "metrics.jsonl").exists():
                print(f"{name}: skipping {run_dir.name} (not found)")
                continue
            if config not in cache:
                cache[config] = load_curves(run_dir)
            runs.append((label, colour, dashed, cache[config]))
        if len(runs) < 2:
            print(f"{name}: fewer than two runs, no figure")
            continue
        print(plot_comparison(title, note, runs, out_dir / f"{name}.png", args.seed))
        tables.append(summary_table(title, runs))
        drawn += 1
    if not drawn:
        return 2
    summary = out_dir / "summary.md"
    summary.write_text(f"# Seed {args.seed} comparisons\n\nverdicts are judged at step {LAST_STEP}; the "
                       "last-5k column averages every eval after step 15000, steadier than one snapshot for runs "
                       "that collapse; extrapolation columns are the last sweep.\n\n" + "\n".join(tables))
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
