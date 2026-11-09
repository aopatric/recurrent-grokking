"""draw the grokking outcomes figure (titled "Instability Overview") from every run on nanda's recipe, and with
--supporting the four seed-41 ablation figures behind it, as two rows of three panels.

    uv run python figures/ablations.py [--root results/runs] [--seed 41] [--out figures/out/dynamics] [--smooth median]
        [--supporting]

grokking_outcomes.png and .md: one row per run, grouped by what the ablation changes: its val accuracy over training
against the grokking threshold with when it grokked, the share of its time after grokking below the threshold, and
the best val accuracy it reached, and its accuracy at 20k; the per-run numbers and the instability numbers go to the
.md.

the supporting figures (<out>/supporting/): top row: the training side (mean train loss over 1-3 hops, fourier sparsity of the embedding, total squared
weight). bottom row, on one shared axis: test losses on 1-hop pairs never trained on, 3-hop chains that contain a
held-out pair, and 6-hop chains of training pairs (longer than any chain seen in training). the top row and the
1-hop test loss are logged every 100 steps; the 3- and 6-hop losses come from the 1000-step sweeps. a triangle on
each panel's top edge marks a run's grokking step (mean val accuracy reaches 0.9).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D

import collapse
import common as cm

pp = cm.pp
plt = cm.plt

# each figure: file name, title, and its runs as (config, label, colour, dashed)
FIGURES = [
    ("looping", "Looping without normalization",
     [("t1_anchor_nope_ref", "1 layer, 1 pass (Nanda's model)", pp.SERIES[0], False),
      ("t2_ut_T2", "1 layer looped 2×", pp.SERIES[1], False),
      ("t1_ut_ref", "1 layer looped 8×", pp.SERIES[2], False),
      ("t1_unshared_L8", "8 distinct layers", pp.SERIES[3], False)]),
    ("normalization", "Normalization",
     [("t1_ut_ref", "plain loop, no norm", pp.SERIES[0], False),
      ("t2_ut_statenorm_ref", "plain loop, norm on the carried state", pp.SERIES[1], False),
      ("t1_ut_sandwich", "plain loop, sandwich norms", pp.SERIES[2], False),
      ("t2_huginn_nosandwich", "Huginn, norm on the carried state", pp.SERIES[1], True),
      ("t1_huginn_ref", "Huginn, sandwich norms (reference)", pp.SERIES[2], True)]),
    ("injection", "Input injection and prelude/coda (Huginn)",
     [("t1_huginn_ref", "Huginn (reference)", pp.SERIES[0], False),
      ("t1_huginn_embedstart", "embedding start instead of noise", pp.SERIES[2], False),
      ("t1_huginn_noinject", "no input injection", pp.SERIES[1], False),
      ("t1_huginn_noprecoda", "no prelude/coda", pp.SERIES[3], False)]),
    ("weight_decay", "Weight decay strength (Huginn)",
     [("t2_huginn_wd0", "λ = 0", cm.RAMP[0], False),
      ("t2_huginn_wd05", "λ = 0.5", cm.RAMP[1], False),
      ("t1_huginn_ref", "λ = 1 (reference)", cm.RAMP[2], False),
      ("t3_huginn_wd2", "λ = 2", cm.RAMP[3], False)]),
]

# each panel: (measure, x key, title, y label, kind); "loss" panels get a log axis and the uniform-guess line
PANELS = [
    ("train", "steps", "train loss, mean over 1–3 hops", "cross-entropy", "loss"),
    ("gini", "steps", "Fourier sparsity of W_E (Gini)", "gini coefficient", "linear"),
    ("wnorm", "steps", "total squared weight", "sum of squares", "log"),
    ("grid_test", "steps", "test loss, 1-hop pairs not trained on", "cross-entropy", "loss"),
    ("sys3", "sweep_steps", "test loss, 3-hop chains with a held-out pair", "cross-entropy", "loss"),
    ("hop6", "sweep_steps", "test loss, 6-hop chains (longer than training)", "cross-entropy", "loss"),
]
# rolling median windows, in points: 5 evals (500 steps) or 3 sweeps (3000 steps)
WINDOWS = {"steps": 5, "sweep_steps": 3}
# loss axes set their range from this step on, past the large losses of the first few evals
LOSS_FROM = 1000


# the grokking outcomes figure: (group header, [(config, label)]), all seed 41 on nanda's recipe
# model families, coloured as in the instability figure so a run has one colour in both: key -> (colour, legend name)
FAMILIES = {
    "huginn": (pp.SERIES[0], "Huginn-Style Loop"),
    "plain": (pp.SERIES[2], "Plain Loop"),
    "stack": (pp.SERIES[3], "Unshared Stack"),
    "one": (pp.INK_2, "Nanda's One-Layer Model"),
}
# the outcomes figure's rows: every run in results/ on nanda's recipe for all 20k steps, grouped by what the ablation
# changes: (group, [(config, seed, family, label[, suffix])]). seed-42 rows sit under their seed-41 row. the plain
# loop with sandwich norms has two seed-41 runs: the ablation sweep's (5090) and its rerun in the t5 suite (4090)
OUTCOME_GROUPS = [
    ("No Normalization", [
        ("t1_anchor_nope_ref", 41, "one", "one-layer model (Nanda's)"),
        ("t1_anchor_nope_ref", 42, "one", "one-layer model (Nanda's)"),
        ("t2_ut_T2", 41, "plain", "plain loop, 2 passes"),
        ("t1_ut_ref", 41, "plain", "plain loop, 8 passes"),
        ("t1_ut_inject", 41, "plain", "plain loop, 8 passes, input injection"),
        ("t2_ut_precoda", 41, "plain", "plain loop, 8 passes, prelude and coda"),
        ("t1_unshared_L8", 41, "stack", "8 distinct layers")]),
    ("Normalization", [
        ("t2_ut_statenorm_ref", 41, "plain", "plain loop, state norm"),
        ("t1_ut_sandwich", 41, "plain", "plain loop, sandwich norms"),
        ("t5_ut_sandwich_const", 41, "plain", "plain loop, sandwich norms", " (4090 rerun)"),
        ("t5_ut_sandwich_const", 42, "plain", "plain loop, sandwich norms"),
        ("t5_unshared_sandwich_const", 41, "stack", "8 distinct layers, sandwich norms"),
        ("t5_unshared_sandwich_const", 42, "stack", "8 distinct layers, sandwich norms"),
        ("t5_unshared_sandwich_L2_const", 41, "stack", "2 distinct layers, sandwich norms"),
        ("t5_unshared_sandwich_L2_const", 42, "stack", "2 distinct layers, sandwich norms"),
        ("t2_huginn_nosandwich", 41, "huginn", "Huginn-style loop, state norm"),
        ("t1_huginn_ref", 41, "huginn", "Huginn-style reference, sandwich norms"),
        ("t1_huginn_ref", 42, "huginn", "Huginn-style reference, sandwich norms")]),
    ("Huginn-Style Loop, One Change From the Reference", [
        ("t1_huginn_noinject", 41, "huginn", "no input injection"),
        ("t1_huginn_noprecoda", 41, "huginn", "no prelude or coda"),
        ("t1_huginn_embedstart", 41, "huginn", "embedding start instead of noise"),
        ("t1_huginn_fixedT8", 41, "huginn", "fixed depth 8"),
        ("t1_huginn_fixedT8", 42, "huginn", "fixed depth 8"),
        ("t2_huginn_fullbackprop", 41, "huginn", "full backprop through the loop"),
        ("t1_huginn_mindepth2", 41, "huginn", "min depth 2"),
        ("t2_huginn_wd0", 41, "huginn", "weight decay λ = 0 (reference: 1)"),
        ("t2_huginn_wd05", 41, "huginn", "weight decay λ = 0.5"),
        ("t3_huginn_wd2", 41, "huginn", "weight decay λ = 2")]),
]
OUTCOMES_TITLE = "Instability Overview"
# marker at 20k for a run that never reached the grokking threshold
NO_GROK = dict(marker="x", markersize=8, markeredgewidth=2, linestyle="none", clip_on=False)
# the accuracy panel's three marks per run, on the row's line: (marker, size) for the best val accuracy, the mean
# over the last LAST_MEAN steps, and the val accuracy at 20k; drawn square, then triangle, then star, so the star is
# on top where they overlap
BEST_MARK = ("*", 12)
MEAN_MARK = ("^", 8)
FINAL_MARK = ("s", 7)
LAST_MEAN = 500


def outcome_runs(root):
    """every row of the outcomes figure with its numbers.

    args:
        root: folder holding run directories.
    returns:
        list of dicts: group, label, config, seed, family, colour, steps, acc, spans, best, last_mean (mean val
        accuracy over the last LAST_MEAN steps), below (share of evals
        after grokking below the threshold, None if it didn't grok), verdict, device.
    """
    rows = []
    for group, members in OUTCOME_GROUPS:
        for config, seed, family, label, *suffix in members:
            d = cm.run_dir(f"{config}_p113_h3_x6_s{seed}", root)
            steps, acc = cm.val_series(d)
            meta = json.loads((Path(d) / "meta.json").read_text())
            v = cm.verdicts.verdict(cm.verdicts.load_rows(d))
            below = None if v["t_grok"] is None else float(np.mean(acc[steps >= v["t_grok"]] < cm.GROK_ACC))
            rows.append({"group": group, "label": f"{label}, seed {seed}{suffix[0] if suffix else ''}",
                         "config": config, "seed": seed, "family": family, "colour": FAMILIES[family][0],
                         "steps": steps, "acc": acc, "spans": cm.below_spans(steps, acc, 0, np.median(np.diff(steps))),
                         "best": float(acc.max()), "last_mean": float(acc[steps > cm.LAST_STEP - LAST_MEAN].mean()),
                         "below": below, "verdict": v,
                         "device": meta.get("device", "?").replace("NVIDIA GeForce ", "")})
    return rows


def plot_outcomes(root, out):
    """draw the outcomes figure: one row per run on nanda's recipe, grouped by what the ablation changes. left, its
    val accuracy over training against the threshold, with when it grokked; middle, the share of its time after
    grokking below the threshold; right, the best val accuracy it reached (star) and its val accuracy at the end
    (square), with the mean over the last 500 steps between them (triangle). the per-run numbers, and the instability
    numbers for the runs that grokked, go to a markdown file next to it.

    args:
        root: folder holding run directories.
        out: output path of the image; the notes go next to it with a .md suffix.
    returns:
        (image path, notes path).
    """
    runs = outcome_runs(root)
    ys, heads, bottom = cm.row_positions([r["group"] for r in runs])
    units = -bottom
    # wider than the instability figure: its longer row labels take more of each side, since the panels are centred
    fig, (ax_t, ax_b, ax_e) = cm.dot_figure((18.5, 0.36 * units + 2.2), [1.35, 0.8, 0.8])
    for r, yy in zip(runs, ys):
        cm.timeline(ax_t, yy, r["colour"], 0, r["spans"])
        t_grok = r["verdict"]["t_grok"]
        if t_grok is None:
            ax_t.plot([cm.LAST_STEP], [yy], color=r["colour"], **NO_GROK)
        else:
            ax_t.plot([t_grok], [yy], **cm.seed_dot(r["colour"], r["seed"]))
        if r["below"] is not None:
            ax_b.plot([r["below"]], [yy], **cm.seed_dot(r["colour"], r["seed"]))
        for (marker, size), x in [(FINAL_MARK, r["verdict"]["val"]), (MEAN_MARK, r["last_mean"]),
                                  (BEST_MARK, r["best"])]:
            ax_e.plot([x], [yy], **{**cm.seed_dot(r["colour"], r["seed"], size), "marker": marker})
    cm.label_rows(ax_t, [r["label"] for r in runs], ys, heads, bottom)
    cm.style_dot_panels((ax_t, ax_b, ax_e), [
        ("Validation Drops", "step", (0, cm.LAST_STEP)),
        (f"Post-Grok Time Below val={cm.GROK_ACC:g}", "share of evals after grokking", (0, 1)),
        ("Validation Accuracy", "mean val accuracy, 1–3 hops", (0, 1))])
    cm.accuracy_marks(ax_e, bottom)
    families = [Line2D([], [], **cm.seed_dot(c, cm.SEEDS[0])) for c, _ in FAMILIES.values()]
    seeds = [Line2D([], [], **cm.seed_dot(pp.INK, s)) for s in cm.SEEDS]
    gap = Line2D([], [], linestyle="none")
    no_grok = Line2D([], [], color=pp.INK, **NO_GROK)
    labels = ([name for _, name in FAMILIES.values()] + [""]
              + [f"{fill}: seed {s}" for fill, s in zip(("filled", "hollow"), cm.SEEDS)]
              + ["", "didn't grok by 20k"])
    cm.line_key(ax_t)
    marks = [Line2D([], [], **{**cm.seed_dot(pp.INK_2, cm.SEEDS[0], size), "marker": marker})
             for marker, size in (BEST_MARK, MEAN_MARK, FINAL_MARK)]
    cm.panel_key(ax_e, marks, ["best", f"last-{LAST_MEAN} mean", "final"], handlelength=1, spacing=1.2,
                 textpad=0.5)
    cm.finish_dot_figure(fig, (ax_t, ax_b, ax_e), OUTCOMES_TITLE, families + [gap] + seeds + [gap, no_grok], labels)
    return cm.save(fig, out), write_outcome_notes(root, runs, str(Path(out).with_suffix(".md")))


def write_outcome_notes(root, runs, out):
    """write the outcomes figure's details and per-run numbers, and the instability numbers for the runs that
    grokked, for writing the appendix.

    args:
        root: folder holding run directories.
        runs: rows from outcome_runs.
        out: output path.
    returns:
        the output path.
    """
    def find(config, seed=41):
        return next(r for r in runs if r["config"] == config and r["seed"] == seed and "rerun" not in r["label"])

    def grok(r):
        return r["verdict"]["t_grok"]

    no_norm = [r for r in runs if r["group"] == OUTCOME_GROUPS[0][0] and r["family"] != "one"]
    one_layer = [r for r in runs if r["family"] == "one"]
    rerun = next(r for r in runs if "rerun" in r["label"])
    stack8 = [find("t5_unshared_sandwich_const", s) for s in cm.SEEDS]
    wd05 = find("t2_huginn_wd05")
    not_needed = [find(c) for c in ("t1_huginn_embedstart", "t1_huginn_fixedT8", "t2_huginn_fullbackprop")]
    lines = [f"# {OUTCOMES_TITLE}: notes", "",
             "Generated by figures/ablations.py from results/runs/. Not part of the figure.", "",
             "## What the figure shows", "",
             f"- The {len(runs)} runs listed in figures/ablations.py, all trained on Nanda's recipe for the full 20k steps (constant lr 1e-3, "
             "weight decay λ = 1 unless stated, full-batch, no clipping), one row each, grouped by what the ablation "
             "changes. Seed 41 filled, seed 42 hollow, with each seed-42 row under its seed-41 row. Colours are model "
             "families.",
             f"- Left (\"Validation Drops\"): each run from step 0 to 20k, thick where its mean val accuracy over 1-3 "
             f"hops was at or above {cm.GROK_ACC:g} and thin where it was below (each eval covers half the eval "
             f"spacing on either side). The dot is where it grokked: the first eval at or above {cm.GROK_ACC:g} "
             f"(src/verdicts.py). An × at 20k marks a run that never got there.",
             f"- Middle (\"Post-Grok Time Below val={cm.GROK_ACC:g}\"): the share of its evals after grokking below "
             f"{cm.GROK_ACC:g}; evals are evenly spaced, so it's the share of that time. Empty for runs that didn't "
             "grok; the right column's stars show how close they came.",
             "- Right (\"Validation Accuracy\"): three marks per run. The star is the highest mean val accuracy at "
             f"any eval up to 20k, the triangle the average over the evals in the last {LAST_MEAN} steps (after step "
             f"{cm.LAST_STEP - LAST_MEAN}), and the square the mean val accuracy at 20k. Where they overlap, the star is "
             "on top, then the triangle; values that close are effectively equal.",
             "- The Huginn-style rows in the last group change one thing from the reference (sandwich norms, prelude "
             "and coda, injection, noise start, random depth, truncated backprop, λ = 1).",
             "- GPUs: the seed-41 sweep runs ran on an RTX 5090, the seed-42 runs and the `t5_*` runs (stacks with "
             "sandwich norms, the sandwich loop's rerun) on an RTX 4090. Grokking time can differ by up to 2x across "
             "GPUs (gpu column below).",
             "- The 2×3 panels behind the seed-41 sweep rows (train and test losses, W_E's gini, weight norm) are "
             "drawn with figures/ablations.py --supporting.", "",
             "- Left out: the two-phase runs (`t6_*`), which leave Nanda's recipe at 14k. At seed 42 they match runs "
             "shown until then: " + "; ".join(f"`{tw}` grokked at {a}, `{base}` at {b}"
                                               for tw, base, a, b in collapse.twin_groks(root)) + ".", "",
             "## Chance levels", "",
             f"- Mean val accuracy: 1 / 113 = {cm.CHANCE_ACC:.3f}, the dotted line marked chance.", "",
             "## Per run", "",
             f"| group | run | config | gpu | grokked at | best val | mean val, last {LAST_MEAN} steps | val at 20k | "
             "train at 20k | stable after grokking |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in runs:
        v = r["verdict"]
        when = "—" if grok(r) is None else str(grok(r))
        stable = "—" if v["stable"] is None else f"{v['stable']:.2f}"
        lines.append(f"| {r['group']} | {r['label']} | `{r['config']}` | {r['device']} | {when} | {r['best']:.2f} | "
                     f"{r['last_mean']:.2f} | {v['val']:.2f} | {v['train']:.2f} | {stable} |")
    lines += ["", "\"stable after grokking\" is the share of evals in src/verdicts.py's window after grokking that "
              f"stay ≥ {cm.GROK_ACC:g}.", "",
              "## For the text: grokking", "",
              f"- Without normalization, none of the {len(no_norm)} models deeper than one layer grokked within 20k "
              f"(best val accuracy {min(r['best'] for r in no_norm):.2f}-{max(r['best'] for r in no_norm):.2f}), "
              "looped or not, with or without injection or a prelude and coda; these are seed 41 only. The one-layer "
              "model grokked at both seeds (" + ", ".join(f"{grok(r)}" for r in one_layer) + ").",
              "- With normalization the loops and stacks can grok, but not reliably: the 8-layer stack with sandwich "
              f"norms didn't grok at seed 41 (best {stack8[0]['best']:.2f}) and did at seed 42 ({grok(stack8[1])}); "
              f"the plain loop with sandwich norms, same config and seed, grokked at {grok(find('t1_ut_sandwich'))} "
              f"on the 5090 and {grok(rerun)} on the 4090; and the Huginn-style loop with only a state norm didn't "
              f"grok (best {find('t2_huginn_nosandwich')['best']:.2f}).",
              f"- Within 20k steps, the Huginn-style loop needed its injection, its prelude and coda, and weight decay "
              f"λ ≥ 1 to grok (λ = 0.5 reached {wd05['best']:.2f} at best and was still rising, "
              f"{wd05['verdict']['val']:.2f} at 20k); its noise start, random depth and truncated backprop weren't "
              "needed (grokked at " + ", ".join(f"{grok(r)}" for r in not_needed) + "; fixed depth also at seed 42, "
              f"{grok(find('t1_huginn_fixedT8', 42))}).",
              "- Claims hold under Nanda's recipe only; no hyperparameter sweep was done. Most ablations are one seed.",
              "", "# After grokking: the instability", "",
              "The runs above that grokked, with Nanda's one-layer model as the control. The same numbers as "
              "figures/collapse.py --supporting's instability figure, which shows only these runs.", ""]
    lines += collapse.instability_sections(root, collapse.grokked_runs(root))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("\n".join(lines) + "\n")
    return out


def load_curves(path):
    """read the six measures and the grokking step from one run, up to the judged step.

    args:
        path: run directory.
    returns:
        dict with eval steps, sweep steps, one array per measure, and t_grok.
    """
    rows, evals, sweeps = cm.load_rows(path)
    sys3 = [np.mean([g["loss"] for k, g in r["at_eval_depth"]["systematic"].items() if k.startswith("3.")])
            for r in sweeps]
    return {
        "steps": np.array([r["step"] for r in evals]),
        "train": np.array([cm.mean_loss(r["train"]) for r in evals]),
        "gini": np.array([r["progress"]["gini_W_E"] for r in evals]),
        "wnorm": np.array([r["progress"]["sum_sq_weights"] for r in evals]),
        "grid_test": np.array([r["progress"]["grid_test_loss"] for r in evals]),
        "sweep_steps": np.array([r["step"] for r in sweeps]),
        "sys3": np.array(sys3),
        "hop6": np.array([r["at_eval_depth"]["extrap_depth"]["6"]["loss"] for r in sweeps]),
        "t_grok": cm.grok_step(rows),
    }


def plot_figure(title, runs, out, smooth="median"):
    """draw one six-panel figure.

    args:
        title: figure title.
        runs: list of (label, colour, dashed, curves).
        out: output path.
        smooth: "median" for rolling medians over faint raw curves, "none" for raw curves only.
    returns:
        path to the written figure.
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.4), facecolor=pp.SURFACE)
    for ax in axes[1, 1:]:
        ax.sharey(axes[1, 0])
    for ax, (key, xkey, panel, ylabel, kind) in zip(axes.flat, PANELS):
        pp.style_axis(ax, panel, ylabel, log_y=kind in ("loss", "log"))
        window = WINDOWS[xkey] if smooth == "median" else 1
        for label, colour, dashed, c in runs:
            cm.draw(ax, c[xkey], c[key], colour, label=label, dashed=dashed, window=window)
            cm.mark_step(ax, c["t_grok"], colour, hollow=dashed)
        if kind == "loss":
            cm.loss_axis(ax)
        ax.set_xlim(0, cm.LAST_STEP)
    # loss ranges: the top-left panel alone, the bottom row together
    for panels in ([0], [3, 4, 5]):
        keys = [PANELS[i][:2] for i in panels]
        later = [c[k][c[x] >= LOSS_FROM] for k, x in keys for *_, c in runs]
        axes.flat[panels[0]].set_ylim(*cm.log_limits(later))
    for i, (*_, kind) in enumerate(PANELS):
        if kind == "loss":
            cm.tidy_log_ticks(axes.flat[i])
    cm.legend(fig, axes[0, 0], len(runs), y=0.935)
    note = "▼ grokking step (mean val accuracy ≥ 0.9; hollow for dashed runs). dotted: ln 113, the loss of a uniform guess."
    if smooth == "median":
        note += " lines: rolling median over 500 steps (3 sweeps for 3- and 6-hop); faint: raw."
    fig.suptitle(title, x=0.01, y=0.99, ha="left", fontsize=14, color=pp.INK)
    fig.text(0.01, 0.955, note, fontsize=9, color=pp.INK_2, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.89 if len(runs) > 3 else 0.91))
    return cm.save(fig, out)


def main(argv=None):
    """run the cli.

    args:
        argv: argument list, or None for sys.argv.
    returns:
        exit code: 0 ok.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(cm.RESULTS / "runs"))
    ap.add_argument("--seed", type=int, default=41, help="seed of the --supporting figures")
    ap.add_argument("--out", default=str(cm.OUT / "dynamics"))
    ap.add_argument("--smooth", choices=["median", "none"], default="median")
    ap.add_argument("--supporting", action="store_true",
                    help="also draw the four 2×3 ablation figures, which aren't in the write-up, into <out>/supporting/")
    args = ap.parse_args(argv)
    print(*plot_outcomes(args.root, f"{args.out}/grokking_outcomes.png"), sep="\n")
    if not args.supporting:
        return 0
    cache = {}
    for name, title, spec in FIGURES:
        runs = []
        for config, label, colour, dashed in spec:
            if config not in cache:
                cache[config] = load_curves(cm.run_dir(f"{config}_p113_h3_x6_s{args.seed}", args.root))
            runs.append((label, colour, dashed, cache[config]))
        print(plot_figure(title, runs, f"{args.out}/supporting/{name}.png", args.smooth))
    return 0


if __name__ == "__main__":
    sys.exit(main())
