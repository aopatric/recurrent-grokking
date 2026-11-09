"""draw the write-up's fixes figure, and with --supporting the instability, collapse and two-phase figures. the
instability numbers also go into the outcomes figure's notes (figures/ablations.py).

    uv run python figures/collapse.py [--root results/runs] [--out figures/out/dynamics] [--supporting]

fixes.png and .md: from one grokked huginn-style checkpoint, 4000-step branches that change only the optimizer (high-
loss steps and val accuracy over the last 1000 steps), and from scratch, three runs split at step 14000 into nanda's
recipe and the two-phase recipe (clipping and lr decay), each pair identical until the switch: the plain loop at seeds
42 and 41 and the huginn-style reference at seed 42.

the supporting figures (<out>/supporting/): instability, every run that grokked on nanda's recipe after it grokked
(the outcomes figure shows the same runs among all the others); collapse, huginn's reference run and five weight decay
branches over time; two_phase, the two sandwich-loop runs' val accuracy and loss at every step.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D

import common as cm

pp = cm.pp
plt = cm.plt

SOURCE = "t1_huginn_ref_p113_h3_x6_s41"
BRANCH_STEP = 8944
BRANCH_LEN = 4000
# each arm: (arm name, label, colour, dashed), ordered by weight decay
ARMS = [
    ("wd0", "λ = 0", cm.RAMP[0], False),
    ("wd05", "λ = 0.5", cm.RAMP[1], False),
    ("control", "λ = 1 (unchanged)", cm.RAMP[2], False),
    ("wd2", "λ = 2", cm.RAMP[3], False),
    ("wd2_lr05", "λ = 2, half lr", cm.RAMP[3], True),
]
# a training step counts as high-loss above this cross-entropy
HIGH_LOSS = 1.0
BIN = 500
HIGH_LOSS_TITLE = f"share of steps with loss > {HIGH_LOSS:g}, per {BIN} steps"
HIGH_LOSS_LABEL = "share of steps"
RECIPES = [
    ("t5_ut_sandwich_const_p113_h3_x6_s41", "Nanda's recipe (constant lr)", pp.SERIES[0]),
    ("t6_ut_sandwich_twophase_p113_h3_x6_s41", "two-phase: clipping and lr decay from 14k", pp.SERIES[1]),
]
SWITCH = 14000


def val_curve(path):
    """eval steps and mean val accuracy over 1-3 hops, with the grokking step.

    args:
        path: run directory.
    returns:
        (steps, accuracy, t_grok).
    """
    rows, evals, _ = cm.load_rows(path, last=10 ** 9)
    steps = np.array([r["step"] for r in evals])
    return steps, np.array([cm.mean_acc(r["val"]) for r in evals]), cm.grok_step(rows)


def high_loss_counts(steps, loss, length, width=BIN):
    """number of high-loss training steps in consecutive bins.

    args:
        steps: step numbers (starting at 1).
        loss: per-step training loss.
        length: total steps covered.
        width: bin width in steps.
    returns:
        (bin centres, counts).
    """
    edges = np.arange(0, length + width, width) + 0.5
    counts, _ = np.histogram(steps[loss > HIGH_LOSS], bins=edges)
    return edges[:-1] + width / 2, counts


def step_loss_panel(ax, title):
    """style a per-step training loss panel.

    args:
        ax: matplotlib axis.
        title: panel title.
    """
    pp.style_axis(ax, title, "cross-entropy", log_y=True)
    cm.loss_axis(ax)
    cm.tidy_log_ticks(ax)


def plot_collapse(root, out):
    """draw the four-panel collapse figure.

    args:
        root: folder holding run directories.
        out: output path.
    returns:
        path to the written figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.4), facecolor=pp.SURFACE)
    src = cm.run_dir(SOURCE, root)
    steps, acc, t_grok = val_curve(src)
    keep = steps <= cm.LAST_STEP
    ax = axes[0, 0]
    pp.style_axis(ax, "Huginn on Nanda's recipe: val accuracy", "mean val accuracy, 1–3 hops")
    ax.plot(steps[keep], acc[keep], color=pp.SERIES[0], linewidth=1.0)
    ax = axes[0, 1]
    pp.style_axis(ax, f"Huginn on Nanda's recipe: {HIGH_LOSS_TITLE}", HIGH_LOSS_LABEL)
    s = cm.load_steps(src)
    x, n = high_loss_counts(s["step"], s["loss"], cm.LAST_STEP)
    later = x > BIN  # the first bin is the initial descent from an untrained model, not instability
    ax.plot(x[later], n[later] / BIN, color=pp.SERIES[0], linewidth=1.6, marker="o", markersize=3)
    for ax in axes[0]:
        cm.mark_step(ax, t_grok, pp.SERIES[0])
        ax.axvline(BRANCH_STEP, color=pp.INK_2, linewidth=0.9, linestyle=cm.DASHED)
        ax.set_xlim(0, cm.LAST_STEP)
    axes[0, 0].text(BRANCH_STEP + 200, 0.04, "branch point", color=pp.INK_2, fontsize=8.5)

    ax_acc, ax_hl = axes[1]
    pp.style_axis(ax_acc, f"branches from step {BRANCH_STEP}: val accuracy", "mean val accuracy, 1–3 hops")
    pp.style_axis(ax_hl, f"branches: {HIGH_LOSS_TITLE}", HIGH_LOSS_LABEL)
    for arm, label, colour, dashed in ARMS:
        d = cm.run_dir(f"{SOURCE}_from{BRANCH_STEP}_{arm}", root)
        b_steps, b_acc, _ = val_curve(d)
        style = cm.DASHED if dashed else "-"
        cm.draw(ax_acc, b_steps, b_acc, colour, label=label, dashed=dashed, window=5)
        bs = cm.load_steps(d)
        x, n = high_loss_counts(bs["step"], bs["loss"], BRANCH_LEN)
        ax_hl.plot(x, n / BIN, color=colour, linewidth=1.6, linestyle=style, marker="o", markersize=3.5)
    for ax in axes[1]:
        ax.set_xlabel("steps since the branch point", fontsize=9, color=pp.INK_2)
        ax.set_xlim(0, BRANCH_LEN)
    for ax in (axes[0, 1], ax_hl):
        ax.set_ylim(0, 1)
    cm.legend(fig, ax_acc, len(ARMS) + 1, y=0.905)
    fig.suptitle("On Nanda's recipe, Huginn is unstable from early in training, and weight decay sets the rate",
                 x=0.01, y=0.99, ha="left", fontsize=14, color=pp.INK)
    fig.text(0.01, 0.958, "▼ grokking step (mean val accuracy ≥ 0.9). training is full-batch, so a high-loss step is "
             f"one where the whole training set's loss exceeds {HIGH_LOSS:g}; the source run's first 500 steps (the initial "
             "descent) are left out.\nbranches change only the optimizer "
             "and reuse the source run's depth draws; their accuracy is a rolling median over 500 steps (faint: raw).",
             fontsize=9, color=pp.INK_2, ha="left", va="top", linespacing=1.5)
    fig.tight_layout(rect=(0, 0, 1, 0.85))
    return cm.save(fig, out)


def plot_two_phase(root, out):
    """draw the two-panel two-phase vs constant figure.

    args:
        root: folder holding run directories.
        out: output path.
    returns:
        path to the written figure.
    """
    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(13, 4.8), facecolor=pp.SURFACE)
    pp.style_axis(ax_acc, "val accuracy", "mean val accuracy, 1–3 hops")
    for name, label, colour in RECIPES:
        d = cm.run_dir(name, root)
        steps, acc, t_grok = val_curve(d)
        keep = steps <= cm.LAST_STEP
        ax_acc.plot(steps[keep], acc[keep], color=colour, linewidth=1.0, label=label)
        cm.mark_step(ax_acc, t_grok, colour)
        s = cm.load_steps(d)
        keep_s = s["step"] <= cm.LAST_STEP
        ax_loss.plot(s["step"][keep_s], s["loss"][keep_s], color=colour, linewidth=0.4)
    step_loss_panel(ax_loss, "training loss at every step")
    for ax in (ax_acc, ax_loss):
        ax.axvline(SWITCH, color=pp.INK_2, linewidth=0.9, linestyle=cm.DASHED)
        ax.set_xlim(0, cm.LAST_STEP)
    ax_acc.text(SWITCH - 200, 0.5, "recipes split", color=pp.INK_2, fontsize=8.5, ha="right")
    cm.legend(fig, ax_acc, 2, y=0.9)
    fig.suptitle("Sandwich loop: Nanda's recipe vs the two-phase recipe", x=0.01, y=0.99, ha="left",
                 fontsize=14, color=pp.INK)
    fig.text(0.01, 0.925, "▼ grokking step (mean val accuracy ≥ 0.9). dotted: ln 113. the two runs are identical "
             "until step 14000.", fontsize=9, color=pp.INK_2, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.8))
    return cm.save(fig, out)


SEEDS = cm.SEEDS
# the runs with a step log behind the instability notes' before-grokking table: the plain loop and the unshared
# stacks on nanda's recipe, all with sandwich norms, (config, label, colour, marker)
STACKS = [
    ("t5_ut_sandwich_const", "plain loop, 8 passes", pp.SERIES[2], "o"),
    ("t5_unshared_sandwich_const", "8 distinct layers", pp.SERIES[3], "s"),
    ("t5_unshared_sandwich_L2_const", "2 distinct layers", "#9b6ad6", "^"),
]
# high-loss shares are drawn from this step on: every run's loss is first below 1 by step 812, so later high-loss
# steps are instability, not the initial descent
COUNT_FROM = 1000
# the early window the notes quote high-loss shares over: (start, end) step
EARLY = (500, 3500)
INSTABILITY_TITLE = "Instability Overview"
# the instability figure's rows: every run in results/ that grokked on nanda's recipe, grouped, with nanda's one-layer
# model as the control: (group, colour, [(config, seed, label[, suffix])]); seed 41 filled, seed 42 hollow. the plain
# loop with sandwich norms has two seed-41 runs: the ablation sweep's (5090) and its rerun in the t5 suite (4090)
GROKKED_GROUPS = [
    ("Huginn-Style Loop", pp.SERIES[0], [
        ("t1_huginn_ref", 41, "reference"), ("t1_huginn_ref", 42, "reference"),
        ("t1_huginn_fixedT8", 41, "fixed depth 8"), ("t1_huginn_fixedT8", 42, "fixed depth 8"),
        ("t1_huginn_embedstart", 41, "embedding start"), ("t2_huginn_fullbackprop", 41, "full backprop"),
        ("t1_huginn_mindepth2", 41, "min depth 2"), ("t3_huginn_wd2", 41, "weight decay λ = 2")]),
    ("Plain Loop", pp.SERIES[2], [
        ("t2_ut_statenorm_ref", 41, "state norm"), ("t1_ut_sandwich", 41, "sandwich norms"),
        ("t5_ut_sandwich_const", 41, "sandwich norms", " (4090 rerun)"), ("t5_ut_sandwich_const", 42, "sandwich norms")]),
    ("Unshared Stack, Sandwich Norms", pp.SERIES[3], [
        ("t5_unshared_sandwich_const", 42, "8 layers"), ("t5_unshared_sandwich_L2_const", 41, "2 layers"),
        ("t5_unshared_sandwich_L2_const", 42, "2 layers")]),
    ("Control: Nanda's One-Layer Model", pp.INK_2, [
        ("t1_anchor_nope_ref", 41, "no normalization"), ("t1_anchor_nope_ref", 42, "no normalization")]),
]
CONTROL = GROKKED_GROUPS[-1][0]
# the notes' over-time table: bins of steps since grokking
SINCE_BIN = 500
# a run with fewer steps than this left after grokking is named in the notes, since its share covers little time
SHORT_AFTER = 4000
# the fixes figure's branch arms, top to bottom, with a gap before each group: (arm, label, starts a group)
FIX_ARMS = [
    ("control", "λ = 1, constant lr (unchanged)", False),
    ("wd0", "λ = 0", True),
    ("wd05", "λ = 0.5", False),
    ("wd2", "λ = 2", False),
    ("wd2_lr05", "λ = 2, half lr", False),
    ("nowd_embed", "no weight decay on the embedding", False),
    ("clip", "clipping at 1.0", True),
    ("lrdecay", "lr decay", False),
    ("clip_lrdecay", "clipping and lr decay", False),
]
# the fixes figure's from-scratch pairs, left to right: (panel title, label in the notes, two-phase run, constant-lr
# twin, two-phase colour). each pair ran on one gpu and its step logs are identical until SWITCH, so the twin is
# drawn throughout and the two-phase run from the switch on. seed 41's plain loop hadn't grokked by the switch; the
# asterisk in its title is for a caption in the post
FIX_PAIRS = [
    ("Plain Loop, Seed 42", "plain loop, seed 42", "t6_ut_sandwich_twophase_p113_h3_x6_s42",
     "t5_ut_sandwich_const_p113_h3_x6_s42", pp.SERIES[2]),
    ("Huginn-Style Loop, Seed 42", "Huginn-style loop, seed 42", "t6_huginn_twophase_p113_h3_x6_s42",
     "t1_huginn_ref_p113_h3_x6_s42", pp.SERIES[0]),
    ("Plain Loop, Seed 41, 4090 Rerun*", "plain loop, seed 41 (4090 rerun)", "t6_ut_sandwich_twophase_p113_h3_x6_s41",
     "t5_ut_sandwich_const_p113_h3_x6_s41", pp.SERIES[2]),
]
# the late window the notes quote the pairs' high-loss shares over: (start, end) step
LATE = (16000, 20000)
# a two-phase run counts as quiet from the first bin after the switch where every later bin's high-loss share is
# below this
QUIET = 0.02
# two-phase runs and their constant-lr twins for the fixes notes: (two-phase config, constant-lr config)
TWINS = [("t6_ut_sandwich_twophase", "t5_ut_sandwich_const"), ("t6_huginn_twophase", "t1_huginn_ref"),
         ("t6_ut_statenorm_twophase", None)]
FIXES_TITLE = "Addressing Training Instability"
LAST_K = 1000


def high_loss_share(path, length=cm.LAST_STEP):
    """a run's share of high-loss steps per bin, from COUNT_FROM on.

    args:
        path: run directory.
        length: total steps covered.
    returns:
        (bin centres, shares).
    """
    s = cm.load_steps(path)
    keep = s["step"] <= length
    x, n = high_loss_counts(s["step"][keep], s["loss"][keep], length)
    later = x > COUNT_FROM
    return x[later], n[later] / BIN


def grok_stats(path):
    """how a run behaved after it grokked, judged up to cm.LAST_STEP.

    args:
        path: run directory.
    returns:
        dict with t_grok, the gpu it trained on, the eval steps and accuracies after grokking, the share of those
        evals below and well below the threshold, the longest stretch below it (steps), the stretches below it as
        [start, end] steps, the val accuracy at the end, and high-loss steps per 1000 after grokking (None without a
        step log).
    """
    v = cm.verdicts.verdict(cm.verdicts.load_rows(path))
    steps, acc = cm.val_series(path)
    after = steps >= v["t_grok"]
    # the longest run of consecutive evals below the threshold, times the eval spacing
    gap, longest, cur = int(np.median(np.diff(steps))), 0, 0
    for a in acc[after]:
        cur = cur + gap if a < cm.GROK_ACC else 0
        longest = max(longest, cur)
    per_k = None
    if (Path(path) / "steps.csv").exists():
        s = cm.load_steps(path)
        m = (s["step"] > v["t_grok"]) & (s["step"] <= cm.LAST_STEP)
        per_k = 1000 * float(np.mean(s["loss"][m] > HIGH_LOSS))
    spans = cm.below_spans(steps[after], acc[after], v["t_grok"], gap)
    device = json.loads((Path(path) / "meta.json").read_text()).get("device", "?").replace("NVIDIA GeForce ", "")
    return {"t_grok": v["t_grok"], "device": device, "since": steps[after] - v["t_grok"], "acc": acc[after],
            "below": float(np.mean(acc[after] < cm.GROK_ACC)), "well_below": float(np.mean(acc[after] < 0.5)),
            "longest": longest, "spans": spans, "end": v["val"], "per_k": per_k}


def grokked_runs(root):
    """every row of the instability figure with its after-grokking stats.

    args:
        root: folder holding run directories.
    returns:
        list of (group, colour, label, config, seed, stats) in GROKKED_GROUPS order.
    """
    return [(group, colour, f"{label}, seed {seed}{suffix[0] if suffix else ''}", config, seed,
             grok_stats(cm.run_dir(f"{config}_p113_h3_x6_s{seed}", root)))
            for group, colour, members in GROKKED_GROUPS for config, seed, label, *suffix in members]


def below_over_time(runs):
    """per bin of steps since grokking, the mean over runs of each run's share of evals below the threshold.

    args:
        runs: rows from grokked_runs.
    returns:
        (bin centres, mean shares, number of runs with data in each bin).
    """
    edges = np.arange(0, cm.LAST_STEP + SINCE_BIN, SINCE_BIN)
    shares = []
    for *_, st in runs:
        idx = np.digitize(st["since"], edges) - 1
        shares.append([np.mean(st["acc"][idx == i] < cm.GROK_ACC) if np.any(idx == i) else np.nan
                       for i in range(len(edges) - 1)])
    shares = np.array(shares)
    n = np.sum(~np.isnan(shares), axis=0)
    mean = np.array([np.nanmean(col) if k else np.nan for col, k in zip(shares.T, n)])
    return edges[:-1] + SINCE_BIN / 2, mean, n


def plot_instability(root, out):
    """draw the instability figure: one row per run that grokked on nanda's recipe, with nanda's one-layer model as
    the control. left, when it grokked and which of its evals after that were below the threshold; middle, the share
    of those evals below it; right, its accuracy at the end. details go to a markdown file next to it.

    args:
        root: folder holding run directories.
        out: output path of the image; the notes go next to it with a .md suffix.
    returns:
        (image path, notes path).
    """
    runs = grokked_runs(root)
    # wide enough to centre the panels with the row labels hanging off the left and the legend box on the right
    fig, (ax_t, ax_b, ax_e) = cm.dot_figure((15.5, 10), [1.35, 0.8, 0.8])
    ys, heads, bottom = cm.row_positions([group for group, *_ in runs])
    for (group, colour, label, config, seed, st), yy in zip(runs, ys):
        cm.timeline(ax_t, yy, colour, st["t_grok"], st["spans"])
        ax_t.plot([st["t_grok"]], [yy], **cm.seed_dot(colour, seed))
        ax_b.plot([st["below"]], [yy], **cm.seed_dot(colour, seed))
        ax_e.plot([st["end"]], [yy], **cm.seed_dot(colour, seed))
    cm.label_rows(ax_t, [label for _, _, label, *_ in runs], ys, heads, bottom)
    cm.style_dot_panels((ax_t, ax_b, ax_e), [
        ("Validation Drops", "step", (0, cm.LAST_STEP)),
        (f"Post-Grok Time Below val={cm.GROK_ACC:g}", "share of evals after grokking", (0, 1)),
        ("Final Validation Accuracy", "mean val accuracy, 1–3 hops", (0, 1))])
    cm.accuracy_marks(ax_e, bottom)
    families = [Line2D([], [], **cm.seed_dot(c, SEEDS[0])) for _, c, _ in GROKKED_GROUPS]
    seeds = [Line2D([], [], **cm.seed_dot(pp.INK, s)) for s in SEEDS]
    gap = Line2D([], [], linestyle="none")
    # long family names wrap after their first comma or colon so the box fits beside the panels
    labels = ([g.replace(", ", ",\n", 1).replace(": ", ":\n", 1) for g, _, _ in GROKKED_GROUPS] + [""]
              + [f"{fill}: seed {s}" for fill, s in zip(("filled", "hollow"), SEEDS)])
    cm.line_key(ax_t)
    cm.finish_dot_figure(fig, (ax_t, ax_b, ax_e), INSTABILITY_TITLE, families + [gap] + seeds, labels)
    notes = write_instability_notes(root, runs, str(Path(out).with_suffix(".md")))
    return cm.save(fig, out), notes


def twin_groks(root):
    """two-phase runs at seed 42 next to the nanda's-recipe runs they match until the switch, with both grokking steps.

    args:
        root: folder holding run directories.
    returns:
        list of (two-phase config, nanda's-recipe config, two-phase grok step, nanda's-recipe grok step).
    """
    return [(tw, base, cm.grok_step(cm.verdicts.load_rows(cm.run_dir(f"{tw}_p113_h3_x6_s{SEEDS[1]}", root))),
             cm.grok_step(cm.verdicts.load_rows(cm.run_dir(f"{base}_p113_h3_x6_s{SEEDS[1]}", root))))
            for tw, base in TWINS if base]


def write_instability_notes(root, runs, out):
    """write the instability figure's details and numbers.

    args:
        root: folder holding run directories.
        runs: rows from grokked_runs.
        out: output path.
    returns:
        the output path.
    """
    short = [r for r in runs if cm.LAST_STEP - r[-1]["t_grok"] < SHORT_AFTER]
    s41 = [r for r in runs if r[3] in ("t1_ut_sandwich", "t5_ut_sandwich_const") and r[4] == SEEDS[0]]
    lines = [f"# {INSTABILITY_TITLE}: notes", "",
             "Generated by figures/collapse.py from results/runs/. Not part of the figure.", "",
             "## What the figure shows", "",
             f"- All {len(runs)} runs in results/ that grokked on Nanda's recipe (constant lr 1e-3, λ = 1 unless "
             "stated, no clipping, 20k full-batch steps), one row each, grouped by family, with Nanda's one-layer "
             "model as the control. Seed 41 filled, seed 42 hollow.",
             f"- Left (\"Validation Drops\"): when each run grokked (the dot; first eval with mean val accuracy ≥ {cm.GROK_ACC:g}), then a "
             f"line to 20k, thick where its evals were at or above {cm.GROK_ACC:g} and thin where they were below it "
             "(each eval covers half the eval spacing on either side). The key for the two line widths sits under that panel's "
             "title.",
             f"- Middle (\"Post-Grok Time Below val={cm.GROK_ACC:g}\"): the share of its evals after grokking below {cm.GROK_ACC:g}; evals are "
             "evenly spaced, so it's the share of that time. Runs with less than "
             f"{SHORT_AFTER // 1000}k steps left after grokking, whose share covers little time (no note in the "
             "figure; the left panel shows it): " + "; ".join(f"{r[2]}, {len(r[-1]['acc'])} evals over {cm.LAST_STEP - r[-1]['t_grok']} steps"
                                  for r in short) + ".",
             "- Right (\"Final Validation Accuracy\"): val accuracy at 20k, with chance and the grokking threshold.",
             "- Left out: runs that didn't grok, among them the 8-layer stack at seed 41 (the sweep's are in the "
             "outcomes figure), and the two-phase runs (`t6_*`), which leave "
             "Nanda's recipe at 14k. At seed 42 those match runs shown until then: "
             + "; ".join(f"`{tw}` grokked at {a}, `{base}` at {b}" for tw, base, a, b in twin_groks(root)) + ".",
             "- GPUs differ within some seed pairs (gpu column below): the seed-41 rows from the ablation sweep ran on "
             "an RTX 5090, the seed-42 rows, the unshared stacks and the rerun on an RTX 4090. Grokking time can "
             "differ by up to 2x across GPUs: the same config and seed for the plain loop with sandwich norms grokked "
             + " and ".join(f"at {r[-1]['t_grok']} on the {r[-1]['device']}" for r in s41) + ".", "",
             "## Chance levels", "",
             f"- Val accuracy: 1 / 113 = {cm.CHANCE_ACC:.3f}, the dotted line marked chance.", ""]
    lines += instability_sections(root, runs)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("\n".join(lines) + "\n")
    return out


def instability_sections(root, runs):
    """the instability numbers for the notes: per run after grokking, over time, before grokking, and the points
    for the text.

    args:
        root: folder holding run directories.
        runs: rows from grokked_runs.
    returns:
        list of markdown lines.
    """
    normalized = [r for r in runs if r[0] != CONTROL]
    below = [st["below"] for *_, st in normalized]
    broken = [r for r in normalized if r[-1]["end"] < cm.GROK_ACC]
    short = [r for r in runs if cm.LAST_STEP - r[-1]["t_grok"] < SHORT_AFTER]
    lowest = min(normalized, key=lambda r: r[-1]["below"])
    ref = {r[4]: r[-1] for r in runs if r[3] == "t1_huginn_ref"}
    lines = ["## After grokking, per run", "",
             f"| family | run | config | gpu | grokked at | steps after | evals below {cm.GROK_ACC:g} | below 0.5 | "
             f"longest stretch below {cm.GROK_ACC:g} | val at 20k | steps with loss > 1 per 1000 after grokking |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for group, _, label, config, seed, st in runs:
        per_k = "—" if st["per_k"] is None else f"{st['per_k']:.0f}"
        lines.append(f"| {group} | {label} | `{config}` | {st['device']} | {st['t_grok']} | "
                     f"{cm.LAST_STEP - st['t_grok']} | "
                     f"{st['below']:.2f} | {st['well_below']:.2f} | {st['longest']} | {st['end']:.2f} | {per_k} |")
    x, mean, n = below_over_time(normalized)
    lines += ["", "## After grokking, over time (normalized runs)", "",
              "| steps since grokking | runs with data | mean share below |", "|---|---|---|"]
    lines += [f"| {int(c - SINCE_BIN / 2)}-{int(c + SINCE_BIN / 2)} | {k} | {m:.2f} |"
              for c, m, k in zip(x, mean, n) if k]
    # the rise over fixed sets of runs: those with 8k and 12k steps of data after grokking, comparing the first and
    # last 2k of that window
    def rise(horizon):
        fixed = [r for r in normalized if cm.LAST_STEP - r[-1]["t_grok"] >= horizon]
        xf, mf, _ = below_over_time(fixed)
        return len(fixed), float(np.mean(mf[xf < 2000])), float(np.mean(mf[(xf > horizon - 2000) & (xf < horizon)]))
    rises = [(h, *rise(h)) for h in (8000, 12000)]
    lines += ["", "## Before grokking (runs with a step log)", "",
              f"| run | seed | first step with loss < 1 | share of steps with loss > 1, {EARLY[0]}-{EARLY[1]} | "
              f"share from {COUNT_FROM} to grokking |", "|---|---|---|---|---|"]
    logged = [(c, label, s) for c, label, _, _ in STACKS for s in SEEDS] + [("t1_huginn_ref", "Huginn-style loop, "
                                                                            "reference", 41)]
    for config, label, seed in logged:
        d = cm.run_dir(f"{config}_p113_h3_x6_s{seed}", root)
        s = cm.load_steps(d)
        step, loss = s["step"], s["loss"]
        v = cm.verdicts.verdict(cm.verdicts.load_rows(d))
        early = (step > EARLY[0]) & (step <= EARLY[1])
        pre = "—" if v["t_grok"] is None else \
            f"{np.mean(loss[(step > COUNT_FROM) & (step < v['t_grok'])] > HIGH_LOSS):.2f}"
        lines.append(f"| {label} | {seed} | {int(step[loss < HIGH_LOSS][0])} | "
                     f"{np.mean(loss[early] > HIGH_LOSS):.2f} | {pre} |")
    lines += ["", "No step log is committed for the other runs or the control. The evals' train loss isn't a "
              "substitute before grokking: it's above 1 while a run is still fitting, and that isn't instability.", "",
              "## For the text: instability", "",
              f"- On Nanda's recipe, every normalized model that grokked lost accuracy again afterwards: "
              f"{len(normalized)} runs across the Huginn-style loop, the plain loop and unshared stacks, at two seeds, "
              f"spent {min(below):.0%}-{max(below):.0%} of their evals after grokking below {cm.GROK_ACC:g}, and "
              f"{len(broken)} of {len(normalized)} ended below it at 20k. Nanda's one-layer model, the reproduction, "
              f"didn't. The lowest share, {lowest[-1]['below']:.0%} ({lowest[2]}), covers only "
              f"{cm.LAST_STEP - lowest[-1]['t_grok']} steps. Runs with less than {SHORT_AFTER // 1000}k steps left "
              "after grokking, whose share covers little time: "
              + "; ".join(f"{r[2]}, {len(r[-1]['acc'])} evals over {cm.LAST_STEP - r[-1]['t_grok']} steps" for r in short)
              + ".",
              f"- It doesn't settle; it gets worse, for the same runs throughout. Mean share of evals below "
              f"{cm.GROK_ACC:g}, first 2k steps after grokking vs the last 2k of the window: "
              + "; ".join(f"the {k} runs with {h // 1000}k steps of data, {a:.0%} to {b:.0%}" for h, k, a, b in rises)
              + ". Bins are noisy (the over-time table).",
              f"- Brief dips and long losses both happen. The Huginn-style reference at seed {SEEDS[1]} dipped "
              f"briefly and often ({ref[SEEDS[1]]['below']:.0%} of evals below {cm.GROK_ACC:g}, never for more than "
              f"{ref[SEEDS[1]]['longest']} steps) and ended at {ref[SEEDS[1]]['end']:.2f}; at seed {SEEDS[0]} it "
              f"stayed below for {ref[SEEDS[0]]['longest']} steps and ended at {ref[SEEDS[0]]['end']:.2f}.",
              "- Where step logs exist, the loss spikes start well before grokking (before-grokking table).",
              "- It isn't specific to loops: unshared stacks do it too, and the 8-layer stack, matched to the 8-pass "
              "loop, is the worst of them.",
              "- Not separated: normalization and depth. The unnormalized deeper models never grokked, so there's no "
              "stable deep comparison. Claims hold under Nanda's recipe only; no hyperparameter sweep was done."]
    return lines


def branch_stats(root):
    """per fixes-figure arm: total high-loss steps over the branch, and val accuracy over its last LAST_K steps.

    args:
        root: folder holding run directories.
    returns:
        list of (arm, label, starts a group, high-loss total, high-loss in the last LAST_K, mean val, min val,
        final val).
    """
    out = []
    for arm, label, group in FIX_ARMS:
        d = cm.run_dir(f"{SOURCE}_from{BRANCH_STEP}_{arm}", root)
        s = cm.load_steps(d)
        _, n = high_loss_counts(s["step"], s["loss"], BRANCH_LEN)
        late = int(np.sum((s["step"] > BRANCH_LEN - LAST_K) & (s["loss"] > HIGH_LOSS)))
        steps, acc = cm.val_series(d, last=BRANCH_LEN)
        tail = acc[steps > BRANCH_LEN - LAST_K]
        out.append((arm, label, group, int(n.sum()), late, float(tail.mean()), float(tail.min()), float(acc[-1])))
    return out


def pair_stats(root):
    """per fixes-figure pair and recipe: how the run did after the switch.

    args:
        root: folder holding run directories.
    returns:
        list of (pair label, recipe, row) with recipe "two-phase" or "constant lr" and row a dict of t_grok, val at
        the switch, high-loss share from the switch to LATE[0] and over LATE, mean and min val over the last LAST_K
        steps, min val after the switch, final val, the first step from which every BIN-step bin's high-loss
        share is below QUIET, and the lr at that step.
    """
    out = []
    for _, label, two_phase, twin, _ in FIX_PAIRS:
        for recipe, name in (("constant lr", twin), ("two-phase", two_phase)):
            d = cm.run_dir(name, root)
            steps, acc = cm.val_series(d)
            s = cm.load_steps(d)

            def share(lo, hi):
                keep = (s["step"] > lo) & (s["step"] <= hi)
                return float(np.mean(s["loss"][keep] > HIGH_LOSS))
            tail = acc[steps > cm.LAST_STEP - LAST_K]
            starts = list(range(SWITCH, cm.LAST_STEP, BIN))
            shares = [share(b, b + BIN) for b in starts]
            quiet = next((b for i, b in enumerate(starts) if max(shares[i:]) < QUIET), None)
            _, evals, _ = cm.load_rows(d, last=10 ** 9)
            lr = next((r["lr"] for r in evals if quiet is not None and r["step"] >= quiet), None)
            out.append((label, recipe, {"quiet": quiet, "lr": lr,
                "t_grok": cm.grok_step(cm.verdicts.load_rows(d)), "at_switch": float(acc[steps <= SWITCH][-1]),
                "early": share(SWITCH, LATE[0]), "late": share(*LATE), "mean": float(tail.mean()),
                "min": float(tail.min()), "min_after": float(acc[steps > SWITCH].min()), "final": float(acc[-1])}))
    return out


def plot_fixes(root, out):
    """draw the fixes figure: from a grokked checkpoint, each optimizer change's high-loss steps and val accuracy;
    from scratch, the sandwich loop on nanda's recipe and on the two-phase recipe; details go to a markdown file.

    args:
        root: folder holding run directories.
        out: output path of the image; the notes go next to it with a .md suffix.
    returns:
        (image path, notes path).
    """
    stats = branch_stats(root)
    fig = plt.figure(figsize=(13.5, 13.5), facecolor=pp.SURFACE, layout="constrained")
    fig.get_layout_engine().set(rect=(0.02, 0.02, 0.96, 0.93), w_pad=0.15, h_pad=0.12)
    top, bottom = fig.subfigures(2, 1, height_ratios=[1, 1.3], hspace=0.06)
    for sub in (top, bottom):
        sub.set_facecolor(pp.SURFACE)

    ax_hl, ax_acc = top.subplots(1, 2, sharey=True, gridspec_kw={"width_ratios": [1.15, 1]})
    ys, y = [], 0.0
    for _, _, group, *_ in stats:
        y -= 0.6 if group else 0
        ys.append(y)
        y -= 1
    for (arm, label, _, total, _, mean, low, _), yy in zip(stats, ys):
        colour = pp.INK_2 if arm == "control" else pp.SERIES[0]
        ax_hl.barh(yy, total / BRANCH_LEN, height=0.62, color=colour)
        ax_hl.text(total / BRANCH_LEN + 0.008, yy, f"{total / BRANCH_LEN:.3f}", va="center", fontsize=cm.NOTE_SIZE,
                   color=pp.INK_2)
        ax_acc.plot([low, mean], [yy, yy], color=colour, linewidth=1.6)
        ax_acc.plot([mean], [yy], marker="o", markersize=7, color=colour, linestyle="none")
        ax_acc.plot([low], [yy], marker="o", markersize=7, markerfacecolor=pp.SURFACE, markeredgecolor=colour,
                    markeredgewidth=1.4, linestyle="none")
    ax_hl.set_yticks(ys)
    ax_hl.set_yticklabels([label for _, label, *_ in stats])
    ax_hl.set_ylim(y + 0.4, 0.6)
    for ax, title, xlabel in [(ax_hl, f"Ratio of High-Loss (>{HIGH_LOSS:g}) Steps",
                               f"proportion of the {BRANCH_LEN} branch steps"),
                              (ax_acc, f"Last-{LAST_K} Step Accuracy", "mean val accuracy, 1–3 hops")]:
        pp.style_axis(ax, "", "")
        # pad leaves room for the accuracy panel's key under its title, as in the outcomes figure
        ax.set_title(title, loc="center", fontsize=cm.PANEL_SIZE, color=pp.INK, pad=34)
        ax.set_xlabel(xlabel, color=pp.INK_2)
        ax.grid(axis="y", visible=False)
        cm.text_sizes(ax)
    ax_hl.tick_params(axis="y", labelsize=cm.LABEL_SIZE, labelcolor=pp.INK)
    ax_hl.set_xlim(0, max(s[3] for s in stats) / BRANCH_LEN * 1.12)
    ax_acc.set_xlim(-0.03, 1.03)
    ax_acc.axvline(cm.CHANCE_ACC, color=pp.INK_2, linewidth=0.9, linestyle=(0, (1, 2)))
    ax_acc.text(cm.CHANCE_ACC + 0.015, y + 0.55, "chance", fontsize=cm.NOTE_SIZE, color=pp.INK_2, va="bottom")
    marks = [Line2D([], [], marker="o", markersize=7, color=pp.INK_2, linestyle="none"),
             Line2D([], [], marker="o", markersize=7, markerfacecolor=pp.SURFACE, markeredgecolor=pp.INK_2,
                    markeredgewidth=1.4, linestyle="none")]
    cm.panel_key(ax_acc, marks, ["mean", "min"], handlelength=1, spacing=1.6, textpad=0.5)

    axes = bottom.subplots(2, len(FIX_PAIRS), sharex=True, sharey=True)
    for (title, _, two_phase, twin, colour), ax_v, ax_s in zip(FIX_PAIRS, axes[0], axes[1]):
        d_tp, d_c = cm.run_dir(two_phase, root), cm.run_dir(twin, root)
        steps, acc = cm.val_series(d_c)
        ax_v.plot(steps, acc, color=pp.INK_2, linewidth=1.1)
        steps, acc = cm.val_series(d_tp)
        ax_v.plot(steps[steps >= SWITCH], acc[steps >= SWITCH], color=colour, linewidth=1.3)
        g_c, g_tp = (cm.grok_step(cm.verdicts.load_rows(d)) for d in (d_c, d_tp))
        cm.mark_step(ax_v, g_c, pp.INK_2)
        if g_tp != g_c:
            cm.mark_step(ax_v, g_tp, colour)
        x, share = high_loss_share(d_c)
        ax_s.plot(x, share, color=pp.INK_2, linewidth=1.5, marker="o", markersize=3)
        x, share = high_loss_share(d_tp)
        after = x >= SWITCH - BIN / 2
        ax_s.plot(x[after], share[after], color=colour, linewidth=1.5, marker="o", markersize=3)
        ax_v.set_title(title, loc="center", fontsize=cm.PANEL_SIZE, color=pp.INK, pad=14)
    for ax in axes.flat:
        pp.style_axis(ax, "", "")
        ax.axvline(SWITCH, color=pp.INK_2, linewidth=0.9, linestyle=cm.DASHED)
        ax.set_xlim(0, cm.LAST_STEP)
        ax.set_ylim(0, 1.02)
        ax.set_xticks(range(0, cm.LAST_STEP + 1, 5000))
        cm.k_steps(ax)
        cm.text_sizes(ax)
    for ax in axes[0]:
        ax.set_xlabel("")
    for ax in axes[1]:
        ax.set_xlabel("step", color=pp.INK_2)
    for ax in axes[0]:
        ax.axhline(cm.CHANCE_ACC, color=pp.INK_2, linewidth=0.9, linestyle=(0, (1, 2)))
    axes[0, 0].set_ylabel("mean val accuracy, 1–3 hops", color=pp.INK_2)
    axes[1, 0].set_ylabel(f"share of steps with loss > {HIGH_LOSS:g},\nper {BIN} steps", color=pp.INK_2)
    axes[0, -1].text(cm.LAST_STEP - 300, cm.CHANCE_ACC + 0.02, "chance", fontsize=cm.NOTE_SIZE, color=pp.INK_2,
                     ha="right", va="bottom")
    axes[1, 0].text(SWITCH - 300, 0.8, "recipes split", fontsize=cm.NOTE_SIZE, color=pp.INK_2, ha="right")
    families = list(dict.fromkeys(c for *_, c in FIX_PAIRS))
    names = {pp.SERIES[2]: "plain loop", pp.SERIES[0]: "Huginn-style loop"}
    handles = [Line2D([], [], color=pp.INK_2, linewidth=1.8)]
    handles += [Line2D([], [], color=c, linewidth=1.8) for c in families]
    handles.append(Line2D([], [], color=pp.INK_2, marker="v", markersize=6, linestyle="none"))
    labels = ["Nanda's recipe throughout (constant lr)"] + [f"two-phase from {SWITCH // 1000}k, {names[c]}"
                                                           for c in families] + ["grokked"]
    # the row's heading is the legend's title, so the legend sits under it and above the panels
    leg = bottom.legend(handles=handles, labels=labels, loc="outside upper center", ncol=len(labels), frameon=False,
                        fontsize=cm.LEGEND_SIZE, labelcolor=pp.INK, handlelength=2.2, columnspacing=1.8,
                        title=f"From Scratch: One Run Until Step {SWITCH}, Then Nanda's Recipe or Two-Phase "
                              "(Clipping and LR Decay)", title_fontsize=cm.LABEL_SIZE + 0.5)
    leg.get_title().set_color(pp.INK)
    leg._legend_box.sep = 14
    fig.suptitle(FIXES_TITLE, x=0.02, y=0.985, ha="left", fontsize=cm.TITLE_SIZE, color=pp.INK)
    return cm.save(fig, out), write_fixes_notes(root, stats, str(Path(out).with_suffix(".md")))


def write_fixes_notes(root, stats, out):
    """write the fixes figure's details and numbers, including every two-phase run at both seeds.

    args:
        root: folder holding run directories.
        stats: rows from branch_stats.
        out: output path.
    returns:
        the output path.
    """
    lines = [f"# {FIXES_TITLE}: notes", "",
             "Generated by figures/collapse.py from results/runs/. Not part of the figure.", "",
             "## What the figure shows", "",
             f"- Top: {len(FIX_ARMS)} branches from the Huginn-style reference's step-{BRANCH_STEP} checkpoint (seed "
             f"41, grokked), each run for {BRANCH_LEN} steps with the source run's depth draws, changing only the "
             "optimizer. Left: the proportion of those steps with full-batch training loss > 1. Right: mean and min val "
             f"accuracy over the last {LAST_K} steps (10 evals, one every 100 steps). Grey: the unchanged control. "
             "The figure doesn't say where the branches start; the caption should.",
             "- Arms: λ is weight decay (Nanda's recipe uses 1); \"half lr\" keeps lr × λ equal to the control's; "
             "clipping is global gradient-norm clipping at 1.0; lr decay goes from 1e-3 to 1e-5 over the branch.",
             f"- Bottom: three runs from scratch on Nanda's recipe, each split at step {SWITCH} into a twin that "
             "keeps the recipe (grey) and one on the two-phase recipe (clipping at 1.0 and lr decay to 1% from the "
             "switch; coloured by model). Each pair ran on one RTX 4090 and its step logs are identical through step "
             f"{SWITCH}, so one grey line is drawn before the split. Both models use sandwich norms (the plain loop is "
             "`t5_ut_sandwich_const` / `t6_ut_sandwich_twophase`; the Huginn-style loop is the reference "
             "`t1_huginn_ref` / `t6_huginn_twophase`). Top row: mean val accuracy. Bottom row: share of steps with "
             f"full-batch training loss > 1, per {BIN} steps, from step {COUNT_FROM}. A grokked mark in the two-phase "
             "colour appears only where it grokked after the split.",
             "- The asterisk on \"Plain Loop, Seed 41, 4090 Rerun*\" is for a caption: that run hadn't grokked by "
             "the split, so it shows grokking during the decay, not holding a solution. Seed 42 is the test of "
             "holding: both models had grokked well before the split.",
             f"- \"4090 rerun\": the pair is the `t5` suite's rerun on the RTX 4090, which the outcomes figure lists "
             f"as \"(4090 rerun)\". The seed-41 sweep's run of the same config (`t1_ut_sandwich`, RTX 5090) grokked at "
             f"{cm.grok_step(cm.verdicts.load_rows(cm.run_dir('t1_ut_sandwich_p113_h3_x6_s41', root)))}; grokking "
             "time can differ by up to 2× across GPUs.", "",
             "## Chance levels", "",
             f"- Val accuracy: 1 / 113 = {cm.CHANCE_ACC:.3f}, the dotted lines.", "",
             "## Branch arms", "",
             f"| arm | high-loss steps (all / last {LAST_K}) | proportion of all {BRANCH_LEN} | val, last {LAST_K} "
             "(mean / min) | final val |", "|---|---|---|---|---|"]
    for _, label, _, total, late, mean, low, final in stats:
        lines.append(f"| {label} | {total} / {late} | {total / BRANCH_LEN:.3f} | {mean:.3f} / {low:.3f} | "
                     f"{final:.3f} |")
    pairs = pair_stats(root)
    lines += ["", "## From scratch: the pairs", "",
              f"After the switch at step {SWITCH}. High-loss share: steps with training loss > {HIGH_LOSS:g}.", "",
              f"| pair | recipe after {SWITCH // 1000}k | grokked at | val at {SWITCH // 1000}k | high-loss share, "
              f"{SWITCH // 1000}k–{LATE[0] // 1000}k | high-loss share, {LATE[0] // 1000}k–{LATE[1] // 1000}k | "
              f"val, last {LAST_K} (mean / min) | val at 20k |", "|---|---|---|---|---|---|---|---|"]
    for label, recipe, r in pairs:
        lines.append(f"| {label} | {recipe} | {r['t_grok'] or '—'} | {r['at_switch']:.3f} | {r['early']:.3f} | "
                     f"{r['late']:.3f} | {r['mean']:.3f} / {r['min']:.3f} | {r['final']:.3f} |")
    row = {(label, recipe): r for label, recipe, r in pairs}
    p42c, p42t = row[("plain loop, seed 42", "constant lr")], row[("plain loop, seed 42", "two-phase")]
    h42c, h42t = row[("Huginn-style loop, seed 42", "constant lr")], row[("Huginn-style loop, seed 42", "two-phase")]
    p41c, p41t = (row[("plain loop, seed 41 (4090 rerun)", r)] for r in ("constant lr", "two-phase"))
    lines += ["", "## Every two-phase run, both seeds", "",
              f"Val over the last {LAST_K} steps of 20k. The constant-lr twin is the same config on Nanda's recipe "
              "(at seed 42 the twins are the same run until 14k; at seed 41 the Huginn-style pair ran on different "
              "GPUs).", "",
              f"| model | seed | grokked at | val at 20k | val, last {LAST_K} (mean / min) | constant-lr twin: val at "
              "20k |", "|---|---|---|---|---|---|"]
    for config, twin in TWINS:
        for seed in SEEDS:
            d = cm.run_dir(f"{config}_p113_h3_x6_s{seed}", root)
            v = cm.verdicts.verdict(cm.verdicts.load_rows(d))
            steps, acc = cm.val_series(d)
            tail = acc[steps > cm.LAST_STEP - LAST_K]
            t = cm.run_dir(f"{twin}_p113_h3_x6_s{seed}", root) if twin else None
            twin_val = f"{cm.verdicts.verdict(cm.verdicts.load_rows(t))['val']:.2f}" if t and t.exists() else "—"
            lines.append(f"| `{config}` | {seed} | {v['t_grok'] or '—'} | {v['val']:.2f} | {tail.mean():.3f} / "
                         f"{tail.min():.3f} | {twin_val} |")
    lines += ["", "## For the text", "",
              "- What the figure is for: the outcomes figure shows the instability; this one shows that it comes from "
              "the recipe and that a standard schedule removes it. That's why the mechanism section uses the two-phase "
              "checkpoints: they're stable solutions, not a snapshot of wherever the model was at step 20k.",
              "- From the same grokked checkpoint, weight decay sets the collapse rate, and it rises monotonically "
              "with λ; λ = 0 fades the collapses out. Halving lr at λ = 2 brings the rate back to the control's, but "
              "not the accuracy.",
              "- Clipping removes most high-loss steps but leaves accuracy around 0.9; lr decay holds accuracy; both "
              "together do both.",
              "- From scratch, the two-phase recipe ends at val ≥ 0.99 in every run that grokked (table above), "
              "across two seeds. The seed-42 state-norm loop didn't grok on it.",
              f"- From scratch, seed 42 is the clean test: both pairs had grokked (val {p42c['at_switch']:.3f} and "
              f"{h42c['at_switch']:.3f} at the split, plain loop and Huginn-style). Kept on Nanda's recipe, the plain loop fell apart (val "
              f"{p42c['final']:.2f} at 20k, loss > 1 in {p42c['late']:.0%} of steps over {LATE[0] // 1000}k–"
              f"{LATE[1] // 1000}k); on two-phase it held ({p42t['final']:.2f}; {p42t['late']:.0%}). It wasn't "
              f"immediate: within the first 1000 steps after the switch, while the linear decay still had the lr "
              f"above 8e-4, the two-phase plain loop dropped to val {p42t['min_after']:.2f}, with loss > 1 in "
              f"{p42t['early']:.0%} of steps over {SWITCH // 1000}k–{LATE[0] // 1000}k (twin: {p42c['early']:.0%}), "
              "then recovered.",
              f"- The seed-42 Huginn-style reference ends at {h42c['final']:.2f} on Nanda's recipe, but its last "
              f"{LAST_K} steps reach a min of {h42c['min']:.2f}, with loss > 1 in {h42c['late']:.0%} of late steps "
              f"(two-phase: {h42t['min']:.3f}; {h42t['late']:.1%}). The final value hides the instability.",
              f"- Seed 41's plain loop grokked during the decay, so it says less about holding a solution. Its twin on "
              f"Nanda's recipe partly recovered on its own (grokked at {p41c['t_grok']}, loss > 1 in "
              f"{p41c['late']:.0%} of late steps, last {LAST_K} {p41c['mean']:.2f} / {p41c['min']:.2f}, "
              f"{p41c['final']:.2f} at 20k) but didn't settle; two-phase: {p41t['late']:.1%}, {p41t['mean']:.2f} / "
              f"{p41t['min']:.2f}, {p41t['final']:.2f}. Don't describe it as fixing itself.",
              "- The two-phase spikes stop while the lr is still high, so they don't stop only because the lr "
              f"gets small: loss > 1 in under {QUIET:.0%} of every {BIN}-step bin from step "
              + ", ".join(f"{r['quiet']} in the {label} (lr ≈ {r['lr']:.1e})" for label, recipe, r in pairs
                          if recipe == "two-phase")
              + ". Only the final settling of accuracy coincides with a small lr.",
              "- Limits to state: the branch arms are one seed and one checkpoint; from scratch, clipping and lr "
              "decay aren't separated (the branch arms separate them: decay holds accuracy, clipping cuts spikes); "
              "it's the standard fix for an unstable recipe, not a finding; no hyperparameter sweep was done.",
              "- Not in these figures (runs not in results/): turning weight decay off at 14k instead, and clipping "
              "from step 0."]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("\n".join(lines) + "\n")
    return out


def main(argv=None):
    """run the cli.

    args:
        argv: argument list, or None for sys.argv.
    returns:
        exit code: 0 ok.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(cm.RESULTS / "runs"))
    ap.add_argument("--out", default=str(cm.OUT / "dynamics"))
    ap.add_argument("--supporting", action="store_true",
                    help="also draw the instability, collapse and two-phase figures, which aren't in the write-up, "
                         "into <out>/supporting/")
    args = ap.parse_args(argv)
    print(*plot_fixes(args.root, f"{args.out}/fixes.png"), sep="\n")
    if args.supporting:
        print(*plot_instability(args.root, f"{args.out}/supporting/instability.png"), sep="\n")
        print(plot_collapse(args.root, f"{args.out}/supporting/collapse.png"))
        print(plot_two_phase(args.root, f"{args.out}/supporting/two_phase.png"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
