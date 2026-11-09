"""draw the write-up's mechanism figures and the seed comparison from the analysis outputs in results/mechanism/.

    uv run python figures/mechanism.py [--mech results/mechanism] [--root results/runs] [--out figures/out/mechanism]
        [--supporting]

attention.png and .md: each model's attention from = over 3- and 6-hop chains, per head, at the first pass of its
looped block; the details and related numbers go to the .md (experiments/mechanism/internals.py).
clocks.png and .md: how sparse each model's W_E and W_U are over frequencies (gini), how much of the logits a clock of
the answer explains with k frequencies, and the loss with the top k readout frequencies removed, each against chance;
the details and per-run numbers go to the .md (internals.py and frequencies.py).
compute_early.png and .md: per plain loop, the accuracy after each pass of a linear probe of the answer on the residual
the unembedding reads and of the model's own output, on 1- and 3-hop chains; the huginn-style loop, the full table and
details go to the .md (internals.py).
mechanism_seeds.md: headline numbers at seeds 41 and 42 side by side, with their mean where both runs grokked.

--supporting also draws figures that aren't in the write-up, into <out>/supporting/:
passes: accuracy read out after each pass, for chains of 1-6 hops, at the final two-phase checkpoints; huginn averaged
over seeds 41 and 42, the sandwich loop one panel per seed (experiments/mechanism/passes.py).
past_depth: readout loss and the size of a 1% nudge to the state at pass 8, run to 256 passes, averaged over the
1-3 hop chains seen in training (experiments/mechanism/past-depth.py).
frequencies: loss with the top k readout frequencies removed, and how much of the logits a clock of the answer
explains with k frequencies (experiments/mechanism/frequencies.py).
state_use: the two-phase huginn across training: accuracy with noise in place of the carried state, and the norms of
the injection matrix's state and input halves (experiments/mechanism/state-use.py). the seed-42 reference huginn is
the same run until the switch at 14k and then keeps a constant lr, so it is drawn after the switch as the branch
without clipping or lr decay.
dilution: per hop count, the readout clock's amplitude relative to 1 hop against an even average's 3 / (h + 2), and
the accuracy (experiments/mechanism/internals.py).
decoding: through each model's blocks, how well a linear readout decodes the answer from the residual at =, after each
block's attention and after the whole block, against the model's own readout per pass (experiments/mechanism/
internals.py and passes.py).
clocks draws seed 42 slashed and dashed, the supporting figures dashed; a run that didn't grok is left out and named.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

import common as cm

pp = cm.pp
plt = cm.plt

# passes figure panels: (config, seeds averaged, title). huginn's seeds agree within 0.06 at every pass, so they are
# averaged; the sandwich loop's differ by up to 0.95, so each gets a panel; the state-norm loop didn't grok at 42
PASSES_PANELS = [
    ("t6_ut_statenorm_twophase", (41,), "plain loop, state norm: seed 41\n(seed 42 didn't grok)"),
    ("t6_ut_sandwich_twophase", (41,), "plain loop, sandwich norms: seed 41"),
    ("t6_ut_sandwich_twophase", (42,), "plain loop, sandwich norms: seed 42"),
    ("t6_huginn_twophase", (41, 42), "Huginn: mean of seeds 41 and 42"),
]
TRAIN_DEPTH = 8
TRAIN_HOPS = ("1", "2", "3")
SEEDS = (41, 42)
# configs analysed at both seeds: (config, label, colour)
TWOPHASE = [
    ("t6_ut_statenorm_twophase", "plain loop, state norm", pp.SERIES[1]),
    ("t6_ut_sandwich_twophase", "plain loop, sandwich norms", pp.SERIES[2]),
    ("t6_huginn_twophase", "Huginn-style loop", pp.SERIES[0]),
]
FIXED = ("t1_huginn_fixedT8", "Huginn, fixed depth 8", pp.SERIES[3])
ANCHOR = ("t1_anchor_nope_ref", "one-layer model", pp.INK_2)
REFERENCE = "t1_huginn_ref"
# a run counts as grokked at a checkpoint from this mean val accuracy on
GROKKED = 0.9
# step where the two-phase recipe starts clipping and decaying the lr
SWITCH = 14000
# passes after the eval depth at which state-use.py counts distinct answers
AFTER = ("1", "8", "56")
SHOW_K = (1, 2, 5, 10, 20, 40, 56)
HOP_COLOURS = LinearSegmentedColormap.from_list("hops", [cm.RAMP[0], cm.RAMP[3]])
# the dilution figure fits the readout clock at this many frequencies, and leaves out runs whose 1-hop fit is worse
CLOCK_K = 20
CLOCK_MIN_R2 = 0.5
SLASHES = "///"
SLASHED_ALPHA = 0.18


def load(mech, kind, name):
    """one model's analysis output.

    args:
        mech: folder holding the passes/ and past-depth/ outputs.
        kind: "passes" or "past-depth".
        name: run name.
    returns:
        the parsed json.
    """
    return json.loads((Path(mech) / kind / f"{name}.json").read_text())


def depth_line(ax, label=True, y=0.3):
    """a dotted vertical line at the training depth.

    args:
        ax: matplotlib axis.
        label: write "training depth" next to it.
        y: height of the label, as a fraction of the axis.
    """
    ax.axvline(TRAIN_DEPTH, color=pp.INK_2, linewidth=0.9, linestyle=(0, (1, 2)))
    if label:
        ax.text(TRAIN_DEPTH * 1.04, y, "training depth", color=pp.INK_2, fontsize=8.5,
                transform=ax.get_xaxis_transform())


def plot_passes(mech, out):
    """draw per-pass accuracy by chain length, one panel per model.

    args:
        mech: folder holding the analysis outputs.
        out: output path.
    returns:
        path to the written figure.
    """
    fig, axes = plt.subplots(1, len(PASSES_PANELS), figsize=(17, 4.8), facecolor=pp.SURFACE, sharey=True)
    for ax, (config, seeds, title) in zip(axes, PASSES_PANELS):
        ds = [load(mech, "passes" if s == SEEDS[0] else "passes-s42", name(config, s)) for s in seeds]
        pp.style_axis(ax, title, "accuracy")
        ax.set_xlabel("pass", fontsize=9, color=pp.INK_2)
        hops = sorted(ds[0]["by_hop"], key=int)
        for i, h in enumerate(hops):
            acc = np.mean([d["by_hop"][h]["acc"] for d in ds], axis=0)
            dashed = h not in TRAIN_HOPS
            ax.plot(np.arange(1, len(acc) + 1), acc, color=HOP_COLOURS(i / (len(hops) - 1)), linewidth=1.6,
                    linestyle=cm.DASHED if dashed else "-", label=f"{h} hop{'s' if h != '1' else ''}")
        depth_line(ax, label=ax is axes[-1])
        ax.set_xlim(1, ds[0]["T"])
        ax.set_ylim(-0.02, 1.02)
    cm.legend(fig, axes[0], 6, y=0.9)
    fig.suptitle("Accuracy read out after each pass, by chain length", x=0.01, y=0.99, ha="left", fontsize=14,
                 color=pp.INK)
    fig.text(0.01, 0.925, "final two-phase checkpoints. dashed: chains longer than any seen in training. dotted vertical "
             "line: the training depth. Huginn's two seeds agree within 0.06 at every pass; the sandwich loop's don't, "
             "so each seed gets a panel.", fontsize=9, color=pp.INK_2, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.82))
    return cm.save(fig, out)


def name(config, seed):
    """full run name of a config at a seed.

    args:
        config: config name without data or seed.
        seed: seed.
    returns:
        run name.
    """
    return f"{config}_p113_h3_x6_s{seed}"


def past_depth_kind(config, seed):
    """the results/mechanism folder holding a run's past-depth output.

    args:
        config: config name.
        seed: seed.
    returns:
        folder name.
    """
    if config == FIXED[0]:
        return "past-depth-fixed"
    return "past-depth" if seed == SEEDS[0] else "past-depth-s42"


def state_use_kind(config):
    """the results/mechanism folder holding a run's state-use output.

    args:
        config: config name.
    returns:
        folder name.
    """
    return {FIXED[0]: "state-use-fixed", REFERENCE: "state-use-variants"}.get(config, "state-use-twophase")


def status(root, run):
    """grokking step and mean val accuracy at the run's last evaluation.

    args:
        root: folder holding run directories.
        run: run name.
    returns:
        (t_grok or None, accuracy).
    """
    rows, evals, _ = cm.load_rows(cm.run_dir(run, root))
    return cm.grok_step(rows), cm.mean_acc(evals[-1]["val"])


def grokked(root, run):
    """whether a run grokked and is still grokked at its last evaluation.

    args:
        root: folder holding run directories.
        run: run name.
    returns:
        bool.
    """
    t_grok, acc = status(root, run)
    return t_grok is not None and acc >= GROKKED


def seed_style(seed):
    """line style for a seed: solid for the first, dashed for the second.

    args:
        seed: seed.
    returns:
        matplotlib linestyle.
    """
    return "-" if seed == SEEDS[0] else cm.DASHED


def plot_past_depth(mech, root, out):
    """draw readout loss and perturbation size against pass, to 256 passes.

    args:
        mech: folder holding the analysis outputs.
        root: folder holding run directories.
        out: output path.
    returns:
        path to the written figure.
    """
    fig, (ax_loss, ax_pert) = plt.subplots(1, 2, figsize=(13, 4.8), facecolor=pp.SURFACE)
    pp.style_axis(ax_loss, "readout loss", "cross-entropy", log_y=True)
    pp.style_axis(ax_pert, "size of a 1% nudge added after pass 8", "nudge size / state size", log_y=True)
    left_out = []
    for config, label, colour in TWOPHASE + [FIXED]:
        for seed in SEEDS:
            run = name(config, seed)
            if not grokked(root, run):
                left_out.append(f"{label} at seed {seed}")
                continue
            d = load(mech, past_depth_kind(config, seed), run)
            passes = np.arange(1, d["T"] + 1)
            loss = np.mean([d["by_hop"][h]["loss"] for h in TRAIN_HOPS], axis=0)
            pert = np.mean([d["by_hop"][h]["pert"] for h in TRAIN_HOPS], axis=0)
            style = seed_style(seed)
            # an exact zero has no place on a log axis
            ax_loss.plot(passes, np.where(loss > 0, loss, np.nan), color=colour, linewidth=1.6, linestyle=style,
                         label=label if seed == SEEDS[0] else None)
            ax_pert.plot(passes, np.where(pert > 0, pert, np.nan), color=colour, linewidth=1.6, linestyle=style)
    for ax in (ax_loss, ax_pert):
        ax.set_xscale("log", base=2)
        ax.set_xlabel("pass", fontsize=9, color=pp.INK_2)
        ax.set_xlim(1, 256)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
        depth_line(ax, label=ax is ax_loss, y=0.94)
        cm.tidy_log_ticks(ax)
    cm.loss_axis(ax_loss)
    cm.tidy_log_ticks(ax_loss)
    cm.legend(fig, ax_loss, 4, y=0.87)
    fig.suptitle("Running the loop past its training depth", x=0.01, y=0.99, ha="left", fontsize=14, color=pp.INK)
    fig.text(0.01, 0.925, "final checkpoints (two-phase recipe, except fixed-depth Huginn on Nanda's recipe); mean over "
             "1–3 hop chains. solid: seed 41; dashed: seed 42. dotted horizontal: ln 113.",
             fontsize=9, color=pp.INK_2, ha="left")
    if left_out:
        fig.text(0.01, 0.89, f"not shown (val accuracy below {GROKKED} at the end): {'; '.join(left_out)}.",
                 fontsize=9, color=pp.INK_2, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.83))
    return cm.save(fig, out)


def plot_frequencies(mech, root, out):
    """draw the excluded loss and the clock-fit r² against the number of frequencies.

    args:
        mech: folder holding the analysis outputs.
        root: folder holding run directories.
        out: output path.
    returns:
        path to the written figure.
    """
    fig, (ax_exc, ax_r2) = plt.subplots(1, 2, figsize=(13, 4.8), facecolor=pp.SURFACE)
    pp.style_axis(ax_exc, "train loss with the top k frequencies removed", "cross-entropy")
    pp.style_axis(ax_r2, "logit variance a clock of the answer explains", "r²")
    left_out = []
    for config, label, colour in [ANCHOR] + TWOPHASE:
        for seed in SEEDS:
            run = name(config, seed)
            if not grokked(root, run):
                left_out.append(f"{label} at seed {seed}")
                continue
            d = load(mech, "frequencies", run)
            style = seed_style(seed)
            ax_exc.plot(d["ks"], d["excluded_loss"], color=colour, linewidth=1.6, linestyle=style, marker="o",
                        markersize=3, label=label if seed == SEEDS[0] else None)
            ax_r2.plot(d["ks"], d["clock_r2"], color=colour, linewidth=1.6, linestyle=style, marker="o",
                       markersize=3)
    ax_exc.axhline(cm.UNIFORM_LOSS, color=pp.INK_2, linewidth=0.9, linestyle=(0, (1, 2)))
    ax_r2.set_ylim(0, 1)
    for ax in (ax_exc, ax_r2):
        ax.set_xscale("log", base=2)
        ax.set_xticks(SHOW_K)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
        ax.set_xlabel("k, frequencies (largest first)", fontsize=9, color=pp.INK_2)
    cm.legend(fig, ax_exc, 4, y=0.9)
    fig.suptitle("How many frequencies the readout uses (1-hop grid)", x=0.01, y=0.99, ha="left", fontsize=14,
                 color=pp.INK)
    note = f" not shown (val accuracy below {GROKKED} at the end): {'; '.join(left_out)}." if left_out else ""
    fig.text(0.01, 0.925, "final checkpoints; loops at their training depth. solid: seed 41; dashed: seed 42. "
             "dotted: ln 113, the loss of a uniform guess." + note, fontsize=9, color=pp.INK_2, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.82))
    return cm.save(fig, out)


def style_key(solid, dashed):
    """legend entries that explain the two line styles of a panel.

    args:
        solid: label for the solid line.
        dashed: label for the dashed line with hollow markers.
    returns:
        list of matplotlib handles.
    """
    line = dict(color=pp.INK_2, linewidth=1.6, marker="o", markersize=4)
    return [Line2D([], [], label=solid, **line),
            Line2D([], [], label=dashed, linestyle=cm.DASHED, markerfacecolor="none", **line)]


def plot_state_use(mech, root, out):
    """draw huginn's reliance on its carried state and its injection-matrix norms across training, for the two-phase
    runs and for the seed-42 branch that keeps a constant lr after the switch.

    args:
        mech: folder holding the analysis outputs.
        root: folder holding run directories.
        out: output path.
    returns:
        path to the written figure.
    """
    # (run, analysis folder, label, colour, mark its grokking step)
    lines = [(name(TWOPHASE[2][0], s), "state-use-twophase", f"two-phase recipe, seed {s}", pp.SERIES[i], True)
             for i, s in enumerate(SEEDS)]
    lines.append((name(REFERENCE, SEEDS[1]), "state-use-constant-lr",
                  f"seed {SEEDS[1]}, constant lr after {SWITCH // 1000}k (the same run until then)", pp.SERIES[2], False))
    fig, (ax_acc, ax_w) = plt.subplots(1, 2, figsize=(13, 5.2), facecolor=pp.SURFACE)
    pp.style_axis(ax_acc, "accuracy, 1–3 hops", "accuracy")
    pp.style_axis(ax_w, "injection matrix W_A, spectral norm of each half", "spectral norm")
    for run, kind, label, c, mark in lines:
        rows = load(mech, kind, run)["checkpoints"]
        steps = [r["step"] for r in rows]
        solid = dict(color=c, linewidth=1.6, marker="o", markersize=4)
        dashed = dict(solid, linestyle=cm.DASHED, markerfacecolor="none")
        ax_acc.plot(steps, [r["normal"] for r in rows], label=label, **solid)
        ax_acc.plot(steps, [r["noise"] for r in rows], **dashed)
        ax_w.plot(steps, [r["blocks"]["state"]["spectral"] for r in rows], **solid)
        ax_w.plot(steps, [r["blocks"]["input"]["spectral"] for r in rows], **dashed)
        if mark:
            for ax in (ax_acc, ax_w):
                cm.mark_step(ax, status(root, run)[0], c)
    ax_acc.set_ylim(-0.02, 1.02)
    ax_w.set_ylim(bottom=0)
    for ax in (ax_acc, ax_w):
        ax.axvline(SWITCH, color=pp.INK_2, linewidth=0.9, linestyle=cm.DASHED)
        ax.set_xlabel("step", fontsize=9, color=pp.INK_2)
    ax_acc.legend(handles=style_key("as trained", "noise as the state"), frameon=False, fontsize=8.5,
                  loc="upper left")
    ax_w.legend(handles=style_key("state half", "input half"), frameon=False, fontsize=8.5, loc="upper right")
    cm.legend(fig, ax_acc, len(lines), y=0.85)
    fig.suptitle("Two-Phase Huginn-Style Loop: Reliance on Its Carried State",
                 x=0.01, y=0.99, ha="left", fontsize=14, color=pp.INK)
    fig.text(0.01, 0.93, "▼ grokking step of each two-phase run (mean val accuracy ≥ 0.9). dashed vertical line: the "
             f"two-phase recipe adds gradient clipping and lr decay from step {SWITCH // 1000}k.",
             fontsize=9, color=pp.INK_2, ha="left")
    fig.text(0.01, 0.895, "noise as the state: fresh random noise replaces the carried state before every pass; a model "
             "that ignores its state keeps its accuracy. the constant-lr run is the Huginn-style reference.",
             fontsize=9, color=pp.INK_2, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.83))
    return cm.save(fig, out)


def plot_dilution(mech, root, out):
    """draw how the readout clock's amplitude and the accuracy change with chain length, against the shrinking an
    even average over the h + 2 tokens would cause.

    args:
        mech: folder holding the analysis outputs.
        root: folder holding run directories.
        out: output path.
    returns:
        path to the written figure.
    """
    fig, (ax_amp, ax_acc) = plt.subplots(1, 2, figsize=(13, 4.8), facecolor=pp.SURFACE)
    pp.style_axis(ax_amp, f"readout clock amplitude ({CLOCK_K} frequencies), relative to 1 hop", "amplitude ratio")
    pp.style_axis(ax_acc, "accuracy", "accuracy")
    hops = np.arange(1, len(TRAIN_HOPS) * 2 + 1)
    ax_amp.plot(hops, 3 / (hops + 2), color=pp.INK_2, linewidth=1.2, linestyle=(0, (1, 2)))
    ax_amp.text(hops[-1], 3 / (hops[-1] + 2) + 0.08, "even average\nover h + 2 tokens", ha="right", va="bottom",
                fontsize=8.5, color=pp.INK_2)
    left_out, no_clock = [], []
    for config, label, colour in [ANCHOR] + TWOPHASE:
        for seed in SEEDS:
            run = name(config, seed)
            if not grokked(root, run):
                left_out.append(f"{label} at seed {seed}")
                continue
            d = load(mech, "internals", run)["dilution"]
            line = dict(color=colour, linewidth=1.6, linestyle=seed_style(seed), marker="o", markersize=3)
            ax_acc.plot(hops, [d[str(h)]["acc"] for h in hops], label=label if seed == SEEDS[0] else None, **line)
            if d["1"][f"r2_{CLOCK_K}"] < CLOCK_MIN_R2:
                no_clock.append(f"{label} at seed {seed}")
                continue
            amp = np.array([d[str(h)][f"amp_{CLOCK_K}"] for h in hops])
            ax_amp.plot(hops, amp / amp[0], **line)
    for ax in (ax_amp, ax_acc):
        ax.axvspan(len(TRAIN_HOPS) + 0.5, hops[-1] + 0.3, color=pp.INK_2, alpha=0.06, linewidth=0)
        ax.set_xlim(0.7, hops[-1] + 0.3)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(hops)
        ax.set_xlabel("hops in the chain (shaded: longer than any seen in training)", fontsize=9, color=pp.INK_2)
    cm.legend(fig, ax_acc, 4, y=0.86)
    fig.suptitle("Every model averages its tokens, but only the one-layer model's answer signal shrinks like an average",
                 x=0.01, y=0.99, ha="left", fontsize=14, color=pp.INK)
    fig.text(0.01, 0.93, "final checkpoints, loops at their training depth; first 1000 chains per hop count. solid: seed "
             "41; dashed: seed 42. attention from = is an even average over the h + 2 tokens in every loop.",
             fontsize=9, color=pp.INK_2, ha="left")
    notes = [f"not in the left panel (clock r² below {CLOCK_MIN_R2} at 1 hop): {'; '.join(no_clock)}." if no_clock
             else "", f"not shown (not grokked): {'; '.join(left_out)}." if left_out else ""]
    fig.text(0.01, 0.895, " ".join(n for n in notes if n), fontsize=9, color=pp.INK_2, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.8))
    return cm.save(fig, out)


def plot_decoding(mech, out):
    """draw, through each model's blocks, how well a linear readout decodes the answer from the residual stream at =,
    against the model's own readout after each pass.

    args:
        mech: folder holding the analysis outputs.
        out: output path.
    returns:
        path to the written figure.
    """
    fig, axes = plt.subplots(1, len(PASSES_PANELS), figsize=(17, 4.8), facecolor=pp.SURFACE, sharey=True)
    for ax, (config, seeds, title) in zip(axes, PASSES_PANELS):
        sums = [load(mech, "internals", name(config, s))["grid"]["sum"] for s in seeds]
        reads = [load(mech, "passes" if s == SEEDS[0] else "passes-s42", name(config, s))["by_hop"]["1"]["acc"]
                 for s in seeds]
        T = len([b for b in sums[0] if b.startswith("core")])
        xs, ys, filled = [], [], []
        for b in sums[0]:
            x = 0 if b == "prelude" else T + 1 if b == "coda" else int(b.split()[-1])
            for dx, key, solid in ((-0.5, "decode_mid", False), (0, "decode", True)):
                xs.append(x + dx)
                ys.append(float(np.mean([s[b][key] for s in sums])))
                filled.append(solid)
        ax.plot(xs, ys, color=pp.SERIES[0], linewidth=1.6)
        ax.scatter([x for x, f in zip(xs, filled) if f], [y for y, f in zip(ys, filled) if f], color=pp.SERIES[0],
                   s=18, zorder=3, label="answer decoded linearly from the residual at = (after each block)")
        ax.scatter([x for x, f in zip(xs, filled) if not f], [y for y, f in zip(ys, filled) if not f],
                   facecolor=pp.SURFACE, edgecolor=pp.SERIES[0], s=18, zorder=3, label="… after that block's attention")
        read = np.mean(reads, axis=0)[:T]
        ax.plot(np.arange(1, T + 1), read, color=pp.INK_2, linewidth=1.4, linestyle=cm.DASHED,
                label="the model's own readout after each pass (1-hop chains)")
        ticks = list(range(1, T + 1))
        labels = [str(k) for k in ticks]
        if "prelude" in sums[0]:
            ticks, labels = [0] + ticks + [T + 1], ["prelude"] + labels + ["coda"]
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_xlim(min(xs) - 0.3, max(ticks) + 0.3)
        ax.set_ylim(-0.02, 1.02)
        pp.style_axis(ax, title, "accuracy")
        ax.set_xlabel("block / pass", fontsize=9, color=pp.INK_2)
    cm.legend(fig, axes[0], 3, y=0.86)
    fig.suptitle("On 1-hop chains, a linear probe finds the answer passes before the plain loops' own readout does; in "
                 "Huginn the prelude already holds most of it",
                 x=0.01, y=0.99, ha="left", fontsize=14, color=pp.INK)
    fig.text(0.01, 0.93, "final two-phase checkpoints. decoded: least squares onto the one-hot answer, fitted on half of "
             "the 1-hop grid and scored on the other half (chance 0.01). the one-layer model decodes at 0.02-0.03 after "
             "its attention and 0.95-0.96 after its MLP.", fontsize=9, color=pp.INK_2, ha="left")
    fig.text(0.01, 0.895, "Huginn's own readout runs each pass's state through the coda; its passes start from a random "
             "state, concatenated with the prelude's output.", fontsize=9, color=pp.INK_2, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.8))
    return cm.save(fig, out)


# the attention figure: (config, label) at seed 41, the looped block's first pass
ATTENTION_PANELS = [ANCHOR[:2]] + [c[:2] for c in TWOPHASE]
# the attention figure counts the one-layer model's heads that put less than this on = itself
SKIP_EQ = 0.1
# the attention figure's second row: the longest chains tested
ATTENTION_LONG = "6"
# the full attention patterns in internals.py are on chains of this length
PATTERN_LONGEST = "3-hop"
# the compute-early table: a probe or readout counts from the first pass at or above this accuracy
READ_AT = 0.9
# the compute-early table's rows: (config, label, seed)
EARLY_ROWS = [(ANCHOR[0], ANCHOR[1], s) for s in SEEDS] + [(TWOPHASE[0][0], TWOPHASE[0][1], 41)] + [
    (TWOPHASE[1][0], TWOPHASE[1][1], s) for s in SEEDS] + [(TWOPHASE[2][0], TWOPHASE[2][1], s) for s in SEEDS]
EARLY_HOPS = ("1", "3")
# the compute-early figure's panels, the plain loops: (1-hop, 3-hop) shades of each model's colour for the probe,
# checked for colour-blind separation against each other and the output's grey
EARLY_SHADES = {TWOPHASE[0][0]: ("#f0875a", "#c4501f"), TWOPHASE[1][0]: ("#5cc89c", "#179467")}
# the compute-early figure's marker per hop count: (marker, filled)
EARLY_MARKS = {"1": ("o", True), "3": ("s", False)}
# the compute-early figure's panels, (config, seed): the two on top read out late after computing early; the last,
# centred below and starred for the post's caption, is the seed-41 sandwich loop, which learned a different solution
EARLY_PANELS = [(TWOPHASE[0][0], 41), (TWOPHASE[1][0], 42), (TWOPHASE[1][0], 41)]


def internals_runs(mech, configs):
    """the internals outputs that exist for some configs, at both seeds.

    args:
        mech: folder holding the analysis outputs.
        configs: config names.
    returns:
        dict from (config, seed) to the parsed json; a run that didn't grok has no output and is skipped.
    """
    out = {}
    for config in configs:
        for seed in SEEDS:
            path = Path(mech) / "internals" / f"{name(config, seed)}.json"
            if path.exists():
                out[(config, seed)] = json.loads(path.read_text())
    return out


def tokens_for(n):
    """axis labels for a chain of n tokens, = last.

    args:
        n: number of tokens.
    returns:
        list of labels: x₀, x₁, ..., =.
    """
    return [f"x{chr(0x2080 + i)}" for i in range(n - 1)] + ["="]


def plot_attention(mech, out):
    """draw the attention from = over a chain, per head, at the first pass of each model's looped block, for chains of
    the longest trained length and of the longest tested length; the details go to a markdown file next to it.

    args:
        mech: folder holding the analysis outputs.
        out: output path of the image; the notes go next to it with a .md suffix.
    returns:
        (image path, notes path).
    """
    seed = SEEDS[0]
    rows = [(TRAIN_HOPS[-1], f"{TRAIN_HOPS[-1]}-hop chains"), (ATTENTION_LONG, f"{ATTENTION_LONG}-hop chains")]
    data = {c: load(mech, "internals", name(c, seed))["attention"]["core pass 1"] for c, _ in ATTENTION_PANELS}
    weight = LinearSegmentedColormap.from_list("weight", [pp.SURFACE, cm.RAMP[0], cm.RAMP[2]])
    fig = plt.figure(figsize=(16.5, 7.3), facecolor=pp.SURFACE, layout="constrained")
    # rect leaves a margin at the image's edge; the pads (inches) space the panels
    fig.get_layout_engine().set(rect=(0.025, 0.035, 0.95, 0.865), h_pad=0.08, w_pad=0.12, wspace=0.07)
    subs = fig.subfigures(len(rows), 1, hspace=0.04)
    for r, ((h, row_label), sub) in enumerate(zip(rows, subs)):
        sub.set_facecolor(pp.SURFACE)
        sub.suptitle(row_label.title(), fontsize=cm.PANEL_SIZE, color=pp.INK_2)
        axes = sub.subplots(1, len(ATTENTION_PANELS))
        n = int(h) + 2
        for c, ((config, label), ax) in enumerate(zip(ATTENTION_PANELS, axes)):
            w = np.array(data[config][h]["mean"])
            # an even average sits at the middle of the colour scale
            ax.pcolormesh(w, cmap=weight, vmin=0, vmax=2 / n, edgecolors=pp.SURFACE, linewidth=2.5)
            ax.invert_yaxis()
            for hd in range(w.shape[0]):
                for k in range(n):
                    ax.text(k + 0.5, hd + 0.5, f"{w[hd, k]:.2f}", ha="center", va="center", fontsize=9,
                            color=pp.INK if w[hd, k] < 1.6 / n else pp.SURFACE)
            ax.set_xticks(np.arange(n) + 0.5)
            ax.set_xticklabels(tokens_for(n), fontsize=cm.LABEL_SIZE + 0.5, color=pp.INK_2)
            ax.set_yticks(np.arange(w.shape[0]) + 0.5)
            ax.set_yticklabels([f"head {i + 1}" for i in range(w.shape[0])] if c == 0 else [],
                               fontsize=cm.LABEL_SIZE, color=pp.INK_2)
            ax.tick_params(length=0, pad=4)
            for side in ax.spines.values():
                side.set_visible(False)
            if r == 0:
                ax.set_title(label.title(), fontsize=cm.PANEL_SIZE, color=pp.INK, pad=10)
    fig.suptitle("The Attention Averaging Shortcut", x=0.025, y=0.96, ha="left", fontsize=cm.TITLE_SIZE,
                 color=pp.INK)
    return cm.save(fig, out), write_attention_notes(mech, rows, str(Path(out).with_suffix(".md")))


def write_attention_notes(mech, rows, out):
    """write the attention figure's details and the related numbers, for writing the text.

    args:
        mech: folder holding the analysis outputs.
        rows: the figure's (hop count, label) rows.
        out: output path.
    returns:
        the output path.
    """
    loops = internals_runs(mech, [c for c, _, _ in TWOPHASE])
    anchors = internals_runs(mech, [ANCHOR[0]])
    labels = dict(ATTENTION_PANELS)
    hops = sorted({h for d in loops.values() for b in d["attention"].values() for h in b}, key=int)
    lines = ["# The Attention Averaging Shortcut: notes", "",
             "Generated by figures/mechanism.py from results/mechanism/internals/ (experiments/mechanism/internals.py). "
             "Not part of the figure.", "", "## What the figure shows", "",
             f"- The attention weight from `=` to each token of the chain, per head, averaged over the first 1000 chains "
             f"of the val set ({rows[0][0]} hops) and of the extrapolation set ({rows[1][0]} hops, longer than any "
             f"chain seen in training). Seed {SEEDS[0]}, "
             "final checkpoints.",
             "- Each model's first pass of its looped block: the one-layer model's only layer; the plain loops' shared "
             "block at pass 1; the Huginn-style loop's core at pass 1.",
             "- Colour: white at 0, the middle of the scale at an even average, 1 / (h + 2): "
             + ", ".join(f"{1 / (int(h) + 2):.3f} at {h} hops" for h, _ in rows) + ".", "",
             "## Loops", "",
             f"- From `=`: largest gap from an even average in any block, pass and head, at hop counts {hops[0]}-"
             f"{hops[-1]} (per-chain maximum, averaged over chains):"]
    for (config, seed), d in sorted(loops.items()):
        gap = max(max(r["max_gap"]) for b in d["attention"].values() for r in b.values())
        ent = min(min(r["entropy"]) for b in d["attention"].values() for r in b.values())
        lines.append(f"  - {labels[config]}, seed {seed}: {gap:.3f} (lowest normalized entropy {ent:.3f})")
    full = max(max(b["gap"]) for d in loops.values() for b in d["patterns"].values())
    lines += [f"- From every position (full pattern on {PATTERN_LONGEST} chains, every block and pass): largest gap "
              f"{full:.3f}, so the loops average evenly over everything each position sees, not only at `=`.",
              "- The state-norm loop didn't grok at seed 42, so it has no seed-42 numbers.", "", "## One-layer model", ""]
    for (config, seed), d in sorted(anchors.items()):
        a = d["attention"]["core pass 1"]
        pat = np.array(d["patterns"]["core pass 1"]["mean"])
        n = pat.shape[-1]
        even = 1 / (np.arange(n) + 1)
        operand = np.abs(pat[:, :-1] - even[None, :-1, None]).max(where=~np.triu(np.ones((n - 1, n), bool), 1)[None],
                                                                   initial=0)
        skip = {h: int((np.array(a[h]["on_eq"]) < SKIP_EQ).sum()) for h, _ in rows}
        lines.append(f"- Seed {seed}: weight on `=` itself per head "
                     + "; ".join(f"{h} hops {', '.join(f'{v:.2f}' for v in a[h]['on_eq'])}" for h, _ in rows)
                     + f". Heads under {SKIP_EQ} on `=`: " + ", ".join(f"{skip[h]} of 4 at {h} hops" for h, _ in rows)
                     + ". Largest gap from even at `=`: "
                     + ", ".join(f"{max(a[h]['max_gap']):.2f} at {h} hops" for h, _ in rows)
                     + f". Operand rows ({PATTERN_LONGEST} chains): within {operand:.3f} of even (mean pattern).")
    lines += ["- So the one-layer model learns the same averaging over the operands; at `=` its heads either skip "
              "`=` or weight it like an operand.", "", "## Related point for the text: dilution", "",
              f"Readout clock amplitude ({CLOCK_K} frequencies) relative to 1 hop, at hops 2-6, against an even "
              "average's 3 / (h + 2):", "",
              "- even average: " + ", ".join(f"{3 / (h + 2):.2f}" for h in range(2, 7))]
    for (config, seed), d in sorted({**anchors, **loops}.items()):
        dil = d["dilution"]
        if dil["1"][f"r2_{CLOCK_K}"] < CLOCK_MIN_R2:
            lines.append(f"- {labels[config]}, seed {seed}: no clock (r² {dil['1'][f'r2_{CLOCK_K}']:.2f} at 1 hop)")
            continue
        ratio = [dil[str(h)][f"amp_{CLOCK_K}"] / dil["1"][f"amp_{CLOCK_K}"] for h in range(2, 7)]
        lines.append(f"- {labels[config]}, seed {seed}: " + ", ".join(f"{v:.2f}" for v in ratio)
                     + f" (6-hop accuracy {dil['6']['acc']:.2f})")
    lines += ["", "The one-layer model's clock shrinks like an average; the loops' barely shrinks. Why (the loops' norms "
              "rescaling the average?) isn't tested."]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("\n".join(lines) + "\n")
    return out


def chance_label(ax, x, y, ha="left", va="bottom"):
    """write "chance" next to a chance line.

    args:
        ax: matplotlib axis.
        x, y: position in data coordinates.
        ha, va: alignment.
    """
    ax.text(x, y, "chance", fontsize=cm.NOTE_SIZE, color=pp.INK_2, ha=ha, va=va)


def seed_fill(colour, seed):
    """bar fill for one seed: shaded for seed 41, a pale fill with slashes in the colour for seed 42.

    args:
        colour: model colour.
        seed: seed.
    returns:
        dict of matplotlib patch keyword arguments.
    """
    if seed == SEEDS[0]:
        return dict(facecolor=colour, edgecolor=colour)
    return dict(facecolor=to_rgba(colour, SLASHED_ALPHA), edgecolor=colour, hatch=SLASHES)


def seed_key():
    """legend entries for the two seeds, each its bar fill next to its line style: a shaded box and a solid line for
    seed 41, a slashed box and a dashed line for seed 42; draw them with a HandlerTuple.

    returns:
        list of (patch, line) handle pairs.
    """
    return [(Patch(linewidth=1.2, **seed_fill(pp.INK_2, s)),
             Line2D([], [], color=pp.INK_2, linewidth=1.8, linestyle=seed_style(s))) for s in SEEDS]


def plot_clocks(mech, root, out):
    """draw how sparse each model's embedding and unembedding are over frequencies, how much of the logits a clock
    of the answer explains, and the loss with the readout's top frequencies removed; the details go to a markdown file
    next to it.

    args:
        mech: folder holding the analysis outputs.
        root: folder holding run directories.
        out: output path of the image; the notes go next to it with a .md suffix.
    returns:
        (image path, notes path).
    """
    configs = [ANCHOR] + TWOPHASE
    runs = {k: v for k, v in internals_runs(mech, [c for c, _, _ in configs]).items() if grokked(root, name(*k))}
    fig = plt.figure(figsize=(17, 6.4), facecolor=pp.SURFACE, layout="constrained")
    fig.get_layout_engine().set(rect=(0.02, 0.03, 0.96, 0.78), h_pad=0.1, w_pad=0.12, wspace=0.08)
    ax_gini, ax_r2, ax_exc = fig.subplots(1, 3, gridspec_kw={"width_ratios": [0.8, 1, 1]})
    for ax, title in [(ax_gini, "Embedding Matrix Sparsity (Gini Coefficient)"),
                      (ax_r2, "Variance Explained by Top K Frequencies"),
                      (ax_exc, "Excluded Loss (Top K Freq. Removed)")]:
        pp.style_axis(ax, "", "")
        cm.text_sizes(ax)
        ax.set_title(title, loc="center", fontsize=cm.PANEL_SIZE, color=pp.INK, pad=10)
        ax.set_xlabel("")

    # gini bars: one group per matrix, one bar per grokked run; shaded for seed 41, slashed for seed 42
    groups = [("W_E", "gini"), ("W_U", "unembed_gini")]
    bars = [(config, seed, colour) for config, _, colour in configs for seed in SEEDS if (config, seed) in runs]
    width = 0.86 / len(bars)
    for g, (_, key) in enumerate(groups):
        for i, (config, seed, colour) in enumerate(bars):
            x = g + (i - (len(bars) - 1) / 2) * width
            ax_gini.bar(x, runs[(config, seed)]["embedding"][key], width * 0.84, linewidth=1.2,
                        **seed_fill(colour, seed))
    chance = np.mean([d["embedding"]["gini_chance"] for d in runs.values()])
    ax_gini.axhline(chance, color=pp.INK_2, linewidth=0.9, linestyle=(0, (1, 2)))
    # the bars leave no room above the line, so the label sits just past the axis end
    ax_gini.text(1.01, chance, "chance", fontsize=cm.NOTE_SIZE, color=pp.INK_2, ha="left", va="center",
                 transform=ax_gini.get_yaxis_transform())
    ax_gini.set_xticks(range(len(groups)))
    ax_gini.set_xticklabels([f"${m[0]}_{m[2]}$" for m, _ in groups], fontsize=cm.PANEL_SIZE, color=pp.INK)
    ax_gini.set_xlim(-0.5, len(groups) - 0.5)
    ax_gini.set_ylim(0, 1)
    ax_gini.set_ylabel("gini (0: even, 1: one frequency)", fontsize=cm.LABEL_SIZE, color=pp.INK_2)

    top = {}
    for config, label, colour in configs:
        for seed in SEEDS:
            if (config, seed) not in runs:
                continue
            d = load(mech, "frequencies", name(config, seed))
            style = dict(color=colour, linewidth=1.8, linestyle=seed_style(seed))
            ax_r2.plot(d["ks"], d["clock_r2"], **style)
            ax_exc.plot(d["ks"], d["excluded_loss"], **style)
            top[(config, seed)] = (max(d["excluded_loss"]), d["ks"], d["excluded_loss"])
    p_half = len(next(iter(runs.values()))["embedding"]["norms"])
    k_all = np.arange(1, p_half + 1)
    ax_r2.plot(k_all, k_all / p_half, color=pp.INK_2, linewidth=0.9, linestyle=(0, (1, 2)))
    chance_label(ax_r2, 12, 12 / p_half - 0.02, va="top")
    ax_r2.set_ylim(0, 1)
    ax_r2.set_ylabel("r²", fontsize=cm.LABEL_SIZE, color=pp.INK_2)
    ax_exc.axhline(cm.UNIFORM_LOSS, color=pp.INK_2, linewidth=0.9, linestyle=(0, (1, 2)))
    chance_label(ax_exc, 1.05, cm.UNIFORM_LOSS + 0.12)
    cap = 7.5
    ax_exc.set_ylim(-0.2, cap)
    ax_exc.set_ylabel("cross-entropy", fontsize=cm.LABEL_SIZE, color=pp.INK_2)
    for (config, seed), (peak, ks, loss) in top.items():
        if peak > cap:
            # mark where a curve leaves the axis, with its peak
            k_out = next(k for k, v in zip(ks, loss) if v > cap)
            ax_exc.text(k_out, cap, f"↑ {peak:.1f}", fontsize=cm.NOTE_SIZE, color=pp.INK_2, ha="left", va="top")
    for ax in (ax_r2, ax_exc):
        ax.set_xscale("log", base=2)
        ax.set_xticks(SHOW_K)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
        ax.set_xlabel("k, frequencies (the readout's largest first)", fontsize=cm.LABEL_SIZE, color=pp.INK_2)
    handles = [Patch(color=c) for *_, c in configs] + seed_key()
    labels = [label for _, label, _ in configs] + [f"seed {s}" for s in SEEDS]
    # three columns, filled down: the models in the first two, the seeds in the last
    fig.legend(handles=handles, labels=labels, handler_map={tuple: HandlerTuple(ndivide=None, pad=0.5)},
               loc="upper center", bbox_to_anchor=(0.5, 0.925), ncol=3, frameon=False, fontsize=cm.LEGEND_SIZE,
               labelcolor=pp.INK, handlelength=3.4, columnspacing=4, labelspacing=0.7)
    fig.suptitle("How Clock-Like Is the Readout?", x=0.02, y=0.97, ha="left", fontsize=cm.TITLE_SIZE, color=pp.INK)
    return cm.save(fig, out), write_clock_notes(mech, root, runs, configs, str(Path(out).with_suffix(".md")))


def write_clock_notes(mech, root, runs, configs, out):
    """write the clocks figure's details, chance levels and per-run numbers, for writing the text.

    args:
        mech: folder holding the analysis outputs.
        root: folder holding run directories.
        runs: dict from (config, seed) to the internals output, grokked runs only.
        configs: (config, label, colour) for each model.
        out: output path.
    returns:
        the output path.
    """
    labels = {c: label for c, label, _ in configs}
    any_run = next(iter(runs.values()))["embedding"]
    p_half = len(any_run["norms"])
    left_out = [f"{labels[c]} at seed {s}" for c, _, _ in configs for s in SEEDS if (c, s) not in runs]
    lines = ["# How Clock-Like Is the Readout?: notes", "",
             "Generated by figures/mechanism.py from results/mechanism/internals/ and results/mechanism/frequencies/ "
             "(experiments/mechanism/internals.py and frequencies.py). Not part of the figure.", "",
             "## What the figure shows", "",
             "- Final checkpoints; loops at their training depth; readout measures on the 1-hop grid (every a + b mod "
             "113). Seed 41: shaded bars and solid lines; seed 42: slashed bars and dashed lines.",
             "- Left: the gini of each weight matrix's norm over the 56 frequencies, W_E over the input numbers and W_U "
             "over the answers. One bar per run.",
             "- Middle: the share of the row-centred logit variance that cos/sin(w(a + b - c)) at the readout's top k "
             "frequencies explains.",
             "- Right: the train-pair loss after removing the readout's top k frequencies from the logits.",
             f"- Not shown (val accuracy below {GROKKED} at the end): {'; '.join(left_out) or 'none'}.", "",
             "## Chance levels", "",
             f"- Gini: a random gaussian matrix of W_U's shape, {any_run['gini_chance']:.3f} ± "
             f"{any_run['gini_chance_sd']:.3f} (mean ± sd over 100 draws). W_E has the same shape over the numbers.",
             f"- r²: a readout with no clock, one-hot on the answer, explains exactly k / {p_half}.",
             f"- Excluded loss: ln 113 = {cm.UNIFORM_LOSS:.2f}, a uniform guess. Every correct readout reaches it by "
             f"k = {p_half}; what matters is how early. The no-clock readout's excluded loss is 0.01 / 0.18 / 2.08 / "
             "4.73 at k = 5 / 20 / 40 / 56.", "", "## Numbers per run", "",
             "| model | seed | W_E gini | W_U gini | W_E norm² on the readout's top 20 | r², k = 5 / 20 / 56 "
             "| excluded loss, k = 5 / 10 / 20 / 40 | peak excluded loss |", "|---|---|---|---|---|---|---|---|"]
    for config, label, _ in configs:
        for seed in SEEDS:
            if (config, seed) not in runs:
                continue
            e = runs[(config, seed)]["embedding"]
            d = load(mech, "frequencies", name(config, seed))
            at = lambda key, ks: " / ".join(f"{d[key][d['ks'].index(k)]:.2f}" for k in ks)  # noqa: E731
            lines.append(f"| {label} | {seed} | {e['gini']:.2f} | {e['unembed_gini']:.2f} | {e['share_20']:.0%} | "
                         f"{at('clock_r2', (5, 20, 56))} | {at('excluded_loss', (5, 10, 20, 40))} | "
                         f"{max(d['excluded_loss']):.1f} |")
    lines += ["", f"An even spread puts {20 / p_half:.0%} of W_E's norm² on any 20 frequencies.", "",
              "## For the text", "",
              "- The readout is a clock: about 5 frequencies in the one-layer model, about 20 in the loops, and W_U is "
              "far sparser over frequencies than W_E.",
              "- The one-layer model's excluded loss overshoots ln 113 (its remaining logits point confidently at wrong "
              "answers): that's logit scale, not more clock.",
              "- The seed-41 sandwich loop has no clock: r² near chance, excluded loss near the no-clock readout's, and "
              "W_U's gini near the other loops' W_E.",
              "- The input side is only partly clock-like. W_E's gini is well above chance but well below W_U's, and "
              "W_E leans on the readout's frequencies (the norm² column). The MLP neurons mostly don't follow a "
              "single frequency (results/mechanism/internals/, grid.neurons). So the clock is clear at the readout; "
              "how the sum is computed before it isn't established."]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("\n".join(lines) + "\n")
    return out


def first_pass(values):
    """the first pass whose accuracy reaches READ_AT.

    args:
        values: accuracies after passes 1, 2, ...
    returns:
        1-based pass, or None if none reaches it.
    """
    return next((k + 1 for k, v in enumerate(values) if v >= READ_AT), None)


def early_rows(mech):
    """per run, the probe on the residual the unembedding reads and the model's own accuracy after each pass, on the
    hop counts in EARLY_HOPS (internals.py's by_depth).

    args:
        mech: folder holding the analysis outputs.
    returns:
        list of (config, label, seed, {hops: {"probe", "readout"}}) in EARLY_ROWS order.
    """
    return [(config, label, seed, load(mech, "internals", name(config, seed))["by_depth"])
            for config, label, seed in EARLY_ROWS]


EARLY_TITLE = "Compute Early, Read Late"


def early_style(colour, hops):
    """line style for one compute-early series: a filled circle for 1 hop, a hollow square for 3 hops.

    args:
        colour: line colour.
        hops: hop count, a key of EARLY_MARKS.
    returns:
        dict of matplotlib line keyword arguments.
    """
    marker, filled = EARLY_MARKS[hops]
    return dict(color=colour, linewidth=1.8, marker=marker, markersize=6, markeredgewidth=1.4,
                markerfacecolor=colour if filled else "none", markeredgecolor=colour)


def write_compute_early(mech, out):
    """draw, for each plain loop, the accuracy after each pass of a linear probe of the answer and of the model's own
    output, on 1- and 3-hop chains; the full table, the left-out models and details go to a markdown file next to it.

    args:
        mech: folder holding the analysis outputs.
        out: output path of the image; the notes go next to it with a .md suffix.
    returns:
        (image path, notes path).
    """
    rows = early_rows(mech)
    by_run = {(config, seed): (label, d) for config, label, seed, d in rows}
    fig = plt.figure(figsize=(12, 10), facecolor=pp.SURFACE, layout="constrained")
    fig.get_layout_engine().set(rect=(0.03, 0.02, 0.94, 0.855), h_pad=0.25, w_pad=0.15, hspace=0.06, wspace=0.04)
    # two panels on top, the third centred below: each panel spans two of four columns
    grid = fig.add_gridspec(2, 4)
    axes = [fig.add_subplot(grid[0, 0:2]), fig.add_subplot(grid[0, 2:4]), fig.add_subplot(grid[1, 1:3])]
    chance = 1 / 113
    for i, (ax, (config, seed)) in enumerate(zip(axes, EARLY_PANELS)):
        label, d = by_run[(config, seed)]
        starred = i == len(EARLY_PANELS) - 1
        pp.style_axis(ax, "", "accuracy" if i != 1 else "")
        ax.set_title(f"{label.title()}: Seed {seed}{'*' if starred else ''}", loc="center", fontsize=cm.PANEL_SIZE,
                     color=pp.INK, pad=10)
        depth = len(d[EARLY_HOPS[0]]["probe"])
        xs = np.arange(1, depth + 1)
        for h in EARLY_HOPS:
            ax.plot(xs, d[h]["readout"], **early_style(pp.INK_2, h))
        for h, shade in zip(EARLY_HOPS, EARLY_SHADES[config]):
            ax.plot(xs, d[h]["probe"], **early_style(shade, h))
        ax.axhline(chance, color=pp.INK_2, linewidth=0.9, linestyle=(0, (1, 2)), zorder=0)
        chance_label(ax, depth + 0.35, chance + 0.02, ha="right")
        ax.set_xticks(xs)
        ax.set_xlim(0.65, depth + 0.35)
        ax.set_ylim(-0.03, 1.03)
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.set_xlabel(f"pass (trained at {TRAIN_DEPTH})", color=pp.INK_2)
        cm.text_sizes(ax)
    shades = list(EARLY_SHADES.values())
    probe = [tuple(Line2D([], [], **early_style(s[j], h)) for s in shades) for j, h in enumerate(EARLY_HOPS)]
    output = [Line2D([], [], **early_style(pp.INK_2, h)) for h in EARLY_HOPS]
    hop_name = lambda h: f"{h} hop{'s' if h != '1' else ''}"  # noqa: E731
    labels = [f"linear probe, {hop_name(h)}" for h in EARLY_HOPS] + [f"model's own output, {hop_name(h)}"
                                                                     for h in EARLY_HOPS]
    # two columns, filled down: the probe on the left, the output on the right
    fig.legend(handles=probe + output, labels=labels, handler_map={tuple: HandlerTuple(ndivide=None, pad=0.4)},
               loc="upper center", bbox_to_anchor=(0.5, 0.935), ncol=2, frameon=False, fontsize=cm.LEGEND_SIZE,
               labelcolor=pp.INK, handlelength=4.2, columnspacing=4, labelspacing=0.7)
    fig.suptitle(EARLY_TITLE, x=0.03, y=0.975, ha="left", fontsize=cm.TITLE_SIZE, color=pp.INK)
    return cm.save(fig, out), write_early_notes(mech, rows, str(Path(out).with_suffix(".md")))


def write_early_notes(mech, rows, out):
    """write the compute-early table's details: definitions, chance, the full table with the one-layer model and the
    accuracies after every pass, and the huginn-style loop's probe inside the loop, for writing the text.

    args:
        mech: folder holding the analysis outputs.
        rows: table rows from early_rows.
        out: output path.
    returns:
        the output path.
    """
    hop_name = lambda h: f"{h} hop{'s' if h != '1' else ''}"  # noqa: E731
    fmt = lambda v: "—" if v is None else str(v)  # noqa: E731
    lines = [f"# {EARLY_TITLE}: notes", "",
             "Generated by figures/mechanism.py from results/mechanism/internals/ (experiments/mechanism/internals.py, "
             "by_depth). Not part of the figure.", "",
             "## What the figure shows", "",
             "- For each plain loop, the accuracy after every pass of: a linear probe of the answer (the model's colour) "
             "and the model's own output (grey). Light line and filled circles: 1 hop; dark line and hollow squares: 3 "
             "hops. Final checkpoints.",
             "- Layout: the two runs that compute early and read late on top; the seed-41 sandwich loop centred below, "
             "its title starred for the post's caption. It learned a different solution (no clock and no 6-hop "
             "accuracy; clocks.md and mechanism_seeds.md), and its probe after pass 1 is only "
             + " / ".join(f"{d[h]['probe'][0]:.2f}" for c, _, s, d in rows if (c, s) == EARLY_PANELS[-1]
                          for h in EARLY_HOPS)
             + " (1 / 3 hops), so it computes late too, though still before its output.",
             "- Both are measured on the same thing: the model is stopped after pass k and the probe reads the "
             "residual at `=` that the unembedding reads (after the coda for the Huginn-style loop). Probe: least "
             "squares onto the one-hot answer, fitted on a random half of the chains; both accuracies are scored on "
             "the other half.",
             f"- Chains: 1 hop is the full grid (every a + b mod 113); 3 hops is {113 ** 2} random chains.",
             "- Left out of the figure: the Huginn-style loop. Its probe and output are both at 0.97-1.00 from pass 1 "
             "at both seeds, but not because its loop reads out faster: its prelude and coda compute and read the "
             "answer, and it barely uses the loop (Inside the Huginn-style loop and For the text, below). A panel "
             "would show no lag for a reason the figure can't show. Its numbers are in the tables below.",
             "- Left out of the figure: the one-layer model, which has one pass, so there's no lag to show. Its "
             "numbers are in the full table below.",
             "- The state-norm loop didn't grok at seed 42, so it has no panel.", "",
             "## Chance levels", "",
             f"- Model's output: 1 / 113 = {1 / 113:.3f}, a uniform guess. The figure's dotted \"chance\" line.",
             "- Probe: about 1 / 113 as well for a residual that carries nothing about the answer (expected, not "
             "measured).", "",
             "## Full table", "",
             "| model | seed | " + " | ".join(f"{hop_name(h)}: {c}" for h in EARLY_HOPS for c in
                                               ("probe after pass 1", "output after pass 1", "probe from pass",
                                                "output from pass")) + " |",
             "|---|---|" + "---|" * 4 * len(EARLY_HOPS)]
    for _, label, seed, d in rows:
        cells = []
        for h in EARLY_HOPS:
            cells += [f"{d[h]['probe'][0]:.2f}", f"{d[h]['readout'][0]:.2f}", fmt(first_pass(d[h]["probe"])),
                      fmt(first_pass(d[h]["readout"]))]
        lines.append(f"| {label} | {seed} | " + " | ".join(cells) + " |")
    depth = max(len(d[EARLY_HOPS[0]]["probe"]) for *_, d in rows)
    lines += ["", "## After every pass (probe / output)", "",
              "| model | seed | hops | " + " | ".join(str(k) for k in range(1, depth + 1)) + " |",
              "|---|---|---|" + "---|" * depth]
    for _, label, seed, d in rows:
        if len(d[EARLY_HOPS[0]]["probe"]) == 1:
            continue
        for h in d:
            lines.append(f"| {label} | {seed} | {h} | "
                         + " | ".join(f"{p:.2f} / {o:.2f}" for p, o in zip(d[h]["probe"], d[h]["readout"])) + " |")
    huginn = TWOPHASE[2][0]
    lines += ["", "## Inside the Huginn-style loop", "",
              "The probe after each block, with the model run to its training depth (internals.py, grid and chains). "
              "The prelude computes most of the answer; the coda finishes it.", "",
              "| seed | hops | prelude | core pass 1 | core pass 8 | coda |", "|---|---|---|---|---|---|"]
    for seed in SEEDS:
        d = load(mech, "internals", name(huginn, seed))
        for h in EARLY_HOPS:
            s = d["grid"]["sum"] if h == "1" else d["chains"][h]["sum"]
            lines.append(f"| {seed} | {h} | " + " | ".join(f"{s[b]['decode']:.2f}" for b in
                                                           ("prelude", "core pass 1", f"core pass {d['T']}", "coda"))
                         + " |")
    lines += ["", "## For the text", "",
              "- The plain loops: the answer is decodable passes before the model outputs it, at every seed. After "
              "pass 1 the probe already reads 0.92-0.93 at 1 hop where the output is at chance (except the seed-41 "
              "sandwich loop, which learned a different solution and computes later).",
              f"- With more hops the probe gets there later but the output doesn't. First pass at ≥ {READ_AT}, 1 → 3 "
              "hops: probe " + ", ".join(f"{fmt(first_pass(d['1']['probe']))} → {fmt(first_pass(d['3']['probe']))}"
                                         for c, *_, d in rows if c in EARLY_SHADES)
              + "; output " + ", ".join(f"{fmt(first_pass(d['1']['readout']))} → "
                                        f"{fmt(first_pass(d['3']['readout']))}"
                                        for c, *_, d in rows if c in EARLY_SHADES)
              + " (in order: " + "; ".join(f"{label}, seed {seed}" for c, label, seed, _ in rows
                                           if c in EARLY_SHADES) + "). "
              "So the computation takes longer with more hops, while the readout comes at about the same pass, near "
              f"the training depth ({TRAIN_DEPTH}).",
              "- The Huginn-style loop has no lag, but not because its loop reads out faster: its prelude and coda "
              "compute and read the answer (Inside the Huginn-style loop, above), and at these checkpoints it barely "
              "uses the loop: accuracy with noise in place of the carried state is "
              + " / ".join(f"{measure(mech, huginn, s, 'noise')[0]:.2f}" for s in SEEDS) + ", and W_A's state half is "
              + " / ".join(f"{measure(mech, huginn, s, 'blocks')[0]:.2f}" for s in SEEDS)
              + " (seeds 41 / 42; state-use.py).",
              f"- The one-layer model's 3-hop probe is 0.89-0.90, just under {READ_AT}, so its \"probe from pass\" "
              "is — in the full table: a threshold effect, not a difference."]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("\n".join(lines) + "\n")
    return out


def t_star(acc, frac=0.95):
    """the first pass reaching frac of the best accuracy.

    args:
        acc: per-pass accuracies.
        frac: fraction of the best.
    returns:
        1-based pass, or nan when the best is near chance.
    """
    best = max(acc)
    if best < 0.1:
        return float("nan")
    return next(i + 1 for i, a in enumerate(acc) if a >= frac * best)


def final_state_row(mech, config, seed):
    """the last checkpoint row of a run's state-use output.

    args:
        mech: folder holding the analysis outputs.
        config: config name.
        seed: seed.
    returns:
        dict for one checkpoint.
    """
    return load(mech, state_use_kind(config), name(config, seed))["checkpoints"][-1]


def measure(mech, config, seed, what):
    """the numbers of one seed-table row for one run.

    args:
        mech: folder holding the analysis outputs.
        config: config name.
        seed: seed.
        what: passes (passes to 95% of the best accuracy, chains of 1, 3 and 6 hops), depth (accuracy at the
            training depth and at pass 256), nudge (size of the 1% nudge at pass 256), excluded (loss with the top
            5, 10, 20, 40 frequencies removed), clock (clock-fit r² with 5, 20, 56 frequencies), noise (accuracy with
            noise as the state), blocks (state and input halves of W_A), distinct (distinct answers among 2-hop
            chains 1, 8 and 56 passes after the injected input is zeroed) or step (the probed checkpoint).
    returns:
        list of numbers.
    """
    run = name(config, seed)
    if what == "passes":
        d = load(mech, "passes" if seed == SEEDS[0] else "passes-s42", run)
        return [t_star(d["by_hop"][h]["acc"]) for h in ("1", "3", "6")]
    if what in ("depth", "nudge"):
        d = load(mech, past_depth_kind(config, seed), run)
        if what == "nudge":
            return [float(np.mean([d["by_hop"][h]["pert"][-1] for h in TRAIN_HOPS]))]
        acc = np.mean([d["by_hop"][h]["acc"] for h in TRAIN_HOPS], axis=0)
        return [acc[TRAIN_DEPTH - 1], acc[-1]]
    if what in ("excluded", "clock"):
        d = load(mech, "frequencies", run)
        at = {k: i for i, k in enumerate(d["ks"])}
        if what == "excluded":
            return [d["excluded_loss"][at[k]] for k in (5, 10, 20, 40)]
        return [d["clock_r2"][at[k]] for k in (5, 20, 56)]
    row = final_state_row(mech, config, seed)
    if what == "noise":
        return [row["noise"]]
    if what == "blocks":
        return [row["blocks"]["state"]["spectral"], row["blocks"]["input"]["spectral"]]
    if what == "distinct":
        v = row["by_hop"]["2"]["inject"]["zero_input"]
        return [v[a]["distinct"] for a in AFTER]
    return [row["step"]]


# sections of the seed table: (config, heading, [(row label, what to measure, number format)])
TABLE = [
    (TWOPHASE[2][0], "Huginn, two-phase recipe", [
        ("passes to 95% of the best accuracy, chains of 1 / 3 / 6 hops", "passes", "{:.0f}"),
        ("accuracy at pass 8 / 256", "depth", "{:.2f}"),
        ("1% nudge after pass 8, size at pass 256", "nudge", "{:.0e}"),
        ("loss with the top 5 / 10 / 20 / 40 frequencies removed", "excluded", "{:.2f}"),
        ("clock r² with 5 / 20 / 56 frequencies", "clock", "{:.2f}"),
        ("accuracy with noise as the state", "noise", "{:.2f}"),
        ("W_A state / input half, spectral norm", "blocks", "{:.2f}"),
        ("distinct answers, 2-hop chains, 1 / 8 / 56 passes after the input is zeroed", "distinct", "{:.0f}"),
    ]),
    (TWOPHASE[0][0], "plain loop with a state norm, two-phase recipe", [
        ("passes to 95% of the best accuracy, chains of 1 / 3 / 6 hops", "passes", "{:.0f}"),
        ("accuracy at pass 8 / 256", "depth", "{:.2f}"),
        ("1% nudge after pass 8, size at pass 256", "nudge", "{:.0e}"),
        ("loss with the top 5 / 10 / 20 / 40 frequencies removed", "excluded", "{:.2f}"),
        ("clock r² with 5 / 20 / 56 frequencies", "clock", "{:.2f}"),
    ]),
    (TWOPHASE[1][0], "plain loop with sandwich norms, two-phase recipe", [
        ("passes to 95% of the best accuracy, chains of 1 / 3 / 6 hops", "passes", "{:.0f}"),
        ("accuracy at pass 8 / 256", "depth", "{:.2f}"),
        ("1% nudge after pass 8, size at pass 256", "nudge", "{:.0e}"),
        ("loss with the top 5 / 10 / 20 / 40 frequencies removed", "excluded", "{:.2f}"),
        ("clock r² with 5 / 20 / 56 frequencies", "clock", "{:.2f}"),
    ]),
    (FIXED[0], "Huginn at a fixed depth of 8, Nanda's recipe", [
        ("accuracy at pass 8 / 256", "depth", "{:.2f}"),
        ("1% nudge after pass 8, size at pass 256", "nudge", "{:.0e}"),
        ("accuracy with noise as the state", "noise", "{:.2f}"),
        ("W_A state / input half, spectral norm", "blocks", "{:.2f}"),
        ("distinct answers, 2-hop chains, 1 / 8 / 56 passes after the input is zeroed", "distinct", "{:.0f}"),
    ]),
    (ANCHOR[0], "one-layer model, Nanda's recipe", [
        ("loss with the top 5 / 10 / 20 / 40 frequencies removed", "excluded", "{:.2f}"),
        ("clock r² with 5 / 20 / 56 frequencies", "clock", "{:.2f}"),
    ]),
    (REFERENCE, "Huginn reference, Nanda's recipe, at its most accurate checkpoint", [
        ("checkpoint step", "step", "{:.0f}"),
        ("accuracy with noise as the state", "noise", "{:.2f}"),
    ]),
]


def cell(values, fmt):
    """values joined with slashes.

    args:
        values: list of numbers.
        fmt: format string for one number.
    returns:
        string.
    """
    return " / ".join("—" if v != v else fmt.format(v) for v in values)


def write_seed_table(mech, root, out):
    """write the seed comparison table as markdown.

    args:
        mech: folder holding the analysis outputs.
        root: folder holding run directories.
        out: output path.
    returns:
        path to the written file.
    """
    lines = ["# mechanism results at two seeds", "",
             "each value at seed 41 and seed 42; the mean only where both runs grokked and were still grokked at "
             f"their last evaluation (mean val accuracy ≥ {GROKKED}). the reference huginn is read at its most "
             "accurate checkpoint instead. — in the passes row: the chain length stays near chance at every pass.", ""]
    for config, heading, rows in TABLE:
        ok, head, why = {}, [], {}
        for seed in SEEDS:
            t_grok, acc = status(root, name(config, seed))
            if config == REFERENCE:
                acc = final_state_row(mech, config, seed)["normal"]
                ok[seed] = acc >= GROKKED
            else:
                ok[seed] = t_grok is not None and acc >= GROKKED
            why[seed] = "not grokked" if t_grok is None else "didn't stay grokked"
            if t_grok is None:
                head.append(f"{acc:.2f}, didn't grok")
            else:
                head.append(f"{acc:.2f}, grokked at {t_grok}" + ("" if ok[seed] else " but didn't stay grokked"))
        lines += [f"## {heading}", "", "| | seed 41 | seed 42 | mean |", "|---|---|---|---|",
                  f"| val accuracy at the analysed checkpoint | {head[0]} | {head[1]} | |"]
        for label, what, fmt in rows:
            vals = {s: measure(mech, config, s, what) if ok[s] or s == SEEDS[0] else None for s in SEEDS}
            cells = [cell(vals[s], fmt) if vals[s] is not None else why[s] for s in SEEDS]
            # a mean of two checkpoint steps means nothing
            both = what != "step" and all(ok.values()) and all(vals[s] is not None for s in SEEDS)
            mean = cell(list(np.mean([vals[s] for s in SEEDS], axis=0)), fmt) if both else "—"
            lines.append(f"| {label} | {cells[0]} | {cells[1]} | {mean} |")
        lines.append("")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("\n".join(lines))
    return out


def main(argv=None):
    """run the cli.

    args:
        argv: argument list, or None for sys.argv.
    returns:
        exit code: 0 ok.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mech", default=str(cm.RESULTS / "mechanism"))
    ap.add_argument("--root", default=str(cm.RESULTS / "runs"))
    ap.add_argument("--out", default=str(cm.OUT / "mechanism"))
    ap.add_argument("--supporting", action="store_true", help="also draw the figures that aren't in the write-up")
    args = ap.parse_args(argv)
    print(*plot_attention(args.mech, f"{args.out}/attention.png"), sep="\n")
    print(*plot_clocks(args.mech, args.root, f"{args.out}/clocks.png"), sep="\n")
    print(*write_compute_early(args.mech, f"{args.out}/compute_early.png"), sep="\n")
    print(write_seed_table(args.mech, args.root, f"{args.out}/mechanism_seeds.md"))
    if args.supporting:
        sup = f"{args.out}/supporting"
        print(plot_passes(args.mech, f"{sup}/passes.png"))
        print(plot_past_depth(args.mech, args.root, f"{sup}/past_depth.png"))
        print(plot_frequencies(args.mech, args.root, f"{sup}/frequencies.png"))
        print(plot_state_use(args.mech, args.root, f"{sup}/state_use.png"))
        print(plot_dilution(args.mech, args.root, f"{sup}/dilution.png"))
        print(plot_decoding(args.mech, f"{sup}/decoding.png"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
