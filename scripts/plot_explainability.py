#!/usr/bin/env python3
"""Figure for the territory-stratified explainability run.

Reads data/acs_final/explainability/explainability_results.json and draws, per injected
territory (columns):
  row 1  ACS and ACCO probability against injected dose
  row 2  the 3-way territory softmax against dose (matched rises, mismatched fall)
  row 3  negative control: a rectangular ST offset, ST raised with the T wave untouched
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BASE = "/volume/DeepECG-SSL-finetune/data/acs_final/explainability"
CLASSES = ["LEFT", "RCA", "LCX"]
# validated categorical palette (scripts/validate_palette.js, light surface): all checks pass
COLOR = {"LEFT": "#2a78d6", "RCA": "#eb6834", "LCX": "#1baf7a"}
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
LABEL = {"LEFT": "LEFT (LAD or Left Main)", "RCA": "RCA", "LCX": "LCx"}


def style(ax):
    ax.set_facecolor("#fcfcfb")
    ax.grid(True, color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
        ax.spines[s].set_linewidth(1.0)
    ax.tick_params(colors=MUTED, labelsize=8, length=3)


def main():
    R = json.load(open(f"{BASE}/explainability_results.json"))
    doses = R["doses"]
    fig, axes = plt.subplots(3, 3, figsize=(12.0, 9.2), sharex="row")
    fig.patch.set_facecolor("#fcfcfb")

    for col, c in enumerate(CLASSES):
        d = R["dose_response"][c]

        # ---- row 1: detection heads
        ax = axes[0, col]
        style(ax)
        for key, color, lab in (("acco", INK, "P(complete occlusion)"),
                                ("acs", INK2, "P(ACS)")):
            y = d[key]
            sd = d[key + "_sd"]
            ax.plot(doses, y, color=color, lw=2.0, marker="o", ms=5, zorder=3, label=lab)
            ax.fill_between(doses, [a - b for a, b in zip(y, sd)],
                            [a + b for a, b in zip(y, sd)], color=color, alpha=0.13, lw=0, zorder=2)
        ax.axhline(R["gate"], color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=1)
        ax.text(doses[-1], R["gate"], " gate", color=MUTED, fontsize=7, va="bottom", ha="right")
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(f"inject {LABEL[c]} ischemic pattern", fontsize=10, color=INK, pad=8)
        if col == 0:
            ax.set_ylabel("probability", fontsize=9, color=INK2)
            ax.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=INK2)

        # ---- row 2: territory softmax
        ax = axes[1, col]
        style(ax)
        ends = sorted((d["territory"][cl][-1], cl) for cl in CLASSES)   # ascending
        for cl in CLASSES:
            y = d["territory"][cl]
            sd = d["territory_sd"][cl]
            ax.plot(doses, y, color=COLOR[cl], lw=2.0, marker="o", ms=5,
                    zorder=4 if cl == c else 3, label=LABEL[cl])
            ax.fill_between(doses, [a - b for a, b in zip(y, sd)],
                            [a + b for a, b in zip(y, sd)],
                            color=COLOR[cl], alpha=0.13, lw=0, zorder=2)
        prev = -1.0
        for val, cl in ends:              # push colliding labels upward, never out of frame
            pos = max(val, prev + 0.06)
            prev = pos
            ax.annotate(cl if cl != "LCX" else "LCx", (doses[-1], pos),
                        textcoords="offset points", xytext=(8, 0), color=COLOR[cl],
                        fontsize=8, va="center", fontweight="bold")
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlim(doses[0] - 0.1, doses[-1] + 0.55)
        if col == 0:
            ax.set_ylabel("territory softmax", fontsize=9, color=INK2)
            ax.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=INK2)

        # ---- row 3: negative control
        ax = axes[2, col]
        style(ax)
        nc = R["negative_control"][c]
        ax.plot(nc["mv"], nc["acco"], color=INK, lw=2.0, marker="s", ms=5, zorder=3,
                label="rectangular ST offset")
        st1 = R["pattern"][c].get("delivered_ST_at_s1_mV")
        if st1:
            j = min(range(len(doses)), key=lambda k: abs(doses[k] - 1.0))
            ax.plot([abs(st1)], [d["acco"][j]], marker="o", ms=7, color=COLOR[c], zorder=4)
            ax.annotate("real ischemic pattern\nat the same ST", (abs(st1), d["acco"][j]),
                        textcoords="offset points", xytext=(8, -4), fontsize=7.5, color=COLOR[c])
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel("ST offset in territory leads (mV)", fontsize=9, color=INK2)
        if col == 0:
            ax.set_ylabel("P(complete occlusion)", fontsize=9, color=INK2)
            ax.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=INK2)

    for col in range(3):
        axes[1, col].set_xlabel("dose s  (s = 1 is the average real occlusion)",
                                fontsize=9, color=INK2)
        axes[0, col].set_xlabel("dose s", fontsize=9, color=INK2)

    fig.suptitle("Territory-specific response to real ischemic morphology, generator-free",
                 fontsize=12, color=INK, x=0.5, y=0.985)
    fig.text(0.5, 0.005,
             f"Mean of {len(R['seeds'])} samples of {R['n_hosts']} real normal held-out ECGs; "
             "band = SD across samples. Pattern = mean occluded beat − mean normal beat, "
             "measured on the training split and injected at each R peak.",
             ha="center", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    out = os.path.join(BASE, "territory_doseresponse.png")
    fig.savefig(out, dpi=200, facecolor=fig.get_facecolor())
    print(f"-> {out}")


if __name__ == "__main__":
    main()
