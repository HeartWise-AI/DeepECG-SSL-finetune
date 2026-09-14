#!/usr/bin/env python3
"""Figures for the conditional generator: sampled ACO-positive beats per territory.

Figure 1 (one per territory): the 12-lead mean of sampled ACO-positive beats against the mean of
sampled normal beats, with the mean real held-out occluded beat as reference and a few individual
samples drawn faintly so the reader sees diversity rather than only an average.

Figure 2: the decisive comparison. Per-lead ST change for real occlusion, for the conditional
sample, and for the gradient morph, side by side. Sampling and morphing are two answers to
"what does an occlusion in this territory look like to this model", and only one of them is
supposed to recover the real morphology.
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BASE = "/volume/DeepECG-SSL-finetune/data/acs_final/explainability"
TERR = ["LEFT", "RCA", "LCX"]
NICE = {"LEFT": "LEFT (LAD or Left Main)", "RCA": "RCA", "LCX": "LCx"}
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
ROWS, COLS = 4, 3
FS, PRE = 250, 70
COLOR = {"LEFT": "#2a78d6", "RCA": "#eb6834", "LCX": "#1baf7a"}
NEG, REAL, MORPH, SAMP = "#3d4450", "#8f979d", "#a6401f", "#2a78d6"
INK, INK2, MUTED, GRID, AXIS = "#14181c", "#454e55", "#7f878e", "#e3e4e0", "#c2c8c6"
ST0, ST1 = PRE + int(0.08 * FS), PRE + int(0.16 * FS)
T0, T1 = PRE + int(0.16 * FS), PRE + int(0.40 * FS)


def style(ax):
    ax.set_facecolor("#fcfcfb")
    ax.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=7, length=2)


def st_of(b):
    return b[..., ST0:ST1].mean(-1)


def main():
    d = np.load(f"{BASE}/conditional_samples.npz")
    S = json.load(open(f"{BASE}/conditional_samples.json"))
    w = float(d["guidance"])
    x, y = d["x"], d["y"]
    t = np.arange(x.shape[-1]) / FS * 1000 - PRE / FS * 1000
    norm_mean = x[y == 0].mean(0)
    cos = S[str(w)]["cosine_vs_real"]

    for i, c in enumerate(TERR):
        s = x[y == i + 1]
        real = d[f"real_{c}_mean"]
        fig, axes = plt.subplots(ROWS, COLS, figsize=(12.4, 8.0), sharex=True)
        fig.patch.set_facecolor("#fcfcfb")
        for j, lead in enumerate(LEADS):
            ax = axes[j % ROWS, j // ROWS]
            style(ax)
            for k in range(6):
                ax.plot(t, s[k, j], color=COLOR[c], lw=.6, alpha=.30, zorder=2,
                        label="individual samples" if (j == 0 and k == 0) else None)
            ax.plot(t, real[j], color=REAL, lw=2.4, alpha=.6, zorder=3,
                    label="mean real occluded beat" if j == 0 else None)
            ax.plot(t, norm_mean[j], color=NEG, lw=1.7, zorder=4,
                    label="ACO negative (sampled normal)" if j == 0 else None)
            ax.plot(t, s.mean(0)[j], color=COLOR[c], lw=1.9, zorder=5,
                    label="ACO positive (sampled)" if j == 0 else None)
            ax.text(.015, .88, lead, transform=ax.transAxes, fontsize=9, color=INK,
                    fontweight="bold")
            if j % ROWS == ROWS - 1:
                ax.set_xlabel("ms from R peak", fontsize=8.5, color=INK2)
            if j // ROWS == 0:
                ax.set_ylabel("mV", fontsize=8.5, color=INK2)
        h, lab = axes[0, 0].get_legend_handles_labels()
        fig.legend(h, lab, frameon=False, fontsize=8.5, ncol=4, loc="upper center",
                   bbox_to_anchor=(0.5, 0.955), labelcolor=INK2)
        fig.suptitle(f"Generated ACO-positive beats, {NICE[c]}", fontsize=12.5, color=INK, y=.985)
        r = S[str(w)]
        fig.text(.5, .005,
                 f"{len(s)} samples from P(beat | {c}), classifier-free guidance {w:g}. "
                 f"Classifier reads them at P(occlusion) {r['acco_'+c]:.2f} and calls the "
                 f"territory correctly in {100*r['territory_top1']:.0f}% of samples overall. "
                 f"Cosine against the real ischemic change {cos[c]:+.2f}.",
                 ha="center", fontsize=7.5, color=MUTED)
        fig.tight_layout(rect=[0, .02, 1, .925])
        out = f"{BASE}/sampled_{c.lower()}.png"
        fig.savefig(out, dpi=190, facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"-> {out}")

    # ---------- sampling vs morphing vs real
    m = np.load(f"{BASE}/morphs.npz")
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2))
    fig.patch.set_facecolor("#fcfcfb")
    xi = np.arange(12)
    for k, c in enumerate(TERR):
        ax = axes[k]
        style(ax)
        ax.axhline(0, color=AXIS, lw=1)
        real_d = st_of(d[f"real_{c}_mean"]) - st_of(d["real_norm_mean"])
        samp_d = st_of(x[y == k + 1].mean(0)) - st_of(norm_mean)
        morph_d = st_of(m[f"{c}_tfinal"].mean(0)) - st_of(m[f"{c}_t0"].mean(0))
        ax.bar(xi - .27, real_d, .27, color=REAL, label="real occlusion" if k == 0 else None)
        ax.bar(xi, samp_d, .27, color=SAMP, label="conditional sample" if k == 0 else None)
        ax.bar(xi + .27, morph_d, .27, color=MORPH, label="gradient morph" if k == 0 else None)
        ax.set_xticks(xi)
        ax.set_xticklabels(LEADS, fontsize=7.5, rotation=45)
        if k == 0:
            ax.set_ylabel("Δ ST level (mV)", fontsize=9, color=INK2)
        ax.set_title(f"{NICE[c]}", fontsize=10.5, color=INK, pad=7)
    axes[0].legend(frameon=False, fontsize=8.5, loc="upper left", labelcolor=INK2)
    fig.suptitle("Sampling a territory versus morphing toward it, against real ischemia",
                 fontsize=12.5, color=INK, y=.99)
    fig.text(.5, .005, "Per-lead ST change relative to the matching normal reference: real "
             "held-out occluded beats, beats sampled from the conditional generator, and "
             "gradient morphs of normal beats. Colour encodes method, not territory.",
             ha="center", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=[0, .03, 1, .93])
    out = f"{BASE}/sampled_vs_morph.png"
    fig.savefig(out, dpi=190, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
