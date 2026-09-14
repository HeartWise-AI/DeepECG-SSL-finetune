#!/usr/bin/env python3
"""Figures for the territory morphs: ACO-negative vs ACO-positive counterfactual beats.

Per territory, a standard 3x4 twelve-lead panel showing
  - ACO negative: the decoded real normal beat (t0)
  - ACO positive: the same beat after morphing to the 90th-percentile occlusion risk (tfinal)
  - reference   : the mean real held-out occluded beat of that territory

t0 and tfinal are both decoder outputs, as in the source paper's export, so reconstruction error
cancels and the difference is the morph alone.

Also draws a validation panel: the morph's per-lead ST and T change against the measured real
ischemic change for the same territory.
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BASE = "/volume/DeepECG-SSL-finetune/data/acs_final/explainability"
CLASSES = ["LEFT", "RCA", "LCX"]
NICE = {"LEFT": "LEFT (LAD or Left Main)", "RCA": "RCA", "LCX": "LCx"}
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
ROWS, COLS = 4, 3
FS, PRE = 250, 70
# validated categorical palette (dataviz validator, light surface)
COLOR = {"LEFT": "#2a78d6", "RCA": "#eb6834", "LCX": "#1baf7a"}
NEG, REAL = "#3d4450", "#8f979d"
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


def st_t(beats):
    """(N,12,L) -> per-lead ST level and T amplitude."""
    return beats[..., ST0:ST1].mean(-1), beats[..., T0:T1].max(-1)


def main():
    d = np.load(f"{BASE}/morphs.npz")
    S = json.load(open(f"{BASE}/morph_summary.json"))
    t = np.arange(d[f"{CLASSES[0]}_t0"].shape[-1]) / FS * 1000 - PRE / FS * 1000

    for c in CLASSES:
        t0 = d[f"{c}_t0"]
        tf = d[f"{c}_tfinal"]
        real = d[f"{c}_real_mean"]
        done = d[f"{c}_done"].astype(bool)
        gain = d[f"{c}_p_terr1"] - d[f"{c}_p_terr0"]
        pick = int(np.argsort(-gain)[0]) if not done.any() else int(
            np.arange(len(done))[done][np.argsort(-gain[done])[0]])

        fig, axes = plt.subplots(ROWS, COLS, figsize=(12.4, 8.0), sharex=True)
        fig.patch.set_facecolor("#fcfcfb")
        for j, lead in enumerate(LEADS):
            ax = axes[j % ROWS, j // ROWS]
            style(ax)
            ax.plot(t, real[j], color=REAL, lw=2.4, alpha=.55, zorder=2,
                    label="mean real occluded beat" if j == 0 else None)
            ax.plot(t, t0[pick, j], color=NEG, lw=1.7, zorder=3,
                    label="ACO negative (real normal beat)" if j == 0 else None)
            ax.plot(t, tf[pick, j], color=COLOR[c], lw=1.7, zorder=4,
                    label="ACO positive (morph)" if j == 0 else None)
            ax.text(.015, .88, lead, transform=ax.transAxes, fontsize=9, color=INK,
                    fontweight="bold")
            if j % ROWS == ROWS - 1:
                ax.set_xlabel("ms from R peak", fontsize=8.5, color=INK2)
            if j // ROWS == 0:
                ax.set_ylabel("mV", fontsize=8.5, color=INK2)
        h, lab = axes[0, 0].get_legend_handles_labels()
        fig.legend(h, lab, frameon=False, fontsize=8.5, ncol=3, loc="upper center",
                   bbox_to_anchor=(0.5, 0.955), labelcolor=INK2)
        fig.suptitle(f"Counterfactual beat morph toward complete occlusion, {NICE[c]}",
                     fontsize=12.5, color=INK, y=.985)
        fig.text(.5, .005,
                 f"One held-out normal beat (patient record {int(d[f'{c}_host_rec'][pick])}). "
                 f"P(occlusion) {d[f'{c}_p_acco0'][pick]:.2f} to {d[f'{c}_p_acco1'][pick]:.2f}; "
                 f"P({c}) {d[f'{c}_p_terr0'][pick]:.2f} to {d[f'{c}_p_terr1'][pick]:.2f}; "
                 f"{int(d[f'{c}_steps'][pick])} gradient steps. Both traces are decoder outputs, "
                 "so reconstruction error cancels.",
                 ha="center", fontsize=7.5, color=MUTED)
        fig.tight_layout(rect=[0, .02, 1, .925])
        out = f"{BASE}/morph_{c.lower()}.png"
        fig.savefig(out, dpi=190, facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"-> {out}")

    # ---------- validation panel: morph change vs real ischemic change, per lead
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 6.4), sharex=True)
    fig.patch.set_facecolor("#fcfcfb")
    x = np.arange(12)
    cos = {}
    for k, c in enumerate(CLASSES):
        t0, tf, real = d[f"{c}_t0"], d[f"{c}_tfinal"], d[f"{c}_real_mean"]
        st_m, tt_m = st_t(tf.mean(0))
        st_0, tt_0 = st_t(t0.mean(0))
        st_r, tt_r = st_t(real)
        st_n, tt_n = st_t(t0.mean(0))
        d_morph = np.concatenate([st_m - st_0, tt_m - tt_0])
        d_real = np.concatenate([st_r - st_n, tt_r - tt_n])
        cos[c] = float(d_morph @ d_real / (np.linalg.norm(d_morph) * np.linalg.norm(d_real)))
        for row, (m, r, nm) in enumerate(((st_m - st_0, st_r - st_n, "ST level"),
                                          (tt_m - tt_0, tt_r - tt_n, "T amplitude"))):
            ax = axes[row, k]
            style(ax)
            ax.axhline(0, color=AXIS, lw=1)
            ax.bar(x - .2, r, .4, color=REAL, label="real occlusion" if row == 0 and k == 0 else None)
            ax.bar(x + .2, m, .4, color=COLOR[c], label="morph" if row == 0 and k == 0 else None)
            ax.set_xticks(x)
            ax.set_xticklabels(LEADS, fontsize=7.5, rotation=45)
            if k == 0:
                ax.set_ylabel(f"Δ {nm} (mV)", fontsize=9, color=INK2)
            if row == 0:
                ax.set_title(f"{NICE[c]}   cosine {cos[c]:+.2f}", fontsize=10, color=INK, pad=7)
    axes[0, 0].legend(frameon=False, fontsize=8.5, loc="upper left", labelcolor=INK2)
    fig.suptitle("Does the morph reproduce real ischemic morphology, lead by lead?",
                 fontsize=12.5, color=INK, y=.985)
    fig.text(.5, .005, "Change from the decoded normal beat, averaged over all morphed hosts, "
             "against the mean real held-out occluded beat of the same territory. "
             "Cosine is over the 24-dimensional (ST, T) change vector.",
             ha="center", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=[0, .02, 1, .96])
    out = f"{BASE}/morph_validation.png"
    fig.savefig(out, dpi=190, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"-> {out}")
    json.dump({"cosine_morph_vs_real": cos}, open(f"{BASE}/morph_validation.json", "w"), indent=1)
    print("cosine(morph change, real change):", {k: round(v, 3) for k, v in cos.items()})
    for c in CLASSES:
        s = S[c]
        print(f"{c}: reached stop {s['n_reached_stop']}/{s['n_hosts']}, "
              f"travelled {100*s['frac_of_way']:.0f}% of the latent distance to a real beat")


if __name__ == "__main__":
    main()
