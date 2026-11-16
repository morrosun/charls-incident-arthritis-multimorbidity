# -*- coding: utf-8 -*-
"""Figure 6: four-cohort external validation forest plot (two panels)"""
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.environ.get("W6_OUT", os.path.join(ROOT, "out"))
os.makedirs(OUT, exist_ok=True)

import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

for f in ["Arial", "Helvetica", "DejaVu Sans"]:
    try:
        font_manager.findfont(f, fallback_to_default=False)
        plt.rcParams["font.family"] = f
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

RED = "#9B2335"; GREY = "#7B8794"; INK = "#26303B"

R = pd.read_csv(os.path.join(OUT, "ext_models.csv"))
M = pd.read_csv(os.path.join(OUT, "ext_meta.csv"))
R = R[R["sample"] == "all transitions"]

PANELS = [
    ("Metabolic comorbidity (hypertension or diabetes)",
     ["HRS", "ELSA", "MHAS", "CHARLS"],
     "Metabolic comorbidity (hypertension or diabetes)"),
    ("Body mass index, per 5 kg/m2",
     ["HRS", "ELSA", "CHARLS"],
     "Body mass index, per 5 kg/m$^2$ (measured)"),
]
LAB = {"HRS": "HRS (United States)", "ELSA": "ELSA (England)",
       "MHAS": "MHAS (Mexico)", "CHARLS": "CHARLS (China)"}

fig = plt.figure(figsize=(11.6, 3.5), dpi=300)
gs = fig.add_gridspec(1, 6, width_ratios=[0.46, 1, 0.30, 0.46, 1, 0.30], wspace=0.06,
                      left=0.01, right=0.99, top=0.86, bottom=0.22)

for k, (lab, cohorts, title) in enumerate(PANELS):
    base = 3 * k
    lx = fig.add_subplot(gs[0, base], sharey=None)
    lx.axis("off")
    ax = fig.add_subplot(gs[0, base + 1])
    tx = fig.add_subplot(gs[0, base + 2], sharey=ax)
    tx.axis("off")
    sub = R[(R.exposure_label == lab) & (R.cohort.isin(cohorts))].set_index("cohort").reindex(cohorts)
    meta = M[M.exposure_label == lab].iloc[0]
    names = [LAB[c] for c in cohorts] + ["Pooled (random effects)"]
    rr = list(sub.RR.values) + [meta.RR]
    lo = list(sub.lo.values) + [meta.lo]
    hi = list(sub.hi.values) + [meta.hi]
    y = np.arange(len(names))[::-1]

    for i in range(len(names)):
        last = (i == len(names) - 1)
        ax.plot([lo[i], hi[i]], [y[i], y[i]], lw=(1.5 if last else 1.1),
                color=(RED if last else "#4A5665"), solid_capstyle="butt", zorder=2)
        if last:
            ax.plot([lo[i], hi[i]], [y[i], y[i]], lw=4.2, color=RED, alpha=0.28,
                    solid_capstyle="butt", zorder=1)
            ax.plot(rr[i], y[i], "D", ms=6.0, color=RED, zorder=3)
        else:
            ax.plot(rr[i], y[i], "o", ms=5.0, color=INK, zorder=3)

    ax.axvline(1, color=GREY, lw=0.8, ls="--", zorder=0)
    ax.set_yticks([])
    ax.set_ylim(-0.75, len(names) - 0.25)
    lx.set_ylim(ax.get_ylim())
    lx.set_xlim(0, 1)
    for i in range(len(names)):
        last = (i == len(names) - 1)
        lx.text(0.985, y[i], names[i], va="center", ha="right", fontsize=8.2,
                color=INK, fontweight=("bold" if last else "normal"))

    ax.set_xscale("log")
    ax.set_xlim(0.82, 1.52)
    ax.set_xticks([0.85, 1.0, 1.2, 1.4])
    ax.set_xticklabels(["0.85", "1.0", "1.2", "1.4"], fontsize=8.0)
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("Risk ratio (95% CI, log scale)", fontsize=8.4, labelpad=4)
    ax.set_title(title, fontsize=9.4, color=INK, pad=9, loc="left")

    # numeric column in its own axis, so it can never collide with the next panel
    tx.set_ylim(ax.get_ylim())
    for i in range(len(names)):
        last = (i == len(names) - 1)
        tx.text(0.0, y[i], "%.2f (%.2f\u2013%.2f)" % (rr[i], lo[i], hi[i]),
                va="center", ha="left", fontsize=7.6,
                color=(RED if last else INK),
                fontweight=("bold" if last else "normal"))

    ax.text(0.0, -0.30, "I$^2$ = %.0f%%" % meta.I2, transform=ax.transAxes,
            ha="left", va="top", fontsize=7.6, color=GREY)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=3, color="#9AA6B2")

plt.savefig(os.path.join(OUT, "fig6_external_en.png"), bbox_inches="tight", facecolor="white")
plt.close()
print("saved out/fig6_external_en.png")
