# -*- coding: utf-8 -*-
"""重出五张全英文图（Figure 1–5），使用已固定的结果文件"""
import json, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib import font_manager

# ---- 字体：优先 Arial / Helvetica ----
for f in ["Arial", "Helvetica", "DejaVu Sans"]:
    try:
        font_manager.findfont(f, fallback_to_default=False)
        plt.rcParams["font.family"] = f
        plt.rcParams["font.sans-serif"] = [f]
        print("font:", f)
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["pdf.fonttype"] = 42

RED = "#9B2335"
GREY = "#7B8794"
INK = "#26303B"
BLUE = "#2F5D8A"
PURPLE = "#6B4A7D"
LIGHT = "#E9EDF2"

# =========================================================
# Figure 1 — participant flow
# =========================================================
fig, ax = plt.subplots(figsize=(7.4, 8.2), dpi=300)
ax.set_xlim(0, 100); ax.set_ylim(11, 100); ax.axis("off")

def box(x, y, w, h, title, sub=None, fc="#ffffff", ec="#C8D1DB", lw=0.9,
        ts=9.6, ss=11.4, sc=RED, bold=True):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.35,rounding_size=1.0",
                                fc=fc, ec=ec, lw=lw, zorder=2))
    if sub:
        ax.text(x, y + h * 0.17, title, ha="center", va="center", fontsize=ts, color=INK, zorder=3)
        ax.text(x, y - h * 0.19, sub, ha="center", va="center", fontsize=ss, color=sc,
                fontweight="bold" if bold else "normal", zorder=3)
    else:
        ax.text(x, y, title, ha="center", va="center", fontsize=ts, color=INK, zorder=3)

def arrow(y1, y2, x=34):
    ax.add_patch(FancyArrowPatch((x, y1), (x, y2), arrowstyle="-|>", mutation_scale=9,
                                 lw=0.9, color="#8A97A6", shrinkA=0, shrinkB=0, zorder=1))

def side(y, text, xt=55.5):
    ax.add_patch(FancyArrowPatch((51.4, y), (xt - 1.4, y), arrowstyle="-", lw=0.7,
                                 color="#B8C3CF", linestyle=(0, (3, 2)), zorder=1))
    ax.text(xt, y, text, ha="left", va="center", fontsize=7.9, color="#4E5A67", zorder=3,
            linespacing=1.35,
            bbox=dict(boxstyle="round,pad=0.36", fc="#FAFBFC", ec="#E3E8EE", lw=0.7))

W = 33.5
box(34, 94.0, W, 9.2, "CHARLS 2021\u20132023 annual packages", "19,021 interview records")
arrow(89.4, 85.4)
box(34, 81.2, W, 8.0, "18,047 unique participants")
side(81.2, "974 duplicate records excluded\n(same interview, re-released;\n877 participants)")
arrow(77.2, 73.4)
box(34, 69.2, W, 8.0, "18,021 with determinable status")
side(69.2, "26 excluded\n(neither item answered)")
arrow(65.2, 61.4)
box(34, 57.0, W, 9.2, "At risk: free at previous interview", "12,062")
side(57.0, "5,959 prevalent at baseline\n(construct comparison only)")
arrow(52.4, 48.6)
box(34, 43.4, W, 10.2, "Analytic sample", "11,210 participants",
    fc="#FDF7F8", ec="#DCB0B7", lw=1.05, ss=12.2)
side(43.4, "852 excluded\n746 last interview not in 2020\n106 aged <45 or >95 years")

ax.add_patch(FancyArrowPatch((34, 38.3), (34, 33.0), arrowstyle="-", lw=0.7, color="#B8C3CF",
                             linestyle=(0, (3, 2)), zorder=1))
ax.text(34, 29.4, "Incident arthritis or rheumatism: 543 (4.84%)", ha="center", va="center",
        fontsize=10.0, color=RED, fontweight="bold")
ax.text(34, 23.6, "Follow-up 15,357 person-years  |  incidence density 3.54 per 100 person-years",
        ha="center", va="center", fontsize=8.4, color=INK)
ax.text(34, 17.6, "Interview batch: 2021, 8,113 participants / 372 events;  2022, 2,046 / 94;  2023, 1,051 / 77",
        ha="center", va="center", fontsize=7.4, color="#5D6B7A")

plt.tight_layout()
plt.savefig("out/fig1_flow_en.png", bbox_inches="tight", facecolor="white")
plt.close()

# =========================================================
# Figure 2 — four outcome definitions (construct comparison)
# =========================================================
C = pd.read_csv("out/construct_precov.csv")
order = ["Depressive symptoms (CES-D \u226510)", "Metabolic comorbidity, any", "Metabolic comorbidity, 2+",
         "Dyslipidaemia", "Hypertension", "Diabetes", "Sleep <6 h", "Sleep \u22658 h",
         "Restless sleep", "Loneliness"]
key = ["\u6291\u90c1\u75c7\u72b6\uff08CES-D\u226510\uff09", "\u4ee3\u8c22\u5171\u75c5\u22651", "\u4ee3\u8c22\u5171\u75c5\u22652",
       "\u8840\u8102\u5f02\u5e38", "\u9ad8\u8840\u538b", "\u7cd6\u5c3f\u75c5", "\u7761\u7720<6h", "\u7761\u7720\u22658h",
       "\u7761\u7720\u4e0d\u5b89", "\u5b64\u72ec\u611f"]
piv = C.pivot_table(index="exposure", columns="outcome", values="RR").reindex(key)
lo = C.pivot_table(index="exposure", columns="outcome", values="lo").reindex(key)
hi = C.pivot_table(index="exposure", columns="outcome", values="hi").reindex(key)
OUT = ["A \u65b0\u53d1", "B \u73b0\u60a3 2020", "C \u7d2f\u8ba1 ever", "D \u4f20\u7edf ever"]
LEG = ["Incident (2020\u21922021/23)", "Prevalent (2020)", "Cumulative ever (wave 6)", "Conventional ever (arthre)"]
COL = dict(zip(OUT, [RED, GREY, BLUE, PURPLE]))
OFF = dict(zip(OUT, [0.27, 0.09, -0.09, -0.27]))

fig, ax = plt.subplots(figsize=(7.2, 5.4), dpi=300)
y = np.arange(len(key))[::-1]
for o, lb, c in zip(OUT, LEG, [RED, GREY, BLUE, PURPLE]):
    rr = piv[o].values; l = lo[o].values; h = hi[o].values
    ax.errorbar(rr, y + OFF[o], xerr=[rr - l, h - rr], fmt="o", ms=4.0, lw=1.0,
                capsize=0, color=c, label=lb, alpha=0.95)
ax.axvline(1, color=GREY, lw=0.8, ls="--")
ax.set_yticks(y); ax.set_yticklabels(order, fontsize=8.4)
ax.set_xscale("log"); ax.set_xlim(0.68, 1.75)
ax.set_xticks([0.7, 0.85, 1, 1.2, 1.4, 1.6])
ax.set_xticklabels(["0.7", "0.85", "1.0", "1.2", "1.4", "1.6"], fontsize=8.4)
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_xlabel("Risk ratio (95% CI, log scale)", fontsize=9)
ax.legend(fontsize=7.6, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.13),
          handletextpad=0.4, columnspacing=1.1)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
ax.tick_params(axis="y", length=0)
plt.tight_layout()
plt.savefig("out/fig2_construct_en.png", bbox_inches="tight", facecolor="white")
plt.close()

# =========================================================
# Figure 3 — main forest plot (multiple imputation)
# =========================================================
mi = pd.read_csv("out/models_main_MI.csv")
mi["exposure"] = mi.exposure.replace({"\u5b64\u72ec\u611f\uff08\u7ecf\u5e38/\u603b\u662f\uff09": "\u5b64\u72ec\u611f"})
rows = [
 ("Three-domain count \u22651", "\u4e09\u57df\u5171\u75c5 \u22651", True),
 ("Three-domain count \u22652", "\u4e09\u57df\u5171\u75c5 \u22652", True),
 ("1 domain vs none", "\u4e09\u57df\u5171\u75c5 1", True),
 ("2 domains vs none", "\u4e09\u57df\u5171\u75c5 2", True),
 ("3 domains vs none", "\u4e09\u57df\u5171\u75c5 3", True),
 ("Depressive symptoms (CES-D \u226510)", "\u6291\u90c1\u75c7\u72b6\uff08CES-D\u226510\uff09", False),
 ("Metabolic comorbidity 2+", "\u4ee3\u8c22\u5171\u75c5 \u22652 \u9879", False),
 ("Dyslipidaemia", "\u8840\u8102\u5f02\u5e38", False),
 ("Hypertension", "\u9ad8\u8840\u538b", False),
 ("Diabetes", "\u7cd6\u5c3f\u75c5", False),
 ("Loneliness", "\u5b64\u72ec\u611f", False),
 ("Sleep <6 h", "\u7761\u7720 <6h", False),
 ("Sleep \u22658 h", "\u7761\u7720 \u22658h", False),
 ("Restless sleep", "\u7761\u7720\u4e0d\u5b89", False),
]
lab, rr, l, h, comp = [], [], [], [], []
for en, zh, is_comp in rows:
    r = mi[mi.exposure == zh]
    if not len(r): print("MISS", zh); continue
    r = r.iloc[0]
    lab.append(en); rr.append(r.RR); l.append(r.lo); h.append(r.hi); comp.append(is_comp)

fig, ax = plt.subplots(figsize=(6.8, 5.9), dpi=300)
yy = np.arange(len(lab))[::-1]
for i in range(len(lab)):
    ax.plot([l[i], h[i]], [yy[i], yy[i]], lw=1.15, color="#4A5665", solid_capstyle="butt", zorder=2)
    ax.plot(rr[i], yy[i], "o", ms=5.2, color=(RED if comp[i] else INK), zorder=3)
ax.axvline(1, color=GREY, lw=0.8, ls="--")
ax.set_yticks(yy); ax.set_yticklabels(lab, fontsize=8.4)
ax.set_xscale("log"); ax.set_xlim(0.75, 2.35)
ax.set_xticks([0.8, 1, 1.25, 1.5, 2])
ax.set_xticklabels(["0.8", "1.0", "1.25", "1.5", "2.0"], fontsize=8.4)
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_xlabel("Risk ratio (95% CI, log scale)", fontsize=9)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
ax.tick_params(axis="y", length=0)
plt.tight_layout()
plt.savefig("out/fig3_main_en.png", bbox_inches="tight", facecolor="white")
plt.close()

# =========================================================
# Figure 4 — restricted cubic splines
# =========================================================
rcs = json.load(open("out/rcs.json", encoding="utf-8"))
cfg = [("\u6291\u90c1\u75c7\u72b6\u8bc4\u5206", "Depressive symptoms (CES-D score)", "CES-D score"),
       ("\u591c\u95f4\u7761\u7720", "Night-time sleep duration", "Hours"),
       ("\u5348\u7761", "Daytime napping", "Minutes"),
       ("BMI\uff082015\uff09", "Body mass index (2015)", "kg/m\u00b2"),
       ("CRP\uff082015\uff09", "C-reactive protein (2015)", "mg/L"),
       ("\u5c3f\u9178\uff082015\uff09", "Uric acid (2015)", "mg/dL")]
fig, axes = plt.subplots(2, 3, figsize=(10.6, 6.3), dpi=300)
for ax, (k, title, xlab) in zip(axes.ravel(), cfg):
    c = rcs[k]
    x = np.array(c["x"]); r = np.array(c["rr"]); lo_ = np.array(c["lo"]); hi_ = np.array(c["hi"])
    ax.fill_between(x, lo_, hi_, color=RED, alpha=0.13, lw=0)
    ax.plot(x, r, color=RED, lw=1.6)
    ax.axhline(1, color=GREY, lw=0.8, ls="--")
    ax.set_title("%s\nOverall P = %.2f; non-linear P = %.2f" % (title, c["p_overall"], c["p_nonlin"]),
                 fontsize=8.6, color=INK, pad=6)
    ax.set_xlabel(xlab, fontsize=8.2)
    ax.set_ylabel("Risk ratio (95% CI)", fontsize=8.2)
    ax.tick_params(labelsize=7.8)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    if title.startswith("C-reactive"):
        ax.set_xticks([0, 1, 2, 3]); ax.set_xticklabels(["1", "2", "4", "8"])
plt.tight_layout()
plt.savefig("out/fig4_rcs_en.png", bbox_inches="tight", facecolor="white")
plt.close()

# =========================================================
# Figure 5 — lagged biomarkers
# =========================================================
b = pd.read_csv("out/biomarker.csv")
b = b[(b.model == "M3 +2015\u751f\u6d3b\u65b9\u5f0f/\u5065\u5eb7") & (~b.biomarker.str.contains("\u56db\u5206\u4f4d"))]
MAP = {"CRP\uff08\u6bcf\u7ffb\u500d\uff09": "C-reactive protein, per doubling",
       "\u767d\u7ec6\u80de\uff08\u6bcf +1\u00d710\u2079/L\uff09": "White cell count, per 10\u2079/L",
       "HbA1c\uff08\u6bcf +1%\uff09": "HbA1c, per 1%",
       "\u7a7a\u8179\u8840\u7cd6\uff08\u6bcf +10 mg/dL\uff09": "Fasting glucose, per 10 mg/dL",
       "\u5c3f\u9178\uff08\u6bcf +1 mg/dL\uff09": "Uric acid, per mg/dL",
       "TG/HDL \u6bd4\u503c\uff08\u6bcf +1 log\uff09": "Triglyceride/HDL ratio, per log unit",
       "BMI\uff08\u6bcf +5 kg/m\u00b2\uff09": "Body mass index, per 5 kg/m\u00b2"}
b["en"] = b.biomarker.map(MAP)
b = b.dropna(subset=["en"]).set_index("en")
order5 = ["C-reactive protein, per doubling", "White cell count, per 10\u2079/L", "HbA1c, per 1%",
          "Fasting glucose, per 10 mg/dL", "Uric acid, per mg/dL",
          "Triglyceride/HDL ratio, per log unit", "Body mass index, per 5 kg/m\u00b2"]
b = b.reindex(order5)

fig, ax = plt.subplots(figsize=(6.6, 3.5), dpi=300)
yy = np.arange(len(b))[::-1]
for i, (nm, r) in enumerate(b.iterrows()):
    ax.plot([r.lo, r.hi], [yy[i], yy[i]], lw=1.15, color="#4A5665", solid_capstyle="butt", zorder=2)
    ax.plot(r.RR, yy[i], "o", ms=5.0, color=(RED if nm.startswith("Body") else INK), zorder=3)
ax.axvline(1, color=GREY, lw=0.8, ls="--")
ax.set_yticks(yy); ax.set_yticklabels(b.index, fontsize=8.2)
ax.set_xscale("log"); ax.set_xlim(0.62, 1.32)
ax.set_xticks([0.7, 0.85, 1, 1.15, 1.3])
ax.set_xticklabels(["0.7", "0.85", "1.0", "1.15", "1.3"], fontsize=8.2)
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_xlabel("Risk ratio (95% CI, log scale)", fontsize=9)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
ax.tick_params(axis="y", length=0)
plt.tight_layout()
plt.savefig("out/fig5_biomarker_en.png", bbox_inches="tight", facecolor="white")
plt.close()

print("done")
