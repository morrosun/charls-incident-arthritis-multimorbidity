# -*- coding: utf-8 -*-
"""Formal incident-versus-prevalent comparison (Table 4 panel B).

A single log-link Poisson model in which at-risk participants contribute an
incident row and all participants contribute a prevalent row, with an
exposure-by-outcome-type interaction. The tabulated quantity is
exp(interaction), i.e. how many times larger the prevalent risk ratio is than
the incident one. Variance is clustered on the participant, because the same
person contributes to both rows.

Output: out/construct_competing.csv
"""
# Portable paths: point these environment variables at your local copies of the
# restricted source data (CHARLS / HRS / ELSA / MHAS). No source data is shipped.
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.environ.get("W6_OUT", os.path.join(ROOT, "out"))

import pandas as pd, numpy as np, statsmodels.api as sm, statsmodels.formula.api as smf, warnings
warnings.filterwarnings("ignore")

d = pd.read_csv(os.path.join(OUT, "analytic.csv"), dtype={"ID": str})
d["ziw_year"] = pd.to_numeric(d.ziwtime2.astype(str).str[:4], errors="coerce")
d["iy"] = pd.to_numeric(d.iyear, errors="coerce")
B = d[d.status.isin(["incident", "atrisk_noevent", "prevalent_baseline"]) &
      (d.ziw_year == 2020) & d.age_base.between(45, 95)].copy()
B["age"] = B.age_base
B["age2"] = B.age ** 2 / 100
B["edu_c"] = B.edu.map({1: "Low", 2: "Mid", 3: "High"})
B["reg"] = B.region3.astype(str)
B["metq"] = B.met_q.astype(float)
B["logfu"] = np.log(B.fu_years.clip(0.25))
B["yr"] = B.iy.astype(int).astype(str)
B["dep10"] = (B.cesd >= 10).astype(float).where(B.cesd.notna())
B["metab2"] = (B.metab_n >= 2).astype(float).where(B.metab_n.notna())
B["metab_b"] = (B.metab_n >= 1).astype(float).where(B.metab_n.notna())
for v in ["hibpe", "diabe", "dyslipe"]:
    B[v] = pd.to_numeric(B[v], errors="coerce")
B["sl_short"] = (B.sleep_cat == "<6h").astype(float).where(B.sleep_cat.notna())
B["sl_long"] = (B.sleep_cat == "≥8h").astype(float).where(B.sleep_cat.notna())
B["inc"] = (B.status == "incident").astype(int)
B["prev"] = (B.status == "prevalent_baseline").astype(int)

PRE = ["age", "age2", "female", "C(edu_c)", "married", "rural", "C(reg)",
       "smoke", "drink", "metq"]
PCOL = [c.replace("C(", "").replace(")", "") for c in PRE]


def compete(term, label):
    cols = list(dict.fromkeys([term, "inc", "prev", "status", "communityID", "ID"] + PCOL))
    X = B.dropna(subset=cols).copy()
    inc_rows = X[X.status.isin(["incident", "atrisk_noevent"])].copy()
    inc_rows["y"] = inc_rows["inc"]
    inc_rows["kind"] = 0.0
    prev_rows = X.copy()
    prev_rows["y"] = prev_rows["prev"]
    prev_rows["kind"] = 1.0
    L = pd.concat([inc_rows, prev_rows], ignore_index=True)
    L["int"] = L[term] * L["kind"]
    formula = "y ~ " + term + " + kind + int + " + " + ".join(PRE)
    r = smf.glm(formula, data=L,
                family=sm.families.Poisson()).fit(cov_type="cluster",
                                                  cov_kwds={"groups": L["ID"]})
    b1, b2 = r.params[term], r.params["int"]
    s1, s2 = r.bse[term], r.bse["int"]
    V = np.asarray(r.cov_params())
    i1 = list(r.params.index).index(term)
    i2 = list(r.params.index).index("int")
    sp = np.sqrt(s1 ** 2 + s2 ** 2 + 2 * V[i1, i2])
    return dict(exposure=label,
                RR_inc=np.exp(b1), lo_inc=np.exp(b1 - 1.96 * s1), hi_inc=np.exp(b1 + 1.96 * s1),
                RR_prev=np.exp(b1 + b2), lo_prev=np.exp(b1 + b2 - 1.96 * sp),
                hi_prev=np.exp(b1 + b2 + 1.96 * sp),
                ratio=np.exp(b2), p_int=r.pvalues["int"])


rows = []
for t, lab in [("dep10", "抑郁症状"), ("metab_b", "代谢共病≥1"), ("metab2", "代谢共病≥2"),
               ("dyslipe", "血脂异常"), ("hibpe", "高血压"), ("diabe", "糖尿病"),
               ("sl_short", "睡眠<6h"), ("sl_long", "睡眠≥8h"), ("poor_sleep", "睡眠不安"),
               ("lonely", "孤独感")]:
    rows.append(compete(t, lab))

R = pd.DataFrame(rows)
R.to_csv(os.path.join(OUT, "construct_competing.csv"), index=False, encoding="utf-8-sig")
pd.set_option("display.width", 200)
print(R.round(3).to_string(index=False))
print("\nExposures with ratio > 1 (stronger when prevalent):", int((R.ratio > 1).sum()), "/", len(R))
print("Interaction P < 0.05:", R.loc[R.p_int < 0.05, "exposure"].tolist() or "none")
