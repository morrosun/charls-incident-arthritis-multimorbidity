# -*- coding: utf-8 -*-
"""Four outcome definitions under one model (Table 4 panel A, Figure 2).

The same pre-outcome covariate set and the same log-link Poisson model are
applied to four outcomes in the same wave-6 population:

  A  incident disease in the instrument-defined at-risk set
  B  prevalent disease at the 2020 interview
  C  cumulative ever-diagnosed disease at wave 6 (A plus B)
  D  the conventional `arthre` variable at the 2020 interview

Only pre-outcome covariates are used, because self-rated health, activity
limitations and comorbidity count may be consequences of prevalent disease.
Variance is clustered on community.

Output: out/construct_precov.csv
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
d["core"] = d.status.isin(["incident", "atrisk_noevent", "prevalent_baseline"])
base = (d.core) & (d.ziw_year == 2020) & d.age_base.between(45, 95)


def prep(m):
    m = m.copy()
    m["age"] = m.age_base
    m["age2"] = m.age ** 2 / 100
    m["edu_c"] = m.edu.map({1: "Low", 2: "Mid", 3: "High"})
    m["reg"] = m.region3.astype(str)
    m["metq"] = m.met_q.astype(float)
    m["logfu"] = np.log(m.fu_years.clip(0.25))
    m["yr"] = m.iy.astype(int).astype(str)
    m["dep10"] = (m.cesd >= 10).astype(float).where(m.cesd.notna())
    m["metab2"] = (m.metab_n >= 2).astype(float).where(m.metab_n.notna())
    m["metab_b"] = (m.metab_n >= 1).astype(float).where(m.metab_n.notna())
    m["hibpe"] = pd.to_numeric(m.hibpe, errors="coerce")
    m["diabe"] = pd.to_numeric(m.diabe, errors="coerce")
    m["dyslipe"] = pd.to_numeric(m.dyslipe, errors="coerce")
    m["sl_short"] = (m.sleep_cat == "<6h").astype(float).where(m.sleep_cat.notna())
    m["sl_long"] = (m.sleep_cat == "≥8h").astype(float).where(m.sleep_cat.notna())
    m["cesd5"] = m.cesd / 5
    m["arthre5"] = pd.to_numeric(m.arthre_5, errors="coerce").where(lambda x: x.isin([0, 1]))
    return m


A = prep(d[base & (d.at_risk == 1)])
B = prep(d[base])

# Pre-outcome covariates, identical to the competing-risks comparison.
PRE = ["age", "age2", "female", "C(edu_c)", "married", "rural", "C(reg)",
       "smoke", "drink", "metq", "logfu"]

SPEC = [("抑郁症状（CES-D≥10）", "dep10"), ("代谢共病≥2", "metab2"), ("代谢共病≥1", "metab_b"),
        ("血脂异常", "dyslipe"), ("高血压", "hibpe"), ("糖尿病", "diabe"),
        ("睡眠<6h", "sl_short"), ("睡眠≥8h", "sl_long"), ("睡眠不安", "poor_sleep"),
        ("孤独感", "lonely")]
OUTCOMES = [("A 新发", "incident", A), ("B 现患 2020", "prevalent0", B),
            ("C 累计 ever", "ever_w6", B), ("D 传统 ever", "arthre5", B)]

rows = []
for lab, t in SPEC:
    for oname, y, df in OUTCOMES:
        df2 = df.copy()
        if y != "incident":
            df2["incident"] = df2[y]
        cols = [c.replace("C(", "").replace(")", "") for c in ["incident", t] + PRE + ["communityID"]]
        X = df2.dropna(subset=cols)
        r = smf.glm("incident ~ " + t + " + " + " + ".join(PRE), data=X,
                    family=sm.families.Poisson()).fit(
            cov_type="cluster", cov_kwds={"groups": X["communityID"]})
        b, se, p = r.params[t], r.bse[t], r.pvalues[t]
        rows.append(dict(exposure=lab, outcome=oname, n=len(X), cases=int(X.incident.sum()),
                         RR=np.exp(b), lo=np.exp(b - 1.96 * se), hi=np.exp(b + 1.96 * se), p=p))

C = pd.DataFrame(rows)
C.to_csv(os.path.join(OUT, "construct_precov.csv"), index=False, encoding="utf-8-sig")

piv = C.pivot_table(index="exposure", columns="outcome", values="RR")[
    ["A 新发", "B 现患 2020", "C 累计 ever", "D 传统 ever"]]
print(piv.round(2).to_string())
ps = C.pivot_table(index="exposure", columns="outcome", values="p")
print("\nSignificant at 0.05: A=%d B=%d C=%d D=%d (of %d exposures)" % tuple(
    [int((ps[o] < 0.05).sum()) for o in ["A 新发", "B 现患 2020", "C 累计 ever", "D 传统 ever"]] + [len(ps)]))
