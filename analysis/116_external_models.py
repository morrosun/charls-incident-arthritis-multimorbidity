# -*- coding: utf-8 -*-
"""External validation in HRS, ELSA, MHAS and CHARLS.

One exposure definition and one covariate set per cohort, fitted separately
and pooled with DerSimonian-Laird random effects. Anthropometric measures are
pooled only across cohorts with measured height and weight; MHAS self-reported
BMI is reported separately.

Output: ext_models.csv, ext_meta.csv, ext_meta.json"""
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.environ.get("W6_OUT", os.path.join(ROOT, "out"))
os.makedirs(OUT, exist_ok=True)

import pandas as pd, numpy as np, math, statsmodels.api as sm, statsmodels.formula.api as smf, warnings, json
warnings.filterwarnings("ignore")

COV = ["age", "age2", "female", "C(edu)", "married", "smokev", "drink", "adl", "C(shlt)", "logfu"]

def fit(df, exposure, cohort):
    d = df.copy()
    d["age2"] = d.age ** 2 / 100
    d["shlt"] = pd.to_numeric(d.shlt, errors="coerce").clip(1, 5)
    d["edu"] = pd.to_numeric(d.edu, errors="coerce").where(lambda x: x.isin([1, 2, 3]))
    d["smokev"] = pd.to_numeric(d.smokev, errors="coerce")
    d["drink"] = pd.to_numeric(d.drink, errors="coerce")
    d["adl"] = pd.to_numeric(d.adl, errors="coerce").clip(0, 6)
    need = [exposure, "event", "age", "age2", "female", "edu", "married", "smokev",
            "drink", "adl", "shlt", "logfu", "yr", "ID"]
    d = d.dropna(subset=[c for c in need if c in d.columns])
    if len(d) < 200 or d.event.sum() < 30:
        return None
    # 随访时长作 offset（各队列内波次间隔固定，放入 RHS 会与年份哑变量共线）
    f = f"event ~ {exposure} + age + age2 + female + C(edu) + married + smokev + drink + adl + C(shlt) + C(yr)"
    r = smf.glm(f, data=d, family=sm.families.Poisson(), offset=np.asarray(d["logfu"])).fit(
        cov_type="cluster", cov_kwds={"groups": d["ID"]})
    b, se = r.params[exposure], r.bse[exposure]
    return dict(cohort=cohort, exposure=exposure, n=len(d), persons=int(d["ID"].nunique()),
                events=int(d.event.sum()),
                RR=float(np.exp(b)), lo=float(np.exp(b - 1.96 * se)),
                hi=float(np.exp(b + 1.96 * se)), p=float(r.pvalues[exposure]),
                prev=float(d[exposure].mean()))

rows = []
for coh in ["HRS", "ELSA", "MHAS"]:
    df = pd.read_csv(fos.path.join(OUT, "ext_{coh}.csv"), dtype={"ID": str})
    df["yr"] = df["yr"].astype(str)
    for exp, lab in [("metab", "Metabolic comorbidity (hypertension or diabetes)"),
                     ("bmi5", "Body mass index, per 5 kg/m2"),
                     ("obese", "Obesity (BMI >=28) vs not")]:
        if exp not in df or df[exp].notna().sum() < 500:
            continue
        res = fit(df, exp, coh)
        if res:
            res["exposure_label"] = lab
            res["sample"] = "all transitions"
            res["anthro"] = ("measured" if (coh != "MHAS") else "self-reported") if exp != "metab" else ""
            rows.append(res)
            if "first" in df.columns:
                res2 = fit(df[df["first"] == 1], exp, coh)
                if res2:
                    res2["exposure_label"] = lab
                    res2["sample"] = "first transition only"
                    res2["anthro"] = res.get("anthro", "")
                    rows.append(res2)
            print(f"{coh:5s} {exp:7s} n={res['n']:6d} ev={res['events']:5d} "
                  f"RR={res['RR']:.2f} ({res['lo']:.2f}-{res['hi']:.2f}) p={res['p']:.3f} prev={res['prev']:.2f}")

# ---------- CHARLS 同定义 ----------
d = pd.read_csv(os.path.join(OUT, "analytic.csv"), dtype={"ID": str})
d["z"] = pd.to_numeric(d.ziwtime2.astype(str).str[:4], errors="coerce")
d["iy"] = pd.to_numeric(d.iyear, errors="coerce")
m = d[(d.at_risk == 1) & (d.z == 2020) & d.age_base.between(45, 95)].copy()
m["age"] = m.age_base
m["female"] = m.female
m["edu"] = m.edu
m["married"] = m.married
m["smokev"] = m.smoke
m["drink"] = m.drink
m["adl"] = m.adl
m["shlt"] = m.srh5
m["logfu"] = np.log(m.fu_years.clip(0.25))
m["yr"] = m.iy.astype(int).astype(str)
m["ID"] = m.ID
m["event"] = m.incident
for v in ["hibpe", "diabe"]:
    m[v] = pd.to_numeric(m[v], errors="coerce").where(lambda x: x.isin([0, 1]))
m["metab"] = ((m.hibpe == 1) | (m.diabe == 1)).astype(float)
m.loc[m.hibpe.isna() | m.diabe.isna(), "metab"] = np.nan
m["bmi5"] = (pd.to_numeric(m.bmi_3, errors="coerce").where(lambda x: x.between(10, 60))) / 5
m["obese"] = (pd.to_numeric(m.bmi_3, errors="coerce") >= 28).astype(float).where(
    pd.to_numeric(m.bmi_3, errors="coerce").between(10, 60))
for exp, lab in [("metab", "Metabolic comorbidity (hypertension or diabetes)"),
                 ("bmi5", "Body mass index, per 5 kg/m2"),
                 ("obese", "Obesity (BMI >=28) vs not")]:
    res = fit(m, exp, "CHARLS")
    if res:
        res["exposure_label"] = lab
        res["sample"] = "all transitions"
        res["anthro"] = "measured" if exp != "metab" else ""
        rows.append(res)
        print(f"{'CHARLS':5s} {exp:7s} n={res['n']:6d} ev={res['events']:5d} "
              f"RR={res['RR']:.2f} ({res['lo']:.2f}-{res['hi']:.2f}) p={res['p']:.3f}")

R = pd.DataFrame(rows)
R.to_csv(os.path.join(OUT, "ext_models.csv"), index=False, encoding="utf-8-sig")

# ---------- 随机效应合并（DerSimonian-Laird）----------
def dl(df):
    y = np.log(df.RR.values); se = (np.log(df.hi.values) - np.log(df.lo.values)) / (2 * 1.96)
    w = 1 / se ** 2
    yf = np.sum(w * y) / np.sum(w)
    Q = np.sum(w * (y - yf) ** 2)
    k = len(y); C = np.sum(w) - np.sum(w ** 2) / np.sum(w)
    tau2 = max(0.0, (Q - (k - 1)) / C) if C > 0 else 0.0
    wr = 1 / (se ** 2 + tau2)
    yr = np.sum(wr * y) / np.sum(wr); ser = np.sqrt(1 / np.sum(wr))
    I2 = max(0.0, (Q - (k - 1)) / Q) * 100 if Q > 0 else 0.0
    return dict(k=k, RR=float(np.exp(yr)), lo=float(np.exp(yr - 1.96 * ser)),
                hi=float(np.exp(yr + 1.96 * ser)), p=float(2 * (1 - 0.5 * (1 + math.erf(abs(yr / ser) / np.sqrt(2))))),
                Q=float(Q), I2=float(I2), tau2=float(tau2))

meta = []
Rall = R[R["sample"] == "all transitions"]
# 体格测量指标只合并实测队列（HRS/ELSA/CHARLS）；MHAS 自报单独报告
Rpool = Rall[(Rall.anthro != "self-reported")]
for lab, g in Rpool.groupby("exposure_label"):
    s = dl(g)
    s["exposure_label"] = lab
    s["cohorts"] = ", ".join(g.cohort.tolist())
    meta.append(s)
    print(f"\n[meta] {lab}\n  k={s['k']} pooled RR={s['RR']:.2f} ({s['lo']:.2f}-{s['hi']:.2f}) "
          f"p={s['p']:.4f} I2={s['I2']:.1f}%")

M = pd.DataFrame(meta)
M.to_csv(os.path.join(OUT, "ext_meta.csv"), index=False, encoding="utf-8-sig")
json.dump(meta, open(os.path.join(OUT, "ext_meta.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("\nsaved")
