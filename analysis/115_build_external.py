# -*- coding: utf-8 -*-
"""Build the wave t to t+1 incident transitions in HRS, ELSA and MHAS.

Risk set: free of arthritis at t, with a next interview 1.0-4.5 years later.
Successive available interviews are used, so a skipped wave is allowed; the
interval is carried into the model as an offset.

Output: ext_HRS.csv, ext_ELSA.csv, ext_MHAS.csv, ext_summary.csv"""
# Portable paths: point these environment variables at your local copies of the
# restricted source data (CHARLS / HRS / ELSA / MHAS). No source data is shipped.
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.environ.get("W6_OUT", os.path.join(ROOT, "out"))
CHARLS_RAW = os.environ.get("CHARLS_RAW", os.path.join(ROOT, "data", "raw", "CHARLS"))
CHARLS_WORK = os.environ.get("CHARLS_WORK", os.path.join(ROOT, "data", "working", "charls.dta"))
HRS_FILE = os.environ.get("HRS_FILE", os.path.join(ROOT, "data", "working", "hrs.dta"))
ELSA_FILE = os.environ.get("ELSA_FILE", os.path.join(ROOT, "data", "working", "elsa.dta"))
MHAS_FILE = os.environ.get("MHAS_FILE", os.path.join(ROOT, "data", "working", "mhas.dta"))

import pyreadstat, pandas as pd, numpy as np, os

os.makedirs(OUT, exist_ok=True)

SPEC = {
 "HRS": dict(
   file=os.environ.get("HRS_FILE", os.path.join(ROOT, "data", "working", "hrs.dta")),
   id="hhidpn", cols=["hhidpn","wave","iwendy","arthre","hibpe","diabe","bmi","mbmi","cesd","shlt",
                      "adl6a","raeducl","mstath","smokev","drink","rural","ragender","agey","rabyear"],
   adl="adl6a", year="iwendy"),
 "ELSA": dict(
   file=os.environ.get("ELSA_FILE", os.path.join(ROOT, "data", "working", "elsa.dta")),
   id="idauniqc", cols=["idauniqc","wave","iwindy","arthre","hibpe","diabe","mbmi","cesd","shlt",
                        "adltot6","raeducl","mstath","smokev","drink","ragender","agey","rabyear"],
   adl="adltot6", year="iwindy"),
 "MHAS": dict(
   file=os.environ.get("MHAS_FILE", os.path.join(ROOT, "data", "working", "mhas.dta")),
   id="rahhidnp", cols=["rahhidnp","wave","iwy","arthre","hibpe","diabe","bmi","cesd_m","shlt",
                        "adltot6","raeducl","mstath","smokev","drink","rural","ragender","agey","rabyear"],
   adl="adltot6", year="iwy"),
}

def num(s): return pd.to_numeric(s, errors="coerce")

summary = []
for coh, sp in SPEC.items():
    df, meta = pyreadstat.read_dta(sp["file"], usecols=sp["cols"])
    df = df.rename(columns={sp["id"]: "ID", sp["adl"]: "adl", sp["year"]: "year",
                            "cesd_m": "cesd"})
    for c in ["wave","year","arthre","hibpe","diabe","bmi","mbmi","cesd","shlt","adl","raeducl",
              "mstath","smokev","drink","rural","ragender","agey","rabyear","iwendy"]:
        if c in df: df[c] = num(df[c])
    df = df.dropna(subset=["ID","wave"]).copy()
    df["wave"] = df.wave.astype(int)
    df = df.sort_values(["ID","wave"]).reset_index(drop=True)

    # 下一条观测（同一人）
    g = df.groupby("ID", sort=False)
    for v in ["arthre","year","wave"]:
        df[v + "_next"] = g[v].shift(-1)

    df["interval"] = df["year_next"] - df["year"]
    # 结局与风险集
    df["at_risk"] = ((df.arthre == 0) & df.arthre_next.isin([0, 1]) &
                     df.interval.between(1.0, 4.5)).astype(int)
    df["event"] = ((df.arthre_next == 1) & (df.at_risk == 1)).astype(int)

    # 暴露
    df["hibpe_b"] = df.hibpe.where(df.hibpe.isin([0, 1]))
    df["diabe_b"] = df.diabe.where(df.diabe.isin([0, 1]))
    df["metab"] = ((df.hibpe_b == 1) | (df.diabe_b == 1)).astype(float)
    df.loc[df.hibpe_b.isna() | df.diabe_b.isna(), "metab"] = np.nan
    # 体格测量优先：HRS/ELSA 用实测 mbmi，MHAS 仅有自报 bmi
    if "mbmi" in df.columns:
        df["bmi_raw"] = df["mbmi"]
        df["bmi_meas"] = df["mbmi"].between(10, 60).astype(int)
    elif "bmi" in df.columns:
        df["bmi_raw"] = df["bmi"]
        df["bmi_meas"] = 0
    else:
        df["bmi_raw"] = np.nan
        df["bmi_meas"] = 0
    df.loc[~df.bmi_raw.between(10, 60), "bmi_raw"] = np.nan
    df["bmi5"] = df.bmi_raw / 5.0
    df["obese"] = (df.bmi_raw >= 28).astype(float).where(df.bmi_raw.notna())
    df["cesd"] = df.cesd.where(df.cesd.between(0, 30))
    df["female"] = np.where(df.ragender == 0, 1, np.where(df.ragender == 1, 0, np.nan))
    df["edu"] = df.raeducl.where(df.raeducl.isin([1, 2, 3]))
    df["married"] = np.where(df.mstath.isin([1, 2]), 1, np.where(df.mstath.notna(), 0, np.nan))
    df["age"] = df.agey if "agey" in df else np.nan
    if "rabyear" in df:
        df["age"] = df["age"].fillna(df["year"] - df["rabyear"])
    df["logfu"] = np.log(df.interval.clip(0.5))
    df["shlt"] = df.shlt.where(df.shlt.between(1, 5))
    df["adl"] = df.adl.clip(0, 6)
    df["yr"] = df.year.astype("Int64").astype(str)

    # 年龄限制 45–95，与 CHARLS 主样本一致
    df["age"] = df["age"] if "age" in df else np.nan
    m = df[(df.at_risk == 1) & df.age.between(45, 95)].copy()
    # 每人首次转换（敏感性分析用）
    m = m.sort_values(["ID", "wave"])
    m["first"] = (~m.ID.duplicated()).astype(int)
    m.to_csv(f"{OUT}/ext_{coh}.csv", index=False, encoding="utf-8-sig")
    summary.append(dict(cohort=coh, rows=len(df), at_risk=len(m), persons=int(m.ID.nunique()),
                        events=int(m.event.sum()),
                        rate=100 * m.event.mean() if len(m) else np.nan,
                        med_interval=float(m.interval.median()) if len(m) else np.nan,
                        metab_prev=100 * m.metab.mean() if m.metab.notna().any() else np.nan,
                        bmi_obs=int(m.bmi5.notna().sum()) if "bmi5" in m else 0))
    print(f"{coh}: rows={len(df)} at_risk={len(m)} persons={m.ID.nunique()} events={int(m.event.sum())} "
          f"rate={100*m.event.mean():.2f}% median_interval={m.interval.median():.1f}y "
          f"metab_prev={100*m.metab.mean():.1f}% bmi_n={int(m.bmi5.notna().sum()) if 'bmi5' in m else 0}")

pd.DataFrame(summary).to_csv(f"{OUT}/ext_summary.csv", index=False, encoding="utf-8-sig")
print("\nsaved")
