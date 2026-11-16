# -*- coding: utf-8 -*-
"""Early primary models - SUPERSEDED, retained for provenance.

This version used an offset for follow-up and HC0 variance. The final analysis
uses a free log follow-up term with community-clustered variance (38_cluster.py)
and multiple imputation (40_mi.py). No downstream script reads its output."""
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.environ.get("W6_OUT", os.path.join(ROOT, "out"))
os.makedirs(OUT, exist_ok=True)

import pandas as pd, numpy as np, statsmodels.api as sm, statsmodels.formula.api as smf, warnings
warnings.filterwarnings("ignore")
d = pd.read_csv(os.path.join(OUT, "analytic_raw.csv"), dtype={"ID":str})
d["ziw_year"] = pd.to_numeric(d.ziwtime2.astype(str).str[:4], errors="coerce")
d["in_main"] = ((d.at_risk==1) & (d.ziw_year==2020) & d.age_base.between(45,95)).astype(int)
m = d[d.in_main==1].copy()

m["age"]=m.age_base; m["age2"]=m.age**2/100
m["edu_c"]=m.edu.map({1:"Low",2:"Mid",3:"High"}); m["reg"]=m.region3.astype(str)
m["metq"]=m.met_q.astype("float"); m["adl_c"]=m.adl.clip(0,6); m["srh"]=m.srh5
m["oth"]=m.other_n; m["logfu"]=np.log(m.fu_years.clip(0.25))
m["depress_b"]=m.depress; m["metab_b"]=(m.metab_n>=1).astype(float).where(m.metab_n.notna())
m["psleep_b"]=m.poor_sleep
m["nosoc"]=1-m.socwk
m["cesd5"]=m.cesd/5
m["metab1"]=(m.metab_n==1).astype(float).where(m.metab_n.notna())
m["metab2"]=(m.metab_n>=2).astype(float).where(m.metab_n.notna())
m["sl_short"]=(m.sleep_cat=="<6h").astype(float).where(m.sleep_cat.notna())
m["sl_long"]=(m.sleep_cat=="≥8h").astype(float).where(m.sleep_cat.notna())
m["nap_hi"]=(m.nap_cat==">60").astype(float).where(m.nap_cat.notna())
m["nap_mid"]=(m.nap_cat.isin(["1-30","31-60"])).astype(float).where(m.nap_cat.notna())
for j in [1,2,3]:
    m[f"pms{j}"]=(m.pms_n==j).astype(float).where(m.pms_n.notna())

DEMO=["age","age2","female","C(edu_c)","married","rural","C(reg)"]
LIFE=["smoke","drink","metq","oth","adl_c","srh"]

# (标签, 模型主项, 属于哪个域/需调整的另两域)
SPEC=[
 ("CES-D（每+5分）",["cesd5"],["depress_b","metab_b","psleep_b"]),
 ("抑郁症状 CES-D≥10",["depress_b"],[]),
 ("孤独感（经常/总是）",["lonely"],["metab_b","psleep_b"]),
 ("生活不满意",["dissat"],["metab_b","psleep_b"]),
 ("无每月社交",["nosoc"],["metab_b","psleep_b"]),
 ("代谢共病 1 项 vs 0",["metab1"],["depress_b","psleep_b"]),
 ("代谢共病 ≥2 项 vs 0",["metab2"],["depress_b","psleep_b"]),
 ("高血压",["hibpe"],["depress_b","psleep_b"]),
 ("糖尿病",["diabe"],["depress_b","psleep_b"]),
 ("血脂异常",["dyslipe"],["depress_b","psleep_b"]),
 ("睡眠 <6h vs 6–7.9h",["sl_short"],["depress_b","metab_b","sl_long"]),
 ("睡眠 ≥8h vs 6–7.9h",["sl_long"],["depress_b","metab_b","sl_short"]),
 ("睡眠不安（经常/总是）",["poor_sleep"],["depress_b","metab_b"]),
 ("午睡 1–60min vs 0",["nap_mid"],["depress_b","metab_b","nap_hi"]),
 ("午睡 >60min vs 0",["nap_hi"],["depress_b","metab_b","nap_mid"]),
 ("三域共病 1 vs 0",["pms1"],["pms2","pms3"]),
 ("三域共病 2 vs 0",["pms2"],["pms1","pms3"]),
 ("三域共病 3 vs 0",["pms3"],["pms1","pms2"]),
]
# 抑郁/代谢/睡眠 单域二分类需要同时在模型里：把 depress_b 也放进抑郁模型作为自身主项
def fit(df, terms, extra):
    X = df.dropna(subset=["incident","logfu"]+terms+extra)
    if len(X)==0 or X.incident.sum()==0: return None, len(X), 0
    f = "incident ~ " + " + ".join(terms+DEMO+LIFE+extra)
    r = smf.glm(f, data=X, family=sm.families.Poisson(), offset=X.logfu).fit(cov_type="HC0")
    return r, len(X), int(X.incident.sum())

rows=[]
for label, terms, extra in SPEC:
    extra = [e for e in extra if e not in terms]
    for mod in ["M1 粗模型","M2 +人口学","M3 +生活方式/健康/另两域"]:
        r,n,ev = fit(m, terms, [] if mod!="M3 +生活方式/健康/另两域" else ex)
        if r is None: continue
        # 主项：最后一次出现（互调模型中取第2个）
        t = terms[-1]
        # 对 1 vs 0 的成对模型，主项是第一个
        if label.endswith("1 vs 0"): t = terms[0]
        b,se,p = r.params[t], r.bse[t], r.pvalues[t]
        rows.append(dict(exposure=label, term=t, model=mod, n=n, events=ev,
                         RR=np.exp(b), lo=np.exp(b-1.96*se), hi=np.exp(b+1.96*se), p=p))
res = pd.DataFrame(rows)
res["RR_CI"] = res.apply(lambda r: f"{r.RR:.2f} ({r.lo:.2f}–{r.hi:.2f})", axis=1)
res.to_csv(os.path.join(OUT, "results_incident.csv"), index=False, encoding="utf-8-sig")
print(f"主样本 n={len(m)}, events={int(m.incident.sum())}\n")
pv = res[res.model=="M3 +生活方式/健康/另两域"].copy()
pv["p"]=pv.p.map(lambda x: f"{x:.3f}")
print(pv[["exposure","n","events","RR_CI","p"]].to_string(index=False))
