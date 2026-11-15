# -*- coding: utf-8 -*-
"""构念对照：同一暴露、同一协变量，四种结局定义
 A incident(2020->W6 新发) | B prevalent(2020 现患, W6 gate) | C ever(W6 累计) | D arthre(w5 传统)
"""
import pandas as pd, numpy as np, statsmodels.api as sm, statsmodels.formula.api as smf, warnings
warnings.filterwarnings("ignore")
d = pd.read_csv("out/analytic_raw.csv", dtype={"ID":str})
d["ziw_year"] = pd.to_numeric(d.ziwtime2.astype(str).str[:4], errors="coerce")
d["core"] = d.status.isin(["incident","atrisk_noevent","prevalent_baseline"]).astype(int)
d["baseok"] = (d.ziw_year==2020) & d.age_base.between(45,95)
m = d[(d.core==1) & d.baseok].copy()

m["age"]=m.age_base; m["age2"]=m.age**2/100
m["edu_c"]=m.edu.map({1:"Low",2:"Mid",3:"High"}); m["reg"]=m.region3.astype(str)
m["metq"]=m.met_q.astype("float"); m["adl_c"]=m.adl.clip(0,6); m["srh"]=m.srh5; m["oth"]=m.other_n
m["logfu"]=np.log(m.fu_years.clip(0.25))
m["depress_b"]=m.depress; m["metab_b"]=(m.metab_n>=1).astype(float).where(m.metab_n.notna())
m["psleep_b"]=m.poor_sleep; m["nosoc"]=1-m.socwk; m["cesd5"]=m.cesd/5
m["metab2"]=(m.metab_n>=2).astype(float).where(m.metab_n.notna())
m["sl_short"]=(m.sleep_cat=="<6h").astype(float).where(m.sleep_cat.notna())
m["nap_hi"]=(m.nap_cat==">60").astype(float).where(m.nap_cat.notna())
m["sl_long"]=(m.sleep_cat=="≥8h").astype(float).where(m.sleep_cat.notna())
m["arthre5"]=pd.to_numeric(m.arthre_5,errors="coerce").where(lambda x: x.isin([0,1]))

DEMO=["age","age2","female","C(edu_c)","married","rural","C(reg)"]
LIFE=["smoke","drink","metq","oth","adl_c","srh"]
SPEC=[("抑郁症状 CES-D≥10","depress_b",["metab_b","psleep_b"]),
      ("CES-D（每+5分）","cesd5",["metab_b","psleep_b"]),
      ("孤独感","lonely",["metab_b","psleep_b"]),
      ("代谢共病 ≥2 项","metab2",["depress_b","psleep_b"]),
      ("血脂异常","dyslipe",["depress_b","psleep_b"]),
      ("高血压","hibpe",["depress_b","psleep_b"]),
      ("糖尿病","diabe",["depress_b","psleep_b"]),
      ("睡眠 <6h","sl_short",["depress_b","metab_b","sl_long"]),
      ("睡眠 ≥8h","sl_long",["depress_b","metab_b","sl_short"]),
      ("睡眠不安","poor_sleep",["depress_b","metab_b"]),
      ("午睡 >60min","nap_hi",["depress_b","metab_b"])]
OUTCOMES=[("A 新发（2020→2021/23）","incident", m.at_risk==1),
          ("B 现患（2020 基线）","prevalent0", pd.Series(True,index=m.index)),
          ("C 累计 ever（W6）","ever_w6", pd.Series(True,index=m.index)),
          ("D 传统 ever（w5 arthre）","arthre5", m.arthre5.notna())]

rows=[]
for label,term,extra in SPEC:
    extra=[e for e in extra if e!=term]
    for oname,y,mask in OUTCOMES:
        X = m[mask].copy()
        need=[c.replace("C(","").replace(")","") for c in ["logfu"]+[term]+DEMO+LIFE+extra]
        X = X.dropna(subset=[y]+need)
        if len(X)<50 or X[y].sum()<20: continue
        f = f"{y} ~ " + " + ".join([term]+DEMO+LIFE+extra)
        r = smf.glm(f, data=X, family=sm.families.Poisson(), offset=X.logfu).fit(cov_type="HC0")
        b,se = r.params[term], r.bse[term]
        rows.append(dict(exposure=label, outcome=oname, n=len(X), events=int(X[y].sum()),
                         rate=100*X[y].mean(), RR=np.exp(b), lo=np.exp(b-1.96*se), hi=np.exp(b+1.96*se), p=r.pvalues[term]))
res=pd.DataFrame(rows)
res.to_csv("out/results_construct.csv", index=False, encoding="utf-8-sig")
piv=res.pivot_table(index="exposure", columns="outcome", values="RR")
piv=piv[["A 新发（2020→2021/23）","B 现患（2020 基线）","C 累计 ever（W6）","D 传统 ever（w5 arthre）"]]
print("=== RR 对照（同一模型、同一协变量） ===")
print(piv.round(2).to_string())
print("\n=== 事件数/样本 ===")
print(res.pivot_table(index="exposure",columns="outcome",values="events").to_string())
print("\n=== 新发 vs 现患 的 RR 比值 ===")
r2=piv.copy(); r2["B/A"]=(r2["B 现患（2020 基线）"]/r2["A 新发（2020→2021/23）"]).round(2)
r2["D/A"]=(r2["D 传统 ever（w5 arthre）"]/r2["A 新发（2020→2021/23）"]).round(2)
print(r2[["A 新发（2020→2021/23）","B 现患（2020 基线）","D 传统 ever（w5 arthre）","B/A","D/A"]].round(2).to_string())
print("\n=== 一致性：A 显著方向 vs B/D ===")
sig=res.pivot_table(index="exposure",columns="outcome",values="p")
print((sig<0.05).to_string())
