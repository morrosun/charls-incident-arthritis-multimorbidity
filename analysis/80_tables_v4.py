# -*- coding: utf-8 -*-
"""按修正后的数据重新生成英文表 1–4（markdown）"""
import pandas as pd, numpy as np
from scipy import stats
d=pd.read_csv("out/analytic.csv",dtype={"ID":str})
d["z"]=pd.to_numeric(d.ziwtime2.astype(str).str[:4],errors="coerce")
m=d[(d.at_risk==1)&(d.z==2020)&d.age_base.between(45,95)].copy()
m["age"]=m.age_base
ev=m.incident==1; nv=m.incident==0
L=[]
def f(x,n=1):
    return "—" if pd.isna(x) else f"{x:.{n}f}"
def ttest(a,b):
    a=pd.to_numeric(a,errors="coerce").dropna(); b=pd.to_numeric(b,errors="coerce").dropna()
    return stats.ttest_ind(a,b,equal_var=False).pvalue if len(a)>2 and len(b)>2 else np.nan
def chi(a,b):
    a=a.fillna(0); b=b.fillna(0)
    t=np.array([[a.sum(),len(a)-a.sum()],[b.sum(),len(b)-b.sum()]])
    return stats.chi2_contingency(t)[1]
def pfmt(p): return "<0.001" if p<0.001 else f"{p:.3f}"

L.append("### Table 1. Baseline characteristics by incident status (n = 11,210)\n")
L.append("| Characteristic | Incident (n = 543) | No event (n = 10,667) | P |")
L.append("|---|---:|---:|---:|")
for v,lab,kind in [("age_base","Age, years","m"),("female","Women, %","p"),("edu","Education below lower secondary, %","e1"),
                   ("married","Married, %","p"),("rural","Rural residence, %","p"),("smoke","Current smoker, %","p"),
                   ("drink","Current drinker, %","p"),("cesd","CES-D-10 score","m"),("depress","Depressive symptoms (CES-D ≥10), %","p"),
                   ("lonely","Loneliness, %","p"),("dissat","Life dissatisfaction, %","p"),("socwk","Monthly social participation, %","p"),
                   ("hibpe","Hypertension, %","p"),("diabe","Diabetes, %","p"),("dyslipe","Dyslipidaemia, %","p"),
                   ("metab_n","Metabolic comorbidities 2+, %","m2"),("sleep_h","Night-time sleep, h","m"),("nap_min","Daytime nap, min","m"),
                   ("poor_sleep","Restless sleep, %","p"),("other_n","Non-rheumatic chronic conditions","m"),
                   ("adl","ADL limitations","m"),("srh5","Self-rated health (1–5, higher better)","m"),
                   ("pms_n","Three-domain count 2, %","pm2")]:
    if kind=="m":
        L.append(f"| {lab} | {f(m.loc[ev,v].mean())} ± {f(m.loc[ev,v].std())} | {f(m.loc[nv,v].mean())} ± {f(m.loc[nv,v].std())} | {pfmt(ttest(m.loc[ev,v],m.loc[nv,v]))} |")
    elif kind=="p":
        L.append(f"| {lab} | {f(100*m.loc[ev,v].mean())} | {f(100*m.loc[nv,v].mean())} | {pfmt(chi(m.loc[ev,v],m.loc[nv,v]))} |")
    elif kind=="e1":
        a=(m.loc[ev,"edu"]==1); b=(m.loc[nv,"edu"]==1)
        L.append(f"| {lab} | {f(100*a.mean())} | {f(100*b.mean())} | {pfmt(chi(a,b))} |")
    elif kind=="m2":
        a=(m.loc[ev,"metab_n"]>=2); b=(m.loc[nv,"metab_n"]>=2)
        L.append(f"| {lab} | {f(100*a.mean())} | {f(100*b.mean())} | {pfmt(chi(a,b))} |")
    elif kind=="pm2":
        a=(m.loc[ev,"pms_n"]==2); b=(m.loc[nv,"pms_n"]==2)
        L.append(f"| {lab} | {f(100*a.mean())} | {f(100*b.mean())} | {pfmt(chi(a,b))} |")
L.append("\nValues are mean ± SD or percentage. Continuous variables compared with Welch's t test, categorical with χ². Denominators vary slightly because of missing values.\n")

# Table 2
cc=pd.read_csv("out/models_main_cluster.csv"); mi=pd.read_csv("out/models_main_MI.csv")
cc["exposure"]=cc.exposure.replace({"孤独感（经常/总是）":"孤独感"}); mi["exposure"]=mi.exposure.replace({"孤独感（经常/总是）":"孤独感"})
key=["三域共病，每 +1 域","三域共病 ≥1","三域共病 ≥2","三域共病 1","三域共病 2","三域共病 3",
     "抑郁症状（CES-D≥10）","CES-D（每+5分）","孤独感","生活不满意","无每月社交",
     "代谢共病 ≥1 项","代谢共病 1 项","代谢共病 ≥2 项","高血压","糖尿病","血脂异常",
     "睡眠 <6h","睡眠 ≥8h","睡眠不安","午睡 1–60min","午睡 >60min"]
EN={"三域共病，每 +1 域":"Three-domain count, per domain","三域共病 ≥1":"Any domain vs none","三域共病 ≥2":"Two or more vs fewer",
    "三域共病 1":"One domain vs none","三域共病 2":"Two domains vs none","三域共病 3":"Three domains vs none",
    "抑郁症状（CES-D≥10）":"Depressive symptoms (CES-D ≥10)","CES-D（每+5分）":"CES-D, per 5 points","孤独感":"Loneliness",
    "生活不满意":"Life dissatisfaction","无每月社交":"No monthly social participation","代谢共病 ≥1 项":"Metabolic comorbidity, any",
    "代谢共病 1 项":"Metabolic comorbidity, 1","代谢共病 ≥2 项":"Metabolic comorbidity, 2+","高血压":"Hypertension",
    "糖尿病":"Diabetes","血脂异常":"Dyslipidaemia","睡眠 <6h":"Sleep <6 h vs 6–7.9 h","睡眠 ≥8h":"Sleep ≥8 h vs 6–7.9 h",
    "睡眠不安":"Restless sleep","午睡 1–60min":"Nap 1–60 min vs 0","午睡 >60min":"Nap >60 min vs 0"}
L.append("### Table 2. Multimorbidity domains and incident arthritis or rheumatism\n")
L.append("| Exposure | Complete case, RR (95% CI) | P | Multiple imputation, RR (95% CI) | P |")
L.append("|---|---:|---:|---:|---:|")
import json as _json
_t=pd.read_json("out/trend_MI.json",typ="series")
for k in key:
    if k=="三域共病，每 +1 域":
        r=cc[cc.exposure==k].iloc[0]
        L.append(f"| {EN[k]} | {r.RR:.2f} ({r.lo:.2f}–{r.hi:.2f}) | {pfmt(r.p)} | {_t['rr']:.2f} ({_t['lo']:.2f}–{_t['hi']:.2f}) | {pfmt(_t['p'])} |")
        continue
    r=cc[cc.exposure==k]; s=mi[mi.exposure==k]
    if len(r)==0 or len(s)==0: continue
    r=r.iloc[0]; s=s.iloc[0]
    L.append(f"| {EN[k]} | {r.RR:.2f} ({r.lo:.2f}–{r.hi:.2f}) | {pfmt(r.p)} | {s.RR:.2f} ({s.lo:.2f}–{s.hi:.2f}) | {pfmt(s.p)} |")
L.append(f"\nComplete-case models: n = {int(cc.n.max()):,}, events = {int(cc.events.max())}. Imputed models: n = {int(mi.n.max()):,}, events = 543. All models adjusted for age, sex, education, marital status, rural residence, region, smoking, drinking, physical activity, non-rheumatic comorbidity count, ADL limitations, self-rated health, log follow-up and interview batch, with community-clustered robust variance. Domains are mutually adjusted within the single-domain models. The trend model is fitted with the count as a continuous term.\n")

# Table 3
b=pd.read_csv("out/biomarker.csv")
b3=b[b.model=="M3 +2015生活方式/健康"].set_index("biomarker")
b4=b[b.model=="M4 +2020代谢共病"].set_index("biomarker")
order=["CRP（每翻倍）","白细胞（每 +1×10⁹/L）","HbA1c（每 +1%）","空腹血糖（每 +10 mg/dL）","尿酸（每 +1 mg/dL）","TG/HDL 比值（每 +1 log）","BMI（每 +5 kg/m²）"]
EN3={"CRP（每翻倍）":"C-reactive protein, per doubling","白细胞（每 +1×10⁹/L）":"White cell count, per 10⁹/L",
     "HbA1c（每 +1%）":"HbA1c, per 1%","空腹血糖（每 +10 mg/dL）":"Fasting glucose, per 10 mg/dL",
     "尿酸（每 +1 mg/dL）":"Uric acid, per mg/dL","TG/HDL 比值（每 +1 log）":"Triglyceride/HDL ratio, per log unit",
     "BMI（每 +5 kg/m²）":"Body mass index, per 5 kg/m²"}
L.append("### Table 3. Lagged (2015) biomarkers and incident arthritis or rheumatism\n")
L.append("| Exposure | Model 1, RR (95% CI) | P | Model 2, RR (95% CI) | P |")
L.append("|---|---:|---:|---:|---:|")
for k in order:
    a=b3.loc[k]; c=b4.loc[k]
    L.append(f"| {EN3[k]} | {a.RR:.2f} ({a.lo:.2f}–{a.hi:.2f}) | {pfmt(a.p)} | {c.RR:.2f} ({c.lo:.2f}–{c.hi:.2f}) | {pfmt(c.p)} |")
L.append("\nModel 1 adjusted for 2015 age, sex, education, marital status, rural residence, region, smoking, drinking, chronic-condition count, ADL and self-rated health. Model 2 additionally adjusted for 2020 metabolic comorbidity. Blood assays: n = 5,491, 251 events; BMI: n = 6,513, 315 events.\n")

# Table 4
c=pd.read_csv("out/construct_precov.csv"); k2=pd.read_csv("out/construct_competing.csv")
piv=c.pivot_table(index="exposure",columns="outcome",values="RR")
ord4=["抑郁症状（CES-D≥10）","代谢共病≥1","代谢共病≥2","血脂异常","高血压","糖尿病","睡眠<6h","睡眠≥8h","睡眠不安","孤独感"]
EN4=dict(zip(ord4,["Depressive symptoms (CES-D ≥10)","Metabolic comorbidity, any","Metabolic comorbidity, 2+","Dyslipidaemia",
                   "Hypertension","Diabetes","Sleep <6 h","Sleep ≥8 h","Restless sleep","Loneliness"]))
L.append("### Table 4. The same associations under four outcome definitions, and a formal comparison of the incident and prevalent constructs\n")
L.append("*Panel A. Risk ratios under four outcome definitions (pre-outcome covariates only)*\n")
L.append("| Exposure | A. Incident | B. Prevalent 2020 | C. Cumulative ever | D. Conventional `arthre` |")
L.append("|---|---:|---:|---:|---:|")
for k in ord4:
    L.append(f"| {EN4[k]} | {piv.loc[k,'A 新发']:.2f} | {piv.loc[k,'B 现患 2020']:.2f} | {piv.loc[k,'C 累计 ever']:.2f} | {piv.loc[k,'D 传统 ever']:.2f} |")
cs=c.pivot_table(index="exposure",columns="outcome",values="cases")
L.append(f"| **Cases available** | **{int(cs['A 新发'].min())}–{int(cs['A 新发'].max())}** | **{int(cs['B 现患 2020'].min()):,}–{int(cs['B 现患 2020'].max()):,}** | **{int(cs['C 累计 ever'].min()):,}–{int(cs['C 累计 ever'].max()):,}** | **{int(cs['D 传统 ever'].min()):,}–{int(cs['D 传统 ever'].max()):,}** |")
L.append("\n*Panel B. Ratio of the prevalent to the incident risk ratio (competing-risk model, collapsible scale)*\n")
L.append("| Exposure | Ratio (prevalent / incident) | P for ratio = 1 |")
L.append("|---|---:|---:|")
EN5={"抑郁症状":"Depressive symptoms","代谢共病≥1":"Metabolic comorbidity, any","代谢共病≥2":"Metabolic comorbidity, 2+",
     "血脂异常":"Dyslipidaemia","高血压":"Hypertension","糖尿病":"Diabetes","睡眠<6h":"Sleep <6 h","睡眠≥8h":"Sleep ≥8 h",
     "睡眠不安":"Restless sleep","孤独感":"Loneliness"}
for _,r in k2.iterrows():
    lab=EN5.get(r.exposure,r.exposure)
    if r.p_int<0.05: lab=f"**{lab}**"
    L.append(f"| {lab} | {r.ratio:.2f} | {pfmt(r.p_int)} |")
L.append("\nPanel A fits the identical model to four outcome definitions; the four columns use different denominators, so the estimates are descriptive rather than formally comparable. Panel B provides the formal comparison: a single log-link model in which at-risk participants contribute an incident row and all participants contribute a prevalent row, with an exposure-by-outcome-type interaction; the tabulated quantity is exp(interaction), i.e. how many times larger the prevalent risk ratio is than the incident one. Only short sleep differs.\n")
open("out/tables_v4.md","w",encoding="utf-8").write("\n".join(L))
print("\n".join(L)[:3000])
