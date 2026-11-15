# -*- coding: utf-8 -*-
"""合并表 2（主模型 CC vs MI）与表 3（血检/剂量反应）"""
import pandas as pd, numpy as np
cc=pd.read_csv("out/models_main_cluster.csv")
mi=pd.read_csv("out/models_main_MI.csv")
key=["抑郁症状（CES-D≥10）","CES-D（每+5分）","孤独感","生活不满意","无每月社交",
     "代谢共病 1 项","代谢共病 ≥2 项","高血压","糖尿病","血脂异常",
     "睡眠 <6h","睡眠 ≥8h","睡眠不安","午睡 1–60min","午睡 >60min",
     "三域共病 1","三域共病 2","三域共病 3","三域共病 ≥1","三域共病 ≥2"]
cc["exposure"]=cc.exposure.replace({"孤独感（经常/总是）":"孤独感","代谢共病 1 项":"代谢共病 1 项"})
t=cc.set_index("exposure").reindex(key)[["n","events","RR","lo","hi","p"]]
t.columns=["n_cc","ev_cc","RR_cc","lo_cc","hi_cc","p_cc"]
mi["exposure"]=mi.exposure.replace({"孤独感（经常/总是）":"孤独感"})
mi2=mi.set_index("exposure").reindex(key)[["RR","lo","hi","p","FMI"]]
mi2.columns=["RR_mi","lo_mi","hi_mi","p_mi","FMI"]
T=t.join(mi2)
def f(r,a,b,c):
    return f"{r[a]:.2f} ({r[b]:.2f}–{r[c]:.2f})"
T["CC (95% CI)"]=T.apply(lambda r:f(r,"RR_cc","lo_cc","hi_cc"),axis=1)
T["MI (95% CI)"]=T.apply(lambda r:f(r,"RR_mi","lo_mi","hi_mi"),axis=1)
T["P_cc"]=T.p_cc.map(lambda x:"<0.001" if x<0.001 else f"{x:.3f}")
T["P_mi"]=T.p_mi.map(lambda x:"<0.001" if x<0.001 else f"{x:.3f}")
T["FMI"]=T.FMI.round(3)
T[["n_cc","ev_cc","CC (95% CI)","P_cc","MI (95% CI)","P_mi","FMI"]].to_csv("out/table2_main.csv",encoding="utf-8-sig")
print("== 表2 ==")
print(T[["n_cc","ev_cc","CC (95% CI)","P_cc","MI (95% CI)","P_mi","FMI"]].to_string())

bio=pd.read_csv("out/biomarker.csv")
b3=bio[bio.model.isin(["M3 +2015生活方式/健康","M4 +2020代谢共病"])].copy()
piv=b3.pivot_table(index="biomarker",columns="model",values=["RR","lo","hi","p"])
order=["CRP（每翻倍）","HbA1c（每 +1%）","空腹血糖（每 +10 mg/dL）","尿酸（每 +1 mg/dL）",
       "白细胞（每 +1×10⁹/L）","TG/HDL 比值（每 +1 log）","BMI（每 +5 kg/m²）"]
out=[]
for b in order:
    r=dict(biomarker=b)
    for m,tag in [("M3 +2015生活方式/健康","M3"),("M4 +2020代谢共病","M4")]:
        if (b,m) in piv["RR"].index if False else b in piv["RR"].index:
            try:
                r[f"{tag}_RR"]=piv.loc[b,("RR",m)]; r[f"{tag}_lo"]=piv.loc[b,("lo",m)]; r[f"{tag}_hi"]=piv.loc[b,("hi",m)]; r[f"{tag}_p"]=piv.loc[b,("p",m)]
            except Exception: pass
    out.append(r)
B=pd.DataFrame(out)
B["M3 (95% CI)"]=B.apply(lambda r:f"{r.M3_RR:.2f} ({r.M3_lo:.2f}–{r.M3_hi:.2f})" if pd.notna(r.get("M3_RR")) else "",axis=1)
B["M4 (95% CI)"]=B.apply(lambda r:f"{r.M4_RR:.2f} ({r.M4_lo:.2f}–{r.M4_hi:.2f})" if pd.notna(r.get("M4_RR")) else "",axis=1)
B.to_csv("out/table3_biomarker.csv",index=False,encoding="utf-8-sig")
print("\n== 表3 血检 ==")
print(B[["biomarker","M3 (95% CI)","M3_p","M4 (95% CI)","M4_p"]].round(3).to_string(index=False))
