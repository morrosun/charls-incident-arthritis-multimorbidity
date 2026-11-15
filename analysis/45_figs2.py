# -*- coding: utf-8 -*-
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import font_manager
for f_ in ["Microsoft YaHei","SimHei"]:
    try: font_manager.findfont(f_,fallback_to_default=False); plt.rcParams["font.sans-serif"]=[f_]; break
    except Exception: pass
plt.rcParams["axes.unicode_minus"]=False
mi=pd.read_csv("out/models_main_MI.csv")
mi["exposure"]=mi.exposure.replace({"孤独感（经常/总是）":"孤独感"})
lab=["三域共病 ≥1","三域共病 ≥2","三域共病 1","三域共病 2","三域共病 3",
     "抑郁症状（CES-D≥10）","代谢共病 ≥2 项","血脂异常","高血压","糖尿病",
     "孤独感","睡眠 <6h","睡眠 ≥8h","睡眠不安"]
v=mi.set_index("exposure").reindex(lab).reset_index()
fig,ax=plt.subplots(figsize=(6.8,5.8),dpi=300)
yy=np.arange(len(v))[::-1]
ax.errorbar(v.RR,yy,xerr=[v.RR-v.lo,v.hi-v.RR],fmt="none",lw=1.2,capsize=0,color="#4a5665",zorder=2)
for i,r in v.iterrows():
    c="#9b2335" if "三域" in r.exposure else "#33414f"
    ax.plot(r.RR,yy[i],"o",ms=5.2,color=c,zorder=3)
ax.axvline(1,color="#7b8794",lw=.8,ls="--")
ax.set_yticks(yy); ax.set_yticklabels(v.exposure,fontsize=8.6)
ax.set_xscale("log"); ax.set_xlim(0.75,2.3)
ax.set_xticks([0.8,1,1.25,1.5,2]); ax.set_xticklabels(["0.8","1.0","1.25","1.5","2.0"],fontsize=8.6)
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_xlabel("相对危险度 RR（95% CI，对数刻度）",fontsize=9)
for s in ["top","right"]: ax.spines[s].set_visible(False)
ax.tick_params(axis="y",length=0)
plt.tight_layout(); plt.savefig("out/fig2_main.png",bbox_inches="tight"); plt.close()

# 血检森林图
b=pd.read_csv("out/biomarker.csv")
b=b[(b.model=="M3 +2015生活方式/健康")&(~b.biomarker.str.contains("四分位"))]
order=["CRP（每翻倍）","白细胞（每 +1×10⁹/L）","HbA1c（每 +1%）","空腹血糖（每 +10 mg/dL）",
       "尿酸（每 +1 mg/dL）","TG/HDL 比值（每 +1 log）","BMI（每 +5 kg/m²）"]
b=b.set_index("biomarker").reindex(order).reset_index()
fig,ax=plt.subplots(figsize=(6.4,3.4),dpi=300)
yy=np.arange(len(b))[::-1]
ax.errorbar(b.RR,yy,xerr=[b.RR-b.lo,b.hi-b.RR],fmt="none",lw=1.2,capsize=0,color="#4a5665",zorder=2)
for i,r in b.iterrows():
    c="#9b2335" if r.biomarker.startswith("BMI") else "#33414f"
    ax.plot(r.RR,yy[i],"o",ms=5,color=c,zorder=3)
ax.axvline(1,color="#7b8794",lw=.8,ls="--")
ax.set_yticks(yy); ax.set_yticklabels([x.replace("×10⁹/L","×10^9/L") for x in b.biomarker],fontsize=8.4)
ax.set_xscale("log"); ax.set_xlim(0.6,1.35)
ax.set_xticks([0.7,0.85,1,1.15,1.3]); ax.set_xticklabels(["0.7","0.85","1.0","1.15","1.3"],fontsize=8.4)
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_xlabel("RR（95% CI，对数刻度）",fontsize=9)
for s in ["top","right"]: ax.spines[s].set_visible(False)
ax.tick_params(axis="y",length=0)
plt.tight_layout(); plt.savefig("out/fig4_biomarker.png",bbox_inches="tight"); plt.close()
print("ok")
