import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import font_manager
for f_ in ["Microsoft YaHei","SimHei"]:
    try: font_manager.findfont(f_,fallback_to_default=False); plt.rcParams["font.sans-serif"]=[f_]; break
    except Exception: pass
plt.rcParams["axes.unicode_minus"]=False
C=pd.read_csv("out/construct_precov.csv")
order=["抑郁症状（CES-D≥10）","代谢共病≥1","代谢共病≥2","血脂异常","高血压","糖尿病","睡眠<6h","睡眠≥8h","睡眠不安","孤独感"]
lab=["抑郁症状","代谢共病≥1","代谢共病≥2","血脂异常","高血压","糖尿病","睡眠<6h","睡眠≥8h","睡眠不安","孤独感"]
piv=C.pivot_table(index="exposure",columns="outcome",values="RR").reindex(order)
LO=C.pivot_table(index="exposure",columns="outcome",values="lo").reindex(order)
HI=C.pivot_table(index="exposure",columns="outcome",values="hi").reindex(order)
OUT=["A 新发","B 现患 2020","C 累计 ever","D 传统 ever"]
col={"A 新发":"#9b2335","B 现患 2020":"#7F8C8D","C 累计 ever":"#2980B9","D 传统 ever":"#8E44AD"}
off={"A 新发":0.27,"B 现患 2020":0.09,"C 累计 ever":-0.09,"D 传统 ever":-0.27}
fig,ax=plt.subplots(figsize=(7.2,5.4),dpi=300)
y=np.arange(len(order))[::-1]
for o in OUT:
    rr=piv[o].values; lo=LO[o].values; hi=HI[o].values
    ax.errorbar(rr,y+off[o],xerr=[rr-lo,hi-rr],fmt="o",ms=4.0,lw=1.0,capsize=0,color=col[o],label=o,alpha=.95)
ax.axvline(1,color="#7b8794",lw=.8,ls="--")
ax.set_yticks(y); ax.set_yticklabels(lab,fontsize=8.6)
ax.set_xscale("log"); ax.set_xlim(0.68,1.75)
ax.set_xticks([0.7,0.85,1,1.2,1.4,1.6]); ax.set_xticklabels(["0.7","0.85","1.0","1.2","1.4","1.6"],fontsize=8.6)
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_xlabel("相对危险度 RR（95% CI，对数刻度）",fontsize=9)
ax.legend(fontsize=8,frameon=False,ncol=4,loc="upper center",bbox_to_anchor=(0.5,1.11))
for s in ["top","right"]: ax.spines[s].set_visible(False)
ax.tick_params(axis="y",length=0)
plt.tight_layout(); plt.savefig("out/fig1_construct.png",bbox_inches="tight"); plt.close()
print("ok")
