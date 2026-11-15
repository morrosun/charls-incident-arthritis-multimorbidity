# -*- coding: utf-8 -*-
import pandas as pd, numpy as np, statsmodels.api as sm, statsmodels.formula.api as smf, warnings, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
warnings.filterwarnings("ignore")
for f in ["Microsoft YaHei","SimHei","SimSun"]:
    try:
        font_manager.findfont(f, fallback_to_default=False); plt.rcParams["font.sans-serif"]=[f]; break
    except Exception: pass
plt.rcParams["axes.unicode_minus"]=False

d=pd.read_csv("out/analytic.csv",dtype={"ID":str})
d["ziw_year"]=pd.to_numeric(d.ziwtime2.astype(str).str[:4],errors="coerce")
d["iy"]=pd.to_numeric(d.iyear,errors="coerce")
d["core"]=d.status.isin(["incident","atrisk_noevent","prevalent_baseline"])
def prep(m):
    m=m.copy()
    m["age"]=m.age_base; m["age2"]=m.age**2/100
    m["edu_c"]=m.edu.map({1:"Low",2:"Mid",3:"High"}); m["reg"]=m.region3.astype(str)
    m["metq"]=m.met_q.astype(float); m["adl_c"]=m.adl.clip(0,6); m["srh"]=m.srh5; m["oth"]=m.other_n
    m["logfu"]=np.log(m.fu_years.clip(0.25)); m["yr"]=m.iy.astype(int).astype(str)
    m["depress_b"]=m.depress; m["metab_b"]=(m.metab_n>=1).astype(float).where(m.metab_n.notna())
    m["metab2"]=(m.metab_n>=2).astype(float).where(m.metab_n.notna()); m["psleep_b"]=m.poor_sleep
    m["cesd5"]=m.cesd/5
    m["sl_short"]=(m.sleep_cat=="<6h").astype(float).where(m.sleep_cat.notna())
    m["sl_long"]=(m.sleep_cat=="≥8h").astype(float).where(m.sleep_cat.notna())
    m["arthre5"]=pd.to_numeric(m.arthre_5,errors="coerce").where(lambda x:x.isin([0,1]))
    return m
base=(d.core)&(d.ziw_year==2020)&d.age_base.between(45,95)
A=prep(d[base&(d.at_risk==1)]); B=prep(d[base])
DEMO=["age","age2","female","C(edu_c)","married","rural","C(reg)"]; LIFE=["smoke","drink","metq","oth","adl_c","srh"]
def fit(df,terms,extra):
    f="incident ~ "+" + ".join(terms+DEMO+LIFE+extra+["logfu","C(yr)"])
    cols=[c.replace("C(","").replace(")","") for c in terms+DEMO+LIFE+extra+["logfu","yr","incident","communityID"]]
    X=df.dropna(subset=cols)
    r=smf.glm(f,data=X,family=sm.families.Poisson()).fit(cov_type="cluster",cov_kwds={"groups":X["communityID"]})
    return r,len(X),int(X.incident.sum())
SPEC=[("抑郁症状","depress_b",["metab_b","psleep_b"]),("代谢共病≥2","metab2",["depress_b","psleep_b"]),
      ("血脂异常","dyslipe",["depress_b","psleep_b"]),("高血压","hibpe",["depress_b","psleep_b"]),
      ("糖尿病","diabe",["depress_b","psleep_b"]),("睡眠<6h","sl_short",["depress_b","metab_b","sl_long"]),
      ("睡眠≥8h","sl_long",["depress_b","metab_b","sl_short"]),("睡眠不安","poor_sleep",["depress_b","metab_b"]),
      ("孤独感","lonely",["metab_b","psleep_b"]),("CES-D 每+5","cesd5",["metab_b","psleep_b"])]
OUT=[("A 新发","incident",A),("B 现患 2020","prevalent0",B),("C 累计 ever","ever_w6",B),("D 传统 ever(w5)","arthre5",B)]
rows=[]
for lab,term,extra in SPEC:
    extra=[e for e in extra if e!=term]
    for oname,y,df in OUT:
        if y=="incident":
            r,n,ev=fit(df,[term],extra)
        else:
            # 替换结局变量
            df=df.copy(); df["incident"]=df[y]
            r,n,ev=fit(df,[term],extra)
        b,se,p=r.params[term],r.bse[term],r.pvalues[term]
        rows.append(dict(exposure=lab,outcome=oname,n=n,cases=ev,RR=np.exp(b),lo=np.exp(b-1.96*se),hi=np.exp(b+1.96*se),p=p))
C=pd.DataFrame(rows); C.to_csv("out/construct.csv",index=False,encoding="utf-8-sig")
piv=C.pivot_table(index="exposure",columns="outcome",values="RR")
order=["A 新发","B 现患 2020","C 累计 ever","D 传统 ever(w5)"]
piv=piv[order]
print(piv.round(2).to_string())

# ---- 图1：四构念森林图 ----
lab_order=["抑郁症状","代谢共病≥2","血脂异常","高血压","糖尿病","睡眠<6h","睡眠≥8h","睡眠不安","孤独感","CES-D 每+5"]
fig,ax=plt.subplots(figsize=(7.2,5.4),dpi=300)
colors={"A 新发":"#C0392B","B 现患 2020":"#7F8C8D","C 累计 ever":"#2980B9","D 传统 ever(w5)":"#8E44AD"}
y=np.arange(len(lab_order))[::-1]
off={"A 新发":0.27,"B 现患 2020":0.09,"C 累计 ever":-0.09,"D 传统 ever(w5)":-0.27}
for o in order:
    s=C[C.outcome==o].set_index("exposure").reindex(lab_order)
    ax.errorbar(s.RR,(y+off[o]),xerr=[s.RR-s.lo,s.hi-s.RR],fmt="o",ms=4.2,lw=1.1,capsize=0,
                color=colors[o],label=o,alpha=.95)
ax.axvline(1,color="k",lw=.8,ls="--")
ax.set_yticks(y); ax.set_yticklabels(lab_order,fontsize=8.5)
ax.set_xscale("log"); ax.set_xlim(0.62,2.3)
ax.set_xticks([0.7,0.85,1,1.25,1.5,2]); ax.set_xticklabels(["0.7","0.85","1.0","1.25","1.5","2.0"],fontsize=8.5)
ax.set_xlabel("相对危险度 RR（对数刻度）",fontsize=9)
ax.legend(fontsize=8,frameon=False,ncol=4,loc="upper center",bbox_to_anchor=(0.5,1.12))
for s in ["top","right"]: ax.spines[s].set_visible(False)
ax.tick_params(axis="y",length=0)
plt.tight_layout(); plt.savefig("out/fig1_construct.png",bbox_inches="tight"); plt.close()

# ---- 图2：三域主结果 ----
mm=pd.read_csv("out/models_main.csv")
v=mm[(mm.model=="M3")&(mm.exposure.isin(["抑郁症状（CES-D≥10）","代谢共病 ≥2 项","血脂异常","高血压","糖尿病","睡眠 <6h","睡眠 ≥8h","睡眠不安","孤独感","三域共病 1","三域共病 2","三域共病 3","三域共病 ≥1","三域共病 ≥2"]))]
lab2=["三域共病 ≥1","三域共病 ≥2","三域共病 1","三域共病 2","三域共病 3","抑郁症状（CES-D≥10）","代谢共病 ≥2 项","血脂异常","高血压","糖尿病","孤独感","睡眠 <6h","睡眠 ≥8h","睡眠不安"]
v=v.set_index("exposure").reindex(lab2).reset_index()
fig,ax=plt.subplots(figsize=(6.6,5.6),dpi=300)
yy=np.arange(len(v))[::-1]
cols=["#C0392B" if "三域" in e else "#34495E" for e in v.exposure]
ax.errorbar(v.RR,yy,xerr=[v.RR-v.lo,v.hi-v.RR],fmt="o",ms=4.6,lw=1.2,capsize=0,color="#34495E",zorder=2)
for i,(x,c) in enumerate(zip(v.RR,cols)):
    ax.plot(x,yy[i],"o",ms=5,color=c,zorder=3)
ax.axvline(1,color="k",lw=.8,ls="--")
ax.set_yticks(yy); ax.set_yticklabels(v.exposure,fontsize=8.5)
ax.set_xscale("log"); ax.set_xlim(0.6,2.4)
ax.set_xticks([0.7,0.85,1,1.25,1.5,2]); ax.set_xticklabels(["0.7","0.85","1.0","1.25","1.5","2.0"],fontsize=8.5)
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_xlabel("相对危险度 RR（95% CI，对数刻度）",fontsize=9)
for s in ["top","right"]: ax.spines[s].set_visible(False)
ax.tick_params(axis="y",length=0)
plt.tight_layout(); plt.savefig("out/fig2_main.png",bbox_inches="tight"); plt.close()
print("figs saved")
