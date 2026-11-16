# -*- coding: utf-8 -*-
"""Twenty multiple imputations, then Rubin combination of the primary models.

Chained equations with Bayesian ridge regression and posterior sampling
(20 imputations, 15 iterations); community-clustered variance within each
imputation; Rubin rules across imputations, with the fraction of missing
information reported.

Output: models_main_MI.csv, models_joint_MI.csv, trend_MI.json"""
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.environ.get("W6_OUT", os.path.join(ROOT, "out"))
os.makedirs(OUT, exist_ok=True)

import pandas as pd, numpy as np, statsmodels.api as sm, statsmodels.formula.api as smf, warnings, json, time
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
warnings.filterwarnings("ignore")

d=pd.read_csv(os.path.join(OUT, "analytic.csv"),dtype={"ID":str})
d["ziw_year"]=pd.to_numeric(d.ziwtime2.astype(str).str[:4],errors="coerce")
d["iy"]=pd.to_numeric(d.iyear,errors="coerce")
m=d[(d.at_risk==1)&(d.ziw_year==2020)&d.age_base.between(45,95)].copy()
m["age"]=m.age_base; m["age2"]=m.age**2/100
m["edu_c"]=m.edu.map({1:"Low",2:"Mid",3:"High"}); m["reg"]=m.region3.astype(str)
m["logfu"]=np.log(m.fu_years.clip(0.25)); m["yr"]=m.iy.astype(int).astype(str)
m["oth"] = m.other_n

# 需要插补的原始变量（其余不缺失或缺失<0.2%）
IMP=["cesd","lonely_f","satlife5","sleeprl4","srh5","sleep_h","nap_min",
     "smoke","drink","met","adl"]
# The outcome is included in the imputation model, as recommended: omitting it
# biases imputed covariates towards the unconditional mean.
PRED = IMP + ["incident", "age", "female", "edu", "married", "rural",
              "hibpe", "diabe", "dyslipe", "oth", "socwk", "logfu",
              "region3", "metab_n"]
base=m[PRED].copy()
base["region3"]=base["region3"].astype("category").cat.codes.replace(-1,np.nan)
for c in ["female","married","rural"]:
    base[c]=pd.to_numeric(base[c],errors="coerce")
print("缺失率(%):", (base[IMP].isna().mean()*100).round(1).to_dict())

M_IMP=20
imputations=[]
t0=time.time()
for k in range(M_IMP):
    imp=IterativeImputer(estimator=BayesianRidge(), sample_posterior=True,
                         max_iter=15, random_state=20261115+k, skip_complete=False)
    arr=imp.fit_transform(base.values)
    X=pd.DataFrame(arr,columns=base.columns,index=base.index)
    # 回填整数/二分类
    X["lonely_f"]=X.lonely_f.round().clip(1,4)
    X["satlife5"]=X.satlife5.round().clip(1,5)
    X["sleeprl4"]=X.sleeprl4.round().clip(1,4)
    X["srh5"]=X.srh5.round().clip(1,5)
    X["smoke"]=X.smoke.round().clip(0,1); X["drink"]=X.drink.round().clip(0,1)
    X["cesd"]=X.cesd.clip(0,30); X["sleep_h"]=X.sleep_h.clip(1,24); X["nap_min"]=X.nap_min.clip(0,600)
    X["adl"]=X.adl.round().clip(0,6); X["met"]=X.met.clip(0)
    imputations.append(X)
print(f"插补完成 {M_IMP} 次，用时 {time.time()-t0:.1f}s")

def derive(X, src):
    z=src.copy()
    z["cesd"]=X.cesd.values; z["srh"]=X.srh5.values; z["adl_c"]=X.adl.values
    z["smoke"]=X.smoke.values; z["drink"]=X.drink.values
    z["metq"]=pd.qcut(X.met.values,4,labels=[1,2,3,4]).codes+1.0
    z["depress"]=(X.cesd.values>=10).astype(float)
    z["lonely"]=(X.lonely_f.values>=3).astype(float)
    z["dissat"]=(X.satlife5.values<=2).astype(float)
    z["poor_sleep"]=(X.sleeprl4.values>=3).astype(float)
    sh=X.sleep_h.values; nz=X.nap_min.values
    z["sleep_cat"]=np.where(sh<6,"<6h",np.where(sh<8,"6-7.9h","≥8h"))
    z["nap_cat"]=np.where(nz<=0.1,"0",np.where(nz<=30,"1-30",np.where(nz<=60,"31-60",">60")))
    z["metab_b"]=(z.metab_n>=1).astype(float); z["metab2"]=(z.metab_n>=2).astype(float)
    z["metab1"]=(z.metab_n==1).astype(float)
    z["depress_b"]=z.depress; z["psleep_b"]=z.poor_sleep
    z["cesd5"]=z.cesd/5; z["nosoc"]=1-z.socwk
    z["sl_short"]=(z.sleep_cat=="<6h").astype(float); z["sl_long"]=(z.sleep_cat=="≥8h").astype(float)
    z["nap_hi"]=(z.nap_cat==">60").astype(float); z["nap_mid"]=z.nap_cat.isin(["1-30","31-60"]).astype(float)
    for j in [1,2,3]: z[f"pms{j}"]=(z.pms_n==j).astype(float)
    z["pms_any"]=(z.pms_n>=1).astype(float); z["pms_ge2"]=(z.pms_n>=2).astype(float)
    z["grp"]=np.select([(z.depress_b==1)&(z.metab_b==1),(z.depress_b==0)&(z.metab_b==1),
                        (z.depress_b==1)&(z.metab_b==0),(z.depress_b==0)&(z.metab_b==0)],
                       ["both","metab_only","dep_only","neither"],default="missing")
    for g in ["dep_only","metab_only","both"]: z[g]=(z.grp==g).astype(float)
    return z

DEMO=["age","age2","female","C(edu_c)","married","rural","C(reg)"]; LIFE=["smoke","drink","metq","oth","adl_c","srh"]
def fit(z,terms,extra=[]):
    f="incident ~ "+" + ".join(terms+DEMO+LIFE+extra+["logfu","C(yr)"])
    cols=[c.replace("C(","").replace(")","") for c in terms+DEMO+LIFE+extra+["logfu","yr","incident","communityID"]]
    X=z.dropna(subset=cols).copy()
    r=smf.glm(f,data=X,family=sm.families.Poisson()).fit(cov_type="cluster",cov_kwds={"groups":X["communityID"]})
    return r

def rubin(est,se):
    est=np.asarray(est); se=np.asarray(se); M=len(est)
    q=est.mean(); W=np.mean(se**2); B=np.var(est,ddof=1) if M>1 else 0
    T=W+(1+1/M)*B; df=(M-1)*(1+W/((1+1/M)*B))**2 if B>0 else np.inf
    from scipy import stats as st
    t=st.t.ppf(0.975,df) if np.isfinite(df) else 1.96
    return q, np.sqrt(T), q-t*np.sqrt(T), q+t*np.sqrt(T), df

SPEC=[("抑郁症状（CES-D≥10）","depress_b",["metab_b","psleep_b"]),("CES-D（每+5分）","cesd5",["metab_b","psleep_b"]),
 ("孤独感（经常/总是）","lonely",["metab_b","psleep_b"]),("生活不满意","dissat",["metab_b","psleep_b"]),
 ("无每月社交","nosoc",["metab_b","psleep_b"]),
 ("代谢共病 ≥1 项","metab_b",["depress_b","psleep_b"]),
 ("代谢共病 1 项","metab1",["depress_b","psleep_b"]),("代谢共病 ≥2 项","metab2",["depress_b","psleep_b"]),
 ("高血压","hibpe",["depress_b","psleep_b"]),("糖尿病","diabe",["depress_b","psleep_b"]),("血脂异常","dyslipe",["depress_b","psleep_b"]),
 ("睡眠 <6h","sl_short",["depress_b","metab_b","sl_long"]),("睡眠 ≥8h","sl_long",["depress_b","metab_b","sl_short"]),
 ("睡眠不安","poor_sleep",["depress_b","metab_b"]),("午睡 1–60min","nap_mid",["depress_b","metab_b","nap_hi"]),
 ("午睡 >60min","nap_hi",["depress_b","metab_b","nap_mid"]),
 ("三域共病 1","pms1",["pms2","pms3"]),("三域共病 2","pms2",["pms1","pms3"]),("三域共病 3","pms3",["pms1","pms2"]),
 ("三域共病 ≥1","pms_any",[]),("三域共病 ≥2","pms_ge2",[])]
zlist=[derive(X,m) for X in imputations]
rows=[]
for lab,term,extra in SPEC:
    extra=[e for e in extra if e!=term]
    est=[];ses=[];ns=[]
    for z in zlist:
        r=fit(z,[term],extra); est.append(r.params[term]); ses.append(r.bse[term]); ns.append(len(z))
    q,sd,lo,hi,df=rubin(est,ses)
    from scipy import stats as st
    p=2*st.t.sf(abs(q/sd),df) if np.isfinite(df) else 2*st.norm.sf(abs(q/sd))
    rows.append(dict(exposure=lab,term=term,n=int(np.mean(ns)),RR=np.exp(q),lo=np.exp(lo),hi=np.exp(hi),p=p,FMI=float(1-np.mean(np.array(ses)**2)/sd**2)))
R=pd.DataFrame(rows); R.to_csv(os.path.join(OUT, "models_main_MI.csv"),index=False,encoding="utf-8-sig")
print("\n=== MI 合并（M=20）===")
print(R[["exposure","n","RR","lo","hi","p","FMI"]].round(3).to_string(index=False))

# Joint 2x2 decomposition and multiplicative interaction
est=[];ses=[]
for z in zlist:
    z["d_x_m"]=z.depress_b*z.metab_b
    r=fit(z,["dep_only","metab_only","both"],[])
    est.append([r.params[t] for t in ["dep_only","metab_only","both"]]); ses.append([r.bse[t] for t in ["dep_only","metab_only","both"]])
jr=[]
for i,t in enumerate(["dep_only","metab_only","both"]):
    q,sd,lo,hi,df=rubin([e[i] for e in est],[s[i] for s in ses])
    from scipy import stats as st
    jr.append(dict(group=t,RR=np.exp(q),lo=np.exp(lo),hi=np.exp(hi),p=2*st.t.sf(abs(q/sd),df)))
est=[];ses=[]
for z in zlist:
    z["d_x_m"]=z.depress_b*z.metab_b
    r=fit(z,["depress_b","metab_b","d_x_m"],[])
    est.append(r.params["d_x_m"]); ses.append(r.bse["d_x_m"])
q,sd,lo,hi,df=rubin(est,ses)
from scipy import stats as st
jr.append(dict(group="乘法交互",RR=np.exp(q),lo=np.exp(lo),hi=np.exp(hi),p=2*st.t.sf(abs(q/sd),df)))
pd.DataFrame(jr).to_csv(os.path.join(OUT, "models_joint_MI.csv"),index=False,encoding="utf-8-sig")
print("\n=== 联合（MI）==="); print(pd.DataFrame(jr).round(3).to_string(index=False))
est=[];ses=[]
for z in zlist:
    r=fit(z,["pms_n"],[]); est.append(r.params["pms_n"]); ses.append(r.bse["pms_n"])
q,sd,lo,hi,df=rubin(est,ses)
print(f"\n趋势（MI）：RR={np.exp(q):.3f} ({np.exp(lo):.3f}-{np.exp(hi):.3f}) p={2*st.t.sf(abs(q/sd),df):.4f}")
json.dump(dict(rr=float(np.exp(q)),lo=float(np.exp(lo)),hi=float(np.exp(hi)),p=float(2*st.t.sf(abs(q/sd),df)),M=M_IMP),
          open(os.path.join(OUT, "trend_MI.json"),"w",encoding="utf-8"),ensure_ascii=False,indent=1)
