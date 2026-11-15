# -*- coding: utf-8 -*-
"""分类剂量反应：CES-D / 睡眠时长 / 午睡 / BMI(2015)"""
import pandas as pd, numpy as np, statsmodels.api as sm, statsmodels.formula.api as smf, warnings, json
warnings.filterwarnings("ignore")
d=pd.read_csv("out/analytic.csv",dtype={"ID":str})
d["ziw_year"]=pd.to_numeric(d.ziwtime2.astype(str).str[:4],errors="coerce")
d["iy"]=pd.to_numeric(d.iyear,errors="coerce")
m=d[(d.at_risk==1)&(d.ziw_year==2020)&d.age_base.between(45,95)].copy()
m["age"]=m.age_base; m["age2"]=m.age**2/100
m["edu_c"]=m.edu.map({1:"Low",2:"Mid",3:"High"}); m["reg"]=m.region3.astype(str)
m["metq"]=m.met_q.astype(float); m["adl_c"]=m.adl.clip(0,6); m["srh"]=m.srh5; m["oth"]=m.other_n
m["logfu"]=np.log(m.fu_years.clip(0.25)); m["yr"]=m.iy.astype(int).astype(str)
m["bmi3"]=pd.to_numeric(m.bmi_3,errors="coerce")
m["cesd_c"]=pd.cut(m.cesd,[-.1,4,9,14,19,30],labels=["0–4","5–9","10–14","15–19","≥20"])
m["sleep_c"]=pd.cut(m.sleep_h,[0.9,5.9,6.9,7.9,8.9,24],labels=["<6","6–6.9","7–7.9","8–8.9","≥9"])
m["nap_c"]=pd.cut(m.nap_min,[-.1,0,30,60,90,600],labels=["0","1–30","31–60","61–90",">90"])
m["bmi_c"]=pd.cut(m.bmi3,[10,18.5,24,28,60],labels=["<18.5","18.5–23.9","24–27.9","≥28"])
DEMO=["age","age2","female","C(edu_c)","married","rural","C(reg)"]; LIFE=["smoke","drink","metq","oth","adl_c","srh"]
def run(var, ref, extra=[]):
    f="incident ~ C(%s) + "%var + " + ".join(DEMO+LIFE+extra+["logfu","C(yr)"])
    X=m.dropna(subset=[var,"incident","logfu","yr","communityID"]+[c.replace("C(","").replace(")","") for c in DEMO+LIFE+extra])
    r=smf.glm(f,data=X,family=sm.families.Poisson()).fit(cov_type="cluster",cov_kwds={"groups":X["communityID"]})
    out=[]
    for lev in X[var].cat.categories:
        if lev==ref: 
            out.append(dict(var=var,level=str(lev),RR=1.0,lo=np.nan,hi=np.nan,p=np.nan,n=int((X[var]==lev).sum()),
                            ev=int(X.loc[X[var]==lev,"incident"].sum()))); continue
        t=f"C({var})[T.{lev}]"
        if t not in r.params: continue
        b,se,p=r.params[t],r.bse[t],r.pvalues[t]
        out.append(dict(var=var,level=str(lev),RR=np.exp(b),lo=np.exp(b-1.96*se),hi=np.exp(b+1.96*se),p=p,
                        n=int((X[var]==lev).sum()),ev=int(X.loc[X[var]==lev,"incident"].sum())))
    # p for trend（以分类序号为连续变量）
    X2=X.copy(); X2["_ord"]=X2[var].cat.codes.astype(float)
    r2=smf.glm("incident ~ _ord + "+" + ".join(DEMO+LIFE+extra+["logfu","C(yr)"]),data=X2,family=sm.families.Poisson()).fit(cov_type="cluster",cov_kwds={"groups":X2["communityID"]})
    b,se,p=r2.params["_ord"],r2.bse["_ord"],r2.pvalues["_ord"]
    out.append(dict(var=var,level="P-trend",RR=np.exp(b),lo=np.exp(b-1.96*se),hi=np.exp(b+1.96*se),p=p,n=len(X2),ev=int(X2.incident.sum())))
    return out
rows=[]
for v,ref in [("cesd_c","0–4"),("sleep_c","7–7.9"),("nap_c","0"),("bmi_c","18.5–23.9")]:
    rows+=run(v,ref)
R=pd.DataFrame(rows); R.to_csv("out/dose_response.csv",index=False,encoding="utf-8-sig")
print(R.groupby("var").apply(lambda x: x[["level","n","ev","RR","lo","hi","p"]].to_string(index=False), include_groups=False).str.cat(sep="\n\n"))
