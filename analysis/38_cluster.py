# -*- coding: utf-8 -*-
"""Primary model with community-clustered robust variance.

Adds communityID from the wave-6 Demographics files to analytic.csv (an
idempotent read-modify-write), then fits modified Poisson models with a free
log follow-up term and interview-year fixed effects for every exposure.

Output: analytic.csv, models_main_cluster.csv, models_joint_cluster.csv,
trend.json"""
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

import pyreadstat, pandas as pd, numpy as np, statsmodels.api as sm, statsmodels.formula.api as smf, warnings, json
warnings.filterwarnings("ignore")
fr=[]
for yr in ["2021","2022","2023"]:
    d,_=pyreadstat.read_dta(f"{CHARLS_RAW}/{yr}/Demographics.dta",usecols=["ID","communityID"])
    fr.append(d)
cid=pd.concat(fr).drop_duplicates("ID")
d=pd.read_csv(os.path.join(OUT, "analytic.csv"),dtype={"ID":str})
d=d[[c for c in d.columns if not c.startswith("communityID")]].merge(cid,on="ID",how="left")
d.to_csv(os.path.join(OUT, "analytic.csv"),index=False,encoding="utf-8-sig")
d["ziw_year"]=pd.to_numeric(d.ziwtime2.astype(str).str[:4],errors="coerce")
d["iy"]=pd.to_numeric(d.iyear,errors="coerce")
m=d[(d.at_risk==1)&(d.ziw_year==2020)&d.age_base.between(45,95)].copy()
m["age"]=m.age_base; m["age2"]=m.age**2/100
m["edu_c"]=m.edu.map({1:"Low",2:"Mid",3:"High"}); m["reg"]=m.region3.astype(str)
m["metq"]=m.met_q.astype(float); m["adl_c"]=m.adl.clip(0,6); m["srh"]=m.srh5; m["oth"]=m.other_n
m["logfu"]=np.log(m.fu_years.clip(0.25)); m["yr"]=m.iy.astype(int).astype(str)
m["depress_b"]=m.depress; m["metab_b"]=(m.metab_n>=1).astype(float).where(m.metab_n.notna())
m["metab1"]=(m.metab_n==1).astype(float).where(m.metab_n.notna())
m["metab2"]=(m.metab_n>=2).astype(float).where(m.metab_n.notna()); m["psleep_b"]=m.poor_sleep
m["nosoc"]=1-m.socwk; m["cesd5"]=m.cesd/5
m["sl_short"]=(m.sleep_cat=="<6h").astype(float).where(m.sleep_cat.notna())
m["sl_long"]=(m.sleep_cat=="≥8h").astype(float).where(m.sleep_cat.notna())
m["nap_hi"]=(m.nap_cat==">60").astype(float).where(m.nap_cat.notna())
m["nap_mid"]=(m.nap_cat.isin(["1-30","31-60"])).astype(float).where(m.nap_cat.notna())
for j in [1,2,3]: m[f"pms{j}"]=(m.pms_n==j).astype(float).where(m.pms_n.notna())
m["pms_any"]=(m.pms_n>=1).astype(float).where(m.pms_n.notna())
m["pms_ge2"]=(m.pms_n>=2).astype(float).where(m.pms_n.notna())
m["grp"]=np.select([(m.depress_b==1)&(m.metab_b==1),(m.depress_b==0)&(m.metab_b==1),
                    (m.depress_b==1)&(m.metab_b==0),(m.depress_b==0)&(m.metab_b==0)],
                   ["both","metab_only","dep_only","neither"],default="missing")
for g in ["dep_only","metab_only","both"]: m[g]=(m.grp==g).astype(float)
DEMO=["age","age2","female","C(edu_c)","married","rural","C(reg)"]; LIFE=["smoke","drink","metq","oth","adl_c","srh"]
def fit(df,terms,extra=[]):
    f="incident ~ "+" + ".join(terms+DEMO+LIFE+extra+["logfu","C(yr)"])
    cols=[c.replace("C(","").replace(")","") for c in terms+DEMO+LIFE+extra+["logfu","yr","incident","communityID"]]
    X=df.dropna(subset=cols)
    r=smf.glm(f,data=X,family=sm.families.Poisson()).fit(cov_type="cluster",cov_kwds={"groups":X["communityID"]})
    return r,len(X),int(X.incident.sum()),X.communityID.nunique()
SPEC=[("抑郁症状（CES-D≥10）","depress_b",["metab_b","psleep_b"]),("CES-D（每+5分）","cesd5",["metab_b","psleep_b"]),
 ("孤独感","lonely",["metab_b","psleep_b"]),("生活不满意","dissat",["metab_b","psleep_b"]),("无每月社交","nosoc",["metab_b","psleep_b"]),
 ("代谢共病 ≥1 项","metab_b",["depress_b","psleep_b"]),
 ("代谢共病 1 项","metab1",["depress_b","psleep_b"]),("代谢共病 ≥2 项","metab2",["depress_b","psleep_b"]),
 ("高血压","hibpe",["depress_b","psleep_b"]),("糖尿病","diabe",["depress_b","psleep_b"]),("血脂异常","dyslipe",["depress_b","psleep_b"]),
 ("睡眠 <6h","sl_short",["depress_b","metab_b","sl_long"]),("睡眠 ≥8h","sl_long",["depress_b","metab_b","sl_short"]),
 ("睡眠不安","poor_sleep",["depress_b","metab_b"]),("午睡 1–60min","nap_mid",["depress_b","metab_b","nap_hi"]),
 ("午睡 >60min","nap_hi",["depress_b","metab_b","nap_mid"]),
 ("三域共病 1","pms1",["pms2","pms3"]),("三域共病 2","pms2",["pms1","pms3"]),("三域共病 3","pms3",["pms1","pms2"]),
 ("三域共病 ≥1","pms_any",[]),("三域共病 ≥2","pms_ge2",[]),("三域共病，每 +1 域","pms_n",[])]
rows=[]
for lab,term,extra in SPEC:
    extra=[e for e in extra if e!=term]
    r,n,ev,nc=fit(m,[term],extra)
    b,se,p=r.params[term],r.bse[term],r.pvalues[term]
    rows.append(dict(exposure=lab,term=term,n=n,events=ev,nclust=nc,RR=np.exp(b),lo=np.exp(b-1.96*se),hi=np.exp(b+1.96*se),p=p))
R=pd.DataFrame(rows); R.to_csv(os.path.join(OUT, "models_main_cluster.csv"),index=False,encoding="utf-8-sig")
print("社区数:", R.nclust.max())
print(R[["exposure","n","events","RR","lo","hi","p"]].round(3).to_string(index=False))
# 联合
r,n,ev,nc=fit(m,["dep_only","metab_only","both"],[])
jr=[]
for t,lab in [("dep_only","仅抑郁"),("metab_only","仅代谢共病"),("both","两者并存")]:
    b,se,p=r.params[t],r.bse[t],r.pvalues[t]
    jr.append(dict(group=lab,n=n,events=ev,RR=np.exp(b),lo=np.exp(b-1.96*se),hi=np.exp(b+1.96*se),p=p))
m["d_x_m"]=m.depress_b*m.metab_b
r2,n2,ev2,nc2=fit(m,["depress_b","metab_b","d_x_m"],[])
b,se,p=r2.params["d_x_m"],r2.bse["d_x_m"],r2.pvalues["d_x_m"]
jr.append(dict(group="乘法交互",n=n2,events=ev2,RR=np.exp(b),lo=np.exp(b-1.96*se),hi=np.exp(b+1.96*se),p=p))
pd.DataFrame(jr).to_csv(os.path.join(OUT, "models_joint_cluster.csv"),index=False,encoding="utf-8-sig")
print("\n联合:"); print(pd.DataFrame(jr).round(3).to_string(index=False))
r3,n3,ev3,nc3=fit(m,["pms_n"],[])
b,se,p=r3.params["pms_n"],r3.bse["pms_n"],r3.pvalues["pms_n"]
print(f"\n趋势 每+1: RR={np.exp(b):.3f} ({np.exp(b-1.96*se):.3f}-{np.exp(b+1.96*se):.3f}) p={p:.4f}")
json.dump(dict(n=int(n3),events=int(ev3),nclust=int(nc3),rr=float(np.exp(b)),lo=float(np.exp(b-1.96*se)),hi=float(np.exp(b+1.96*se)),p=float(p)),
          open(os.path.join(OUT, "trend.json"),"w",encoding="utf-8"),ensure_ascii=False,indent=1)
