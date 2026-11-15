# -*- coding: utf-8 -*-
"""2015(w3) 血检滞后暴露 → 2020-2023 新发关节炎/风湿病"""
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
# 补 2015 协变量
f=os.environ.get("CHARLS_WORK", os.path.join(ROOT, "data", "working", "charls.dta"))
vs=["ID","wave","age","adlab_c","iadl","srh","chronic_num","smokev","drinkl","totmet","raeducl","marry","rural2","bmi"]
ch,_=pyreadstat.read_dta(f,usecols=vs)
w3=ch[ch.wave==3].drop(columns=["wave","bmi"]).rename(columns={c:c+"_w3" for c in vs if c not in ("ID","wave","bmi")})
d=pd.read_csv("out/analytic.csv",dtype={"ID":str}).merge(w3,on="ID",how="left")
d["ziw_year"]=pd.to_numeric(d.ziwtime2.astype(str).str[:4],errors="coerce")
d["iy"]=pd.to_numeric(d.iyear,errors="coerce")
m=d[(d.at_risk==1)&(d.ziw_year==2020)&d.age_base.between(45,95)].copy()
m["age"]=m.age_base; m["edu_c"]=m.edu.map({1:"Low",2:"Mid",3:"High"}); m["reg"]=m.region3.astype(str)
m["logfu"]=np.log(m.fu_years.clip(0.25)); m["yr"]=m.iy.astype(int).astype(str)
m["oth"]=m.other_n
# 2015 协变量
num=lambda s: pd.to_numeric(s,errors="coerce")
m["age3"]=num(m.age_w3); m["age3_2"]=m.age3**2/100
m["srh3"]=num(m.srh_w3).where(lambda x:x.between(1,5))
m["adl3"]=num(m.adlab_c_w3).clip(0,6)
m["oth3"]=num(m.chronic_num_w3).clip(0,14)
m["smoke3"]=num(m.smokev_w3); m["drink3"]=num(m.drinkl_w3)
m["met3"]=num(m.totmet_w3); m["metq3"]=pd.qcut(m.met3,4,labels=[1,2,3,4]).astype(float)
m["bmi3"]=num(m.bmi_3).where(lambda x:x.between(10,60))/5.0  # 每 5 kg/m2
# 血检
m["crp"]=num(m.bl_crp_3).where(lambda x:(x>0)&(x<200)); m["lcrp2"]=np.log2(m.crp)
m["hba1c"]=num(m.bl_hbalc_3).where(lambda x:x.between(3,20))
m["glu"]=num(m.bl_glu_3).where(lambda x:x.between(30,600))/10
m["ua"]=num(m.bl_ua_3).where(lambda x:x.between(1,20))
m["wbc"]=num(m.bl_wbc_3).where(lambda x:x.between(1,30))
m["tghdl"]=np.log((num(m.bl_tg_3)/num(m.bl_hdl_3)).where(lambda x:(x>0)&(x<50)))
m["crp_q"]=pd.qcut(m.crp,4,labels=["Q1","Q2","Q3","Q4"])
m["ua_q"]=pd.qcut(m.ua,4,labels=["Q1","Q2","Q3","Q4"])
have=m.crp.notna()
print(f"有2015血检: n={have.sum()}, 事件={int(m.loc[have,'incident'].sum())} ({100*m.loc[have,'incident'].mean():.2f}%)")
for v in ["crp","hba1c","glu","ua","wbc","tghdl","bmi3"]:
    print(f"  {v:8s} n={m.loc[have,v].notna().sum():5d} 中位={m.loc[have,v].median():8.2f}")

DEMO3=["age3","age3_2","female","C(edu_c)","married","rural","C(reg)"]
LIFE3=["smoke3","drink3","oth3","adl3","srh3"]  # 2015 体力活动缺失 56%，不入模型
BIO=["lcrp2","hba1c","glu","ua","wbc","tghdl","bmi3"]
LBL={"lcrp2":"CRP（每翻倍）","hba1c":"HbA1c（每 +1%）","glu":"空腹血糖（每 +10 mg/dL）",
     "ua":"尿酸（每 +1 mg/dL）","wbc":"白细胞（每 +1×10⁹/L）","tghdl":"TG/HDL 比值（每 +1 log）",
     "bmi3":"BMI（每 +5 kg/m²）"}
rows=[]
for v in BIO:
    for mod,extra in [("M1 粗+年龄性别",[]),("M2 +社会经济",[]),("M3 +2015生活方式/健康",[]),("M4 +2020代谢共病",[])]:
        if mod=="M1 粗+年龄性别":
            trm=[v,"age3","age3_2","female"]
        elif mod=="M2 +社会经济":
            trm=[v,"age3","age3_2","female","C(edu_c)","married","rural","C(reg)"]
        elif mod=="M3 +2015生活方式/健康":
            trm=[v]+DEMO3+LIFE3
        else:
            trm=[v]+DEMO3+LIFE3+["metab_n"]
        f_="incident ~ "+" + ".join(trm)+" + logfu + C(yr)"
        cols=[c.replace("C(","").replace(")","") for c in trm+["logfu","yr","incident","communityID"]]
        X=m.dropna(subset=cols)
        if X.incident.sum()<30: continue
        r=smf.glm(f_,data=X,family=sm.families.Poisson()).fit(cov_type="cluster",cov_kwds={"groups":X["communityID"]})
        b,se,p=r.params[v],r.bse[v],r.pvalues[v]
        rows.append(dict(biomarker=LBL[v],var=v,model=mod,n=len(X),events=int(X.incident.sum()),
                         RR=np.exp(b),lo=np.exp(b-1.96*se),hi=np.exp(b+1.96*se),p=p))
# 四分位
m["wbc_q"]=pd.qcut(m.wbc,4,labels=["Q1","Q2","Q3","Q4"])
for v,lab in [("crp_q","CRP 四分位"),("ua_q","尿酸四分位"),("wbc_q","白细胞四分位")]:
    f_=f"incident ~ C({v}) + "+" + ".join(DEMO3+LIFE3)+" + logfu + C(yr)"
    cols=[v]+DEMO3+LIFE3+["logfu","yr","incident","communityID"]
    X=m.dropna(subset=[c.replace("C(","").replace(")","") for c in cols])
    r=smf.glm(f_,data=X,family=sm.families.Poisson()).fit(cov_type="cluster",cov_kwds={"groups":X["communityID"]})
    for lev in ["Q2","Q3","Q4"]:
        t=f"C({v})[T.{lev}]"
        b,se,p=r.params[t],r.bse[t],r.pvalues[t]
        rows.append(dict(biomarker=f"{lab} {lev} vs Q1",var=v,model="M3 +2015生活方式/健康",n=len(X),
                         events=int(X.incident.sum()),RR=np.exp(b),lo=np.exp(b-1.96*se),hi=np.exp(b+1.96*se),p=p))
R=pd.DataFrame(rows); R.to_csv("out/biomarker.csv",index=False,encoding="utf-8-sig")
print("\n=== 2015 血检 → 新发 ===")
print(R[R.model=="M3 +2015生活方式/健康"][["biomarker","n","events","RR","lo","hi","p"]].round(3).to_string(index=False))
print("\n=== M4（再加 2020 代谢共病）===")
print(R[R.model=="M4 +2020代谢共病"][["biomarker","RR","lo","hi","p"]].round(3).to_string(index=False))
