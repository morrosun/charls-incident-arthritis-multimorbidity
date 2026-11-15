# -*- coding: utf-8 -*-
"""在 analytic_raw.csv 上追加 2018(w4) 暴露块 -> analytic.csv"""
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

import pyreadstat, pandas as pd, numpy as np
f=os.environ.get("CHARLS_WORK", os.path.join(ROOT, "data", "working", "charls.dta"))
vs=["ID","wave","arthre","bmi","cesd10","flonel","satlife","sleep_night","sleep_nap","sleeprl",
    "socwk","social1","social2","social3","social4","social5","social6","social7","social8","social9","social10","social11",
    "freq_social1","freq_social2","freq_social3","freq_social4",
    "totmet","smokev","drinkl","srh","chronic_num","raeducl","marry","rural2","age",
    "hibpe","diabe","dyslipe","hearte","stroke","cancre","lunge","livere","kidneye","digeste","psyche","memrye"]
ch,_=pyreadstat.read_dta(f,usecols=vs)
w4=ch[ch.wave==4].drop(columns="wave").rename(columns={c:c+"_4" for c in ch.columns if c not in ("ID","wave")})

d=pd.read_csv("out/analytic_raw.csv",dtype={"ID":str}).merge(w4,on="ID",how="left")
def num(s): return pd.to_numeric(s,errors="coerce")

# 2018 暴露
d["cesd4"]=num(d.cesd10_4).where(lambda x:x.between(0,30))
d["depress4"]=(d.cesd4>=10).astype(float).where(d.cesd4.notna())
d["lonely4"]=(num(d.flonel_4)>=3).astype(float).where(num(d.flonel_4).between(1,4))
d["dissat4"]=(num(d.satlife_4)<=2).astype(float).where(num(d.satlife_4).between(1,5))
# 社会参与计数（11 项：是否参加）
soc=[f"social{i}_4" for i in range(1,12)]
d["soc_cnt4"]=d[soc].apply(lambda r: np.nansum((pd.to_numeric(r,errors="coerce")==1).astype(int)) if r.notna().any() else np.nan, axis=1)
d["soc_any4"]=(d.soc_cnt4>0).astype(float).where(d.soc_cnt4.notna())
d["sleep_h4"]=num(d.sleep_night_4).where(lambda x:x.between(1,24))
d["sleep_cat4"]=pd.cut(d.sleep_h4,[0.99,5.99,7.99,24],labels=["<6h","6-7.9h","≥8h"])
d["nap4"]=num(d.sleep_nap_4).where(lambda x:x.between(0,600))
d["nap_cat4"]=pd.cut(d.nap4,[-0.1,0.1,30,60,600],labels=["0","1-30","31-60",">60"])
d["poor_sleep4"]=(num(d.sleeprl_4)>=3).astype(float).where(num(d.sleeprl_4).between(1,4))
for v in ["hibpe","diabe","dyslipe"]:
    d[v+"_4b"]=num(d[v+"_4"]).where(lambda x:x.isin([0,1]))
d["metab_n4"]=d[["hibpe_4b","diabe_4b","dyslipe_4b"]].sum(axis=1,min_count=3)
d["bmi4"]=num(d.bmi_4); d["bmi4"]=d.bmi4.where(d.bmi4.between(10,50))
d["smoke4"]=num(d.smokev_4); d["drink4"]=num(d.drinkl_4)
d["met4"]=num(d.totmet_4); d["met_q4"]=pd.qcut(d.met4,4,labels=[1,2,3,4])
d["srh4"]=num(d.srh_4).where(lambda x:x.between(1,5))
d["arthre4"]=num(d.arthre_4).where(lambda x:x.isin([0,1]))
d.to_csv("out/analytic.csv",index=False,encoding="utf-8-sig")
print("saved", d.shape)
m=d[(d.at_risk==1)&(pd.to_numeric(d.ziwtime2.astype(str).str[:4],errors="coerce")==2020)&d.age_base.between(45,95)]
print("主样本", len(m), "事件", int(m.incident.sum()))
for c in ["cesd4","depress4","soc_cnt4","sleep_cat4","poor_sleep4","metab_n4","bmi4","srh4"]:
    print(f"  {c:12s} 缺失 {100*m[c].isna().mean():5.1f}%")
print("soc_cnt4 分布:", m.soc_cnt4.value_counts().sort_index().to_dict())
