# -*- coding: utf-8 -*-
"""Restricted cubic splines.

The spline basis and the design matrix are built by hand, so that the grid and
the reference point share the same knots as the observed data; the alternative
(predicting from a patsy formula) misaligns the basis.

Output: rcs.json, fig3_rcs.png"""
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.environ.get("W6_OUT", os.path.join(ROOT, "out"))
os.makedirs(OUT, exist_ok=True)

import pandas as pd, numpy as np, statsmodels.api as sm, warnings, json, patsy
from scipy import stats as st
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import font_manager
warnings.filterwarnings("ignore")
for f_ in ["Microsoft YaHei","SimHei"]:
    try: font_manager.findfont(f_,fallback_to_default=False); plt.rcParams["font.sans-serif"]=[f_]; break
    except Exception: pass
plt.rcParams["axes.unicode_minus"]=False

d=pd.read_csv(os.path.join(OUT, "analytic.csv"),dtype={"ID":str})
d["ziw_year"]=pd.to_numeric(d.ziwtime2.astype(str).str[:4],errors="coerce")
d["iy"]=pd.to_numeric(d.iyear,errors="coerce")
m=d[(d.at_risk==1)&(d.ziw_year==2020)&d.age_base.between(45,95)].copy()
m["age"]=m.age_base; m["age2"]=m.age**2/100
m["metq"]=m.met_q.astype(float); m["adl_c"]=m.adl.clip(0,6); m["srh"]=m.srh5; m["oth"]=m.other_n
m["logfu"]=np.log(m.fu_years.clip(0.25)); m["yr"]=m.iy.astype(int)
m["bmi3"]=pd.to_numeric(m.bmi_3,errors="coerce").where(lambda x:x.between(10,60))
m["crp"]=pd.to_numeric(m.bl_crp_3,errors="coerce").where(lambda x:(x>0)&(x<200)); m["lcrp2"]=np.log2(m.crp)
m["ua"]=pd.to_numeric(m.bl_ua_3,errors="coerce").where(lambda x:x.between(1,20))
m["edu_mid"]=(m.edu==2).astype(float); m["edu_hi"]=(m.edu==3).astype(float)
for r in ["Central","West","Northeast"]:
    m["reg_"+r]=(m.region3==r).astype(float)
m["yr22"]=(m.yr==2022).astype(float); m["yr23"]=(m.yr==2023).astype(float)
COV=["age","age2","female","married","rural","smoke","drink","metq","oth","adl_c","srh","logfu",
     "edu_mid","edu_hi","reg_Central","reg_West","reg_Northeast","yr22","yr23"]
def rcs(xvar, ref, xlab, xlim=None, df_spline=4, npts=200):
    X=m.dropna(subset=[xvar,"incident","communityID"]+COV).copy()
    xv=X[xvar].values.astype(float); y=X.incident.values.astype(float)
    lo,hi=(np.percentile(xv,1),np.percentile(xv,99)) if xlim is None else xlim
    grid=np.linspace(lo,hi,npts); ref=float(np.clip(ref,lo,hi))
    # 样条基（训练+网格+参照点一起定 knot，保证同一组基）
    allx=np.concatenate([xv,grid,[ref]])
    B=np.asarray(patsy.dmatrix("cr(x, df=%d) - 1"%df_spline,{"x":allx}))
    n=len(X); Btr=B[:n]; Bg=B[n:n+npts]; Br=B[-1]
    D=np.column_stack([Btr, X[COV].values.astype(float)])
    fit=sm.GLM(y,D,family=sm.families.Poisson()).fit(cov_type="cluster",cov_kwds={"groups":X["communityID"].values})
    k=B.shape[1]; beta=fit.params[:k]; V=np.asarray(fit.cov_params())[:k,:k]
    Dd=Bg-Br
    lp=Dd@beta; se=np.sqrt(np.einsum("ij,jk,ik->i",Dd,V,Dd))
    # LRT
    def llf(cols):
        D2=np.column_stack(cols+[X[COV].values.astype(float)]) if cols else X[COV].values.astype(float)
        return sm.GLM(y,D2,family=sm.families.Poisson()).fit().llf, D2.shape[1]
    l_full,k_full=llf([Btr]); l_null,k_null=llf([])
    Blin=np.asarray(patsy.dmatrix("bs(x, df=1, degree=1, include_intercept=False)",{"x":allx}))[:n]
    l_lin,k_lin=llf([Blin])
    p_over=st.chi2.sf(2*(l_full-l_null),k_full-k_null)
    p_nl=st.chi2.sf(2*(l_full-l_lin),k_full-k_lin)
    return dict(x=grid,rr=np.exp(lp),lo=np.exp(lp-1.96*se),hi=np.exp(lp+1.96*se),
                p_overall=float(p_over),p_nonlin=float(p_nl),n=len(X),events=int(y.sum()),ref=ref)
CFG=[("cesd",5,"CES-D 评分",(0,25),"抑郁症状评分"),
     ("sleep_h",7,"夜间睡眠时长（h）",(3,10),"夜间睡眠"),
     ("nap_min",0,"午睡时长（min）",(0,120),"午睡"),
     ("bmi3",22,"BMI（kg/m²，2015）",(17,34),"BMI（2015）"),
     ("lcrp2",float(np.log2(1.3)),"CRP（mg/L）",None,"CRP（2015）"),
     ("ua",4.8,"尿酸（mg/dL）",(2.5,8.5),"尿酸（2015）")]
curves={}
for v,ref,xlab,xlim,nm in CFG:
    o=rcs(v,ref,xlab,xlim); curves[nm]=dict(o,xlab=xlab)
    i_ref=int(np.argmin(abs(o["x"]-o["ref"])))
    print(f"{nm:14s} n={o['n']:5d} ev={o['events']:4d} 总体P={o['p_overall']:.3f} 非线性P={o['p_nonlin']:.3f} "
          f"RR范围={o['rr'].min():.2f}-{o['rr'].max():.2f}")
def _j(v):
    return [float(x) for x in v] if isinstance(v,np.ndarray) else v
json.dump({k:{kk:_j(vv) for kk,vv in v.items()} for k,v in curves.items()},
          open(os.path.join(OUT, "rcs.json"),"w",encoding="utf-8"),ensure_ascii=False,indent=1)
fig,axes=plt.subplots(2,3,figsize=(11,6.4),dpi=300)
for ax,(nm,c) in zip(axes.ravel(),curves.items()):
    ax.fill_between(c["x"],c["lo"],c["hi"],color="#9b2335",alpha=.13,lw=0)
    ax.plot(c["x"],c["rr"],color="#9b2335",lw=1.6)
    ax.axhline(1,color="#8a97a6",lw=.8,ls="--")
    ax.set_title(f"{nm}　总体 P={c['p_overall']:.2f}；非线性 P={c['p_nonlin']:.2f}",fontsize=9)
    ax.set_xlabel(c["xlab"],fontsize=8.5); ax.set_ylabel("RR（95% CI）",fontsize=8.5)
    ax.tick_params(labelsize=8)
    for s in ["top","right"]: ax.spines[s].set_visible(False)
    if nm.startswith("CRP"):
        ax.set_xticks([0,1,2,3]); ax.set_xticklabels(["1","2","4","8"])
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig3_rcs.png"),bbox_inches="tight"); plt.close()
print("saved")
