# -*- coding: utf-8 -*-
"""Build the instrument-defined incident cohort.

Links the three wave-6 annual packages, resolves the 974 duplicate records to
the latest release of the same interview, applies the survey skip pattern to
separate incident from prevalent disease, and attaches wave-5 (2020) exposures
and wave-3 (2015) anthropometry and blood assays.

Output: analytic_raw.csv, build_log.txt"""
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

import pyreadstat, pandas as pd, numpy as np, os, io, sys

RAW = os.environ.get("CHARLS_RAW", os.path.join(ROOT, "data", "raw", "CHARLS"))
WORK = os.environ.get("CHARLS_WORK", os.path.join(ROOT, "data", "working", "charls.dta"))
OUT = os.environ.get("W6_OUT", os.path.join(ROOT, "out"))
os.makedirs(OUT, exist_ok=True)
log = io.StringIO()
def P(*a):
    s = " ".join(str(x) for x in a); print(s); log.write(s+"\n")

# ---------- 1. W6 三年包 ----------
core = ["ID","da011_15_","da012_15_"] + [f"da011_{i}_" for i in range(1,17)] + [f"da012_{i}_" for i in range(1,17)]
core = list(dict.fromkeys(core))
fr = []
for yr in ["2021","2022","2023"]:
    d,_ = pyreadstat.read_dta(f"{RAW}/{yr}/Health_Status_and_Functioning.dta", usecols=core)
    d["src"] = yr
    # 同一年内若重复（不应发生），先去重
    d = d.drop_duplicates("ID")
    fr.append(d)
w6 = pd.concat(fr, ignore_index=True)
P("W6 三年包: 记录数 %d, 唯一 ID %d" % (len(w6), w6.ID.nunique()))

# 访谈年月
si,_ = pyreadstat.read_dta(f"{RAW}/2023/Sample_Infor.dta")
si = si[["ID","iyear","imonth","died","res_wave"]]
w6 = w6.merge(si, on="ID", how="left")
P("Sample_Infor 覆盖: %d / %d" % (w6.iyear.notna().sum(), len(w6)))

# 上次访谈日期（含 2020 波）
dm = []
for yr in ["2021","2022","2023"]:
    d,_ = pyreadstat.read_dta(f"{RAW}/{yr}/Demographics.dta", usecols=["ID","xrage","ziwtime1","ziwtime2"])
    d = d.drop_duplicates("ID"); dm.append(d)
dm = pd.concat(dm, ignore_index=True).drop_duplicates("ID")
w6 = w6.merge(dm, on="ID", how="left")
P("Demographics 覆盖: %d / %d" % (w6.xrage.notna().sum(), len(w6)))

# ---------- 2. 去重 ----------
# 优先级：da011 已回答（=风险集） > 访谈更晚 > 其他变量更全
w6["answered"] = w6.da011_15_.notna().astype(int)
# 877 个 ID 的重复记录均来自同一访谈（iyear/imonth 完全相同），属同一访谈的多次发布版本；
# 因此以「最新发布包」为准（后续发布为更正版），answered 仅作同包内的次级排序。
w6["_order"] = pd.to_numeric(w6["src"], errors="coerce").fillna(0)
w6 = w6.sort_values(["ID","_order","answered"], ascending=[True,False,False])
n_before = len(w6)
w6 = w6.drop_duplicates("ID", keep="first").drop(columns="_order")
P("去重: %d -> %d（丢弃 %d 条重复记录）" % (n_before, len(w6), n_before-len(w6)))

# ---------- 3. 结局与风险集 ----------
w6["status"] = np.select(
    [w6.da011_15_.eq(1), w6.da011_15_.eq(2),
     w6.da011_15_.isna() & w6.da012_15_.isin([1,2,3])],
    ["incident","atrisk_noevent","prevalent_baseline"], default="unclassified")
P("\n结局分布:")
P(w6.status.value_counts().to_string())

# 随访时长
def ym(s):
    t = s.astype(str).str.extract(r"(\d{4})年(\d{2})月")
    return t[0].astype(float) + (t[1].astype(float)-0.5)/12
w6["t_last"] = ym(w6.ziwtime2)
w6["t_w6"]   = pd.to_numeric(w6.iyear, errors="coerce") + (pd.to_numeric(w6.imonth, errors="coerce")-0.5)/12
w6["fu_years"] = w6.t_w6 - w6.t_last
P("\n随访时长(年): " + w6.fu_years.describe().round(3).to_dict().__str__())
P("ziwtime2 年份分布: " + w6.ziwtime2.astype(str).str[:4].value_counts().to_dict().__str__())

# ---------- 4. 基线暴露 (w5=2020) ----------
vs = ["ID","wave","arthre","age","ragender","raeducl","marry","rural2","region","province",
      "smokev","drinkl","totmet","phy_acta","srh","chronic_num","adlab_c","iadl",
      "cesd10","depresl","flonel","satlife","satlifez","socwk","sleep_night","sleep_nap","sleeprl",
      "hibpe","diabe","dyslipe","hearte","stroke","cancre","lunge","livere","kidneye","digeste","psyche","memrye",
      "bmi","mwaist","iwstat","radyear"]
ch,_ = pyreadstat.read_dta(WORK, usecols=vs)
w5 = ch[ch.wave==5].copy()
P("\nw5 记录数: %d" % len(w5))
w5 = w5.rename(columns={c: c+"_5" for c in w5.columns if c not in ("ID","wave")})
w5 = w5.drop(columns="wave")
d = w6.merge(w5, on="ID", how="left")
P("W6 ∩ w5: %d / %d" % (d.age_5.notna().sum(), len(d)))

# w3 (2015) BMI 与血检作为滞后暴露（可选）
b3,_ = pyreadstat.read_dta(WORK, usecols=["ID","wave","bmi","bl_crp","bl_hbalc","bl_glu","bl_ua","bl_tg","bl_hdl","bl_wbc"])
b3 = b3[b3.wave==3].drop(columns="wave").rename(columns={c: c+"_3" for c in ["bmi","bl_crp","bl_hbalc","bl_glu","bl_ua","bl_tg","bl_hdl","bl_wbc"]})
d = d.merge(b3, on="ID", how="left")
P("W6 ∩ w3血检: %d" % d.bl_crp_3.notna().sum())

# ---------- 5. 派生变量 ----------
def num(s): return pd.to_numeric(s, errors="coerce")

d["female"] = np.where(num(d.ragender_5)==0, 1, np.where(num(d.ragender_5)==1, 0, np.nan))
d["age_base"] = num(d.age_5)
d["age_w6"] = num(d.xrage)
d["edu"] = num(d.raeducl_5)
d["married"] = np.where(d.marry_5.isin([1,2,3]), 1, np.where(d.marry_5.notna(), 0, np.nan))
d["rural"] = num(d.rural2_5)
_reg = d.region_5.astype(str).str.strip()
d["region3"] = _reg.map({"东部":"East","中部":"Central","西部":"West","东北":"Northeast"})
d["smoke"] = num(d.smokev_5); d["drink"] = num(d.drinkl_5)
d["met"] = num(d.totmet_5)
d["met_q"] = pd.qcut(d.met, 4, labels=[1,2,3,4])
d["srh5"] = num(d.srh_5); d["srh5"] = d.srh5.where(d.srh5.between(1,5))
d["adl"] = num(d.adlab_c_5); d["iadl5"] = num(d.iadl_5)

# 心理社会
d["cesd"] = num(d.cesd10_5).where(lambda x: x.between(0,30))
d["depress"] = (d.cesd >= 10).astype("float").where(d.cesd.notna())
d["lonely_f"] = num(d.flonel_5).where(lambda x: x.between(1,4))
d["lonely"] = (d.lonely_f >= 3).astype("float").where(d.lonely_f.notna())
d["satlife5"] = num(d.satlife_5).where(lambda x: x.between(1,5))
d["dissat"] = (d.satlife5 <= 2).astype("float").where(d.satlife5.notna())
d["socwk"] = num(d.socwk_5)
d["soc_part"] = d.socwk

# 睡眠
d["sleep_h"] = num(d.sleep_night_5).where(lambda x: x.between(1,24))
d["sleep_cat"] = pd.cut(d.sleep_h, [0.99,5.99,7.99,24], labels=["<6h","6-7.9h","≥8h"])
d["nap_min"] = num(d.sleep_nap_5).where(lambda x: x.between(0,600))
d["nap_cat"] = pd.cut(d.nap_min, [-0.1,0.1,30,60,600], labels=["0","1-30","31-60",">60"])
d["sleeprl4"] = num(d.sleeprl_5).where(lambda x: x.between(1,4))
d["poor_sleep"] = (d.sleeprl4 >= 3).astype("float").where(d.sleeprl4.notna())

# 代谢共病
for v in ["hibpe","diabe","dyslipe"]:
    d[v] = num(d[v+"_5"]).where(lambda x: x.isin([0,1]))
d["metab_n"] = d[["hibpe","diabe","dyslipe"]].sum(axis=1, min_count=3)
d["metab_cat"] = d.metab_n.clip(upper=2).map({0:"0",1:"1",2:"2+"})

# 其他慢病（非风湿）
oth = ["hearte","stroke","cancre","lunge","livere","kidneye","digeste","psyche","memrye"]
for v in oth:
    d[v] = num(d[v+"_5"]).where(lambda x: x.isin([0,1]))
d["other_n"] = d[oth].sum(axis=1, min_count=len(oth))
d["chronic_n5"] = num(d.chronic_num_5)
d["comorb_nonrheum"] = d["other_n"]

# 三域共病（心理-代谢-睡眠）
d["dom_psych"] = d["depress"]
d["dom_metab"] = (d.metab_n >= 1).astype("float").where(d.metab_n.notna())
d["dom_sleep"] = d["poor_sleep"]
d["pms_n"] = d[["dom_psych","dom_metab","dom_sleep"]].sum(axis=1, min_count=3)

# 结局
d["incident"] = (d.status=="incident").astype("float")
d["prevalent0"] = (d.status=="prevalent_baseline").astype("float")
d["ever_w6"] = ((d.status=="incident") | (d.status=="prevalent_baseline")).astype("float")
d["at_risk"] = d.status.isin(["incident","atrisk_noevent"]).astype("float")

d.to_csv(f"{OUT}/analytic_raw.csv", index=False, encoding="utf-8-sig")
P("\n已保存 out/analytic_raw.csv  %d 行 x %d 列" % d.shape)
P("\n=== 关键交叉核对 ===")
P("status × w5 arthre:")
P(pd.crosstab(d.status, d.arthre_5.fillna(-1)).to_string())
P("\n风险集(incl. W6 gate): %d, 事件 %d, 发生率 %.2f%%" % (d.at_risk.sum(), d.incident.sum(), 100*d.incident.sum()/d.at_risk.sum()))
sub = d[(d.at_risk==1) & (d.fu_years.notna())]
P("有随访时长者: %d, 事件 %d, 中位随访 %.2f 年" % (len(sub), sub.incident.sum(), sub.fu_years.median()))
open(f"{OUT}/build_log.txt","w",encoding="utf-8").write(log.getvalue())
