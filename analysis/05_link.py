# WARNING - superseded exploration script, kept for provenance only.
# Its de-duplication rule (prefer the record that answered `da011`) is NOT the
# rule used in the analysis: the 974 repeated records are re-releases of the same
# interview, so the latest annual package must win. See `10_build.py`. The rule
# used here is retained only to document how the issue was first examined, and
# this script produces no input for any downstream step.
#
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
base = os.environ.get("CHARLS_RAW", os.path.join(ROOT, "data", "raw", "CHARLS"))
frames=[]
for yr in ["2021","2022","2023"]:
    df,_ = pyreadstat.read_dta(f"{base}/{yr}/Health_Status_and_Functioning.dta",
                              usecols=["ID","da011_15_","da012_15_"])
    df["src"]=yr; frames.append(df)
w6 = pd.concat(frames, ignore_index=True)
print("union rows", len(w6), "unique ID", w6.ID.nunique())
ov = w6.groupby("ID").src.apply(lambda s: "+".join(sorted(s)))
print(ov.value_counts().to_string())

ch,_ = pyreadstat.read_dta(os.environ.get("CHARLS_WORK", os.path.join(ROOT, "data", "working", "charls.dta")),
                           usecols=["ID","wave","arthre","age","ragender","radyear"])
w5 = ch[ch.wave==5].copy()
print("\nw5 n=",len(w5), " arthre:", w5.arthre.value_counts(dropna=False).to_dict())

# dedupe: keep record with da011 answered first, else first
w6["answered"]=w6.da011_15_.notna().astype(int)
w6 = w6.sort_values(["ID","answered"], ascending=[True,False]).drop_duplicates("ID", keep="first")
print("\nafter dedup n=", len(w6), " answered:", int(w6.answered.sum()))

m = w6.merge(w5[["ID","arthre","age","ragender","radyear"]], on="ID", how="left")
print("matched to w5:", m.arthre.notna().sum(), "of", len(m))

m["status"]=np.select(
    [m.da011_15_.eq(1), m.da011_15_.eq(2), m.da011_15_.isna() & m.da012_15_.isin([1,2,3])],
    ["incident","atrisk_noevent","prevalent_baseline"], default="other")
print("\nstatus x w5 arthre:")
print(pd.crosstab(m.status, m.arthre.fillna(-1)).to_string())

print("\nage of at-risk:", m.loc[m.status.isin(['incident','atrisk_noevent']),'age'].describe()[['count','mean','std']].to_dict())
print("age of prevalent:", m.loc[m.status=='prevalent_baseline','age'].mean())
print("\nincident by src:")
print(pd.crosstab(m.src, m.status).to_string())
