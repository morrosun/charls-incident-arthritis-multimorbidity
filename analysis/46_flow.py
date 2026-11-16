"""Participant flow figure (v2)"""
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.environ.get("W6_OUT", os.path.join(ROOT, "out"))
os.makedirs(OUT, exist_ok=True)

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib import font_manager
for f_ in ["Microsoft YaHei","SimHei"]:
    try: font_manager.findfont(f_,fallback_to_default=False); plt.rcParams["font.sans-serif"]=[f_]; break
    except Exception: pass
plt.rcParams["axes.unicode_minus"]=False

fig,ax=plt.subplots(figsize=(7.6,7.0),dpi=300)
ax.set_xlim(0,100); ax.set_ylim(9,99); ax.axis("off")

def box(x,y,w,h,title,sub=None,fc="#ffffff",ec="#c9d3de",lw=0.9,ts=10.0,ss=11.6,sc="#9b2335"):
    ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle="round,pad=0.35,rounding_size=1.1",
                                fc=fc,ec=ec,lw=lw,zorder=2))
    if sub:
        ax.text(x,y+h*0.16,title,ha="center",va="center",fontsize=ts,color="#1c2430",zorder=3)
        ax.text(x,y-h*0.18,sub,ha="center",va="center",fontsize=ss,color=sc,zorder=3,fontweight="bold")
    else:
        ax.text(x,y,title,ha="center",va="center",fontsize=ts,color="#1c2430",zorder=3)

def arrow(y1,y2,x=34):
    ax.add_patch(FancyArrowPatch((x,y1),(x,y2),arrowstyle="-|>",mutation_scale=9,
                                 lw=0.9,color="#8a97a6",shrinkA=0,shrinkB=0,zorder=1))

def side(y,text,xt=70.5,excl=True):
    ax.add_patch(FancyArrowPatch((51.5,y),(xt-1.6,y),arrowstyle="-",lw=0.7,
                                 color="#b8c3cf",linestyle=(0,(3,2)),zorder=1))
    ax.text(xt,y,text,ha="left",va="center",fontsize=8.3,color="#5d6b7a",zorder=3,
            bbox=dict(boxstyle="round,pad=0.36",fc="#fafbfc",ec="#e3e8ee",lw=0.7))

MAXW, MH = 35, 8.4
# 所有计数来自 flow_counts.json（由 10_build.py 写出），不手写数字
import json
F = json.load(open(os.path.join(OUT, "flow_counts.json"), encoding="utf-8"))
BATCH = "；".join("%d 年 %s 人 / %s 例" % (b["year"], format(b["n"], ","), b["events"])
                 for b in F["batches"])
# 1
box(34,91.5,MAXW,9.6,"CHARLS 2021–2023 三年包","%s 条访谈记录" % format(F["source_records"], ","))
arrow(86.4,82.6)
# 2
box(34,78.3,MAXW,MH,"%s 名唯一受访者" % format(F["unique_participants"], ","))
side(78.3,"排除 %s 条重复记录\n（%s 个 ID 在同一年度包内被重复发布）"
     % (format(F["duplicate_records"], ","), format(F["duplicate_ids"], ",")))
arrow(73.8,70.4)
# 3
box(34,66.1,MAXW,MH,"%s 名可判定患病状态" % format(F["determinable"], ","))
side(66.1,"排除 %s 例\n新发与患病两题皆缺" % F["neither_item_answered"])
arrow(61.6,58.2)
# 4
box(34,53.9,MAXW,MH+1.0,"风险集：上次访问未患病","%s 例" % format(F["at_risk"], ","))
side(53.9,"基线现患 %s 例\n（进入构念对照，不入发病分析）" % format(F["prevalent_baseline"], ","),excl=False)
arrow(49.2,45.8)
# 5
box(34,38.6,MAXW,11.2,"分析样本","%s 人" % format(F["analytic"], ","),fc="#fdf7f8",ec="#dcb0b7",lw=1.05,ss=12.4)
side(38.6,"排除 %s 例\n%s 例上次访谈不在 2020 年\n%s 例年龄 <45 或 >95 岁"
     % (format(F["excluded_total"], ","), format(F["excluded_interview_not_2020"], ","),
        F["excluded_age_outside_45_95"]))

ax.add_patch(FancyArrowPatch((34,32.8),(34,27.4),arrowstyle="-",lw=0.7,color="#b8c3cf",
                             linestyle=(0,(3,2)),zorder=1))
ax.text(34,23.8,"新发关节炎／风湿病 %s 例（%.2f%%）" % (format(F["events"], ","), F["event_percent"]),
        ha="center",va="center",fontsize=10.2,color="#9b2335",fontweight="bold")
ax.text(34,18.0,"随访 %s 人年 ｜ 发病密度 %.2f / 100 人年"
        % (format(F["person_years"], ","), F["incidence_density_per_100py"]),
        ha="center",va="center",fontsize=9.0,color="#33414f")
ax.text(34,12.4,"访谈批次：" + BATCH,
        ha="center",va="center",fontsize=8.2,color="#5d6b7a")

plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig0_flow.png"),bbox_inches="tight",facecolor="white"); plt.close()
print("saved")
