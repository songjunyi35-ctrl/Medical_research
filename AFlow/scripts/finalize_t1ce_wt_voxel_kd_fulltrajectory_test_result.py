#!/usr/bin/env python3
"""Archive user-authorized fulltrajectory test270 and update both leaderboards."""
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/"workspace/BraTS/sammed3d_t1ce_wt/voxel_kd_fulltrajectory_972_20260824/FULLTRAJECTORY_TEST270.json"
REPORT_SHA="9827a75b88dc9239902c7e473a7d75e2d6058f4f515bfd2b4d93003ba7bd34cd"
CKPT_SHA="f7df2a1678564268e899d20411e55c0163df480498273721a7e5b1b71c80d17b"
d=json.loads(REPORT.read_text()); s=d["candidate"]

jp=ROOT/"AFLOW_BRATS_LEADERBOARD.json"; board=json.loads(jp.read_text())
new={"rank":1,"method":"full972_random_init_voxel_kd_epoch168","scan_dice":s["locator_dsc_mean"],"subject_dice":s["locator_subject_dsc_mean"],"point_hit":s["automatic_point_hit_rate"],"locator_forwards":1,"cohort":{"scans":270,"subjects":119},"checkpoint_sha256":CKPT_SHA,"resolution":128,"threshold":0.0,"tta":False,"sam":False,"postprocessing":False,"evidence_status":"user_authorized_after_failed_validation_ci_gate; reused_internal_test; not_pristine_blind; not_external; no_test_tuning_allowed","report_path":str(REPORT.relative_to(ROOT)),"report_sha256":REPORT_SHA,"status":"best_overall_and_single_reused_internal_test_point_estimate_failed_gate_exploratory"}
items=[x for x in board["test270_direct_locator_rank_by_scan"] if x.get("method")!=new["method"]]+[new]
items.sort(key=lambda x:x["scan_dice"],reverse=True)
for i,x in enumerate(items,1): x["rank"]=i
board["test270_direct_locator_rank_by_scan"]=items
vnew={"rank":1,"method":"full972_random_init_voxel_kd_epoch168","scan_dice":0.7400125352044901,"subject_dice":0.7312662169195357,"point_hit":103/108,"test_status":"failed_gate_user_authorized_exploratory"}
vals=[x for x in board["validation108_direct_locator_rank_by_scan"] if x.get("method")!=vnew["method"]]+[vnew]
vals.sort(key=lambda x:x["scan_dice"],reverse=True)
for i,x in enumerate(vals,1): x["rank"]=i
board["validation108_direct_locator_rank_by_scan"]=vals; board["updated_utc"]="2026-08-25"
jp.write_text(json.dumps(board,indent=2,sort_keys=False)+"\n")

mp=ROOT/"AFLOW_BRATS_LEADERBOARD.md"; text=mp.read_text().replace("更新时间：2026-08-19 UTC","更新时间：2026-08-25 UTC")
start=text.index("| 排名 | 方法 | Scan Dice | Subject Dice | Point hit | 成本 | 状态 |",text.index("## Reused internal test270")); end=text.index("\n\n关键增益：",start)
table='''| 排名 | 方法 | Scan Dice | Subject Dice | Point hit | 成本 | 状态 |
|---:|---|---:|---:|---:|---:|---|
| **1** | **Full972 random-init voxel-KD，epoch168** | **`0.751759537`** | **`0.737327107`** | **`0.970370`** | **`1×`** | 用户授权failed-gate探索性；当前总体/单模最高点估计 |
| 2 | 固定logit ensemble：0.75 history168 + 0.25 EMA168 | `0.745054064` | `0.727344609` | `0.955556` | `2×` | 旧总体最高 |
| 3 | 2025/2026 voxel-KD single，global epoch200 | `0.742976353` | `0.725178533` | `0.955556` | `1×` | 旧最高单模型 |
| 4 | historical true-128³ plain，epoch168 | `0.741203949` | `0.723179486` | `0.929630` | `1×` | 旧单模型记录 |
| 5 | EMA168 single | `0.740696468` | `0.724904822` | `0.948148` | `1×` | Dice持平、point更高 |
| 6 | true-160³ full756，epoch200 | `0.737060297` | `0.719728630` | `0.874074` | `~1.95×` | 负结果；160³标签网格 |
| 7 | true-128³ plain，epoch99 | `0.725820068` | `0.709412503` | `0.900000` | `1×` | 2026-08-17记录 |'''
text=text[:start]+table+text[end:]
key="关键增益：\n"
gain="\n- Full972 − ensemble：scan `+0.006705473`、subject `+0.009982499`、point `+0.014815`；scan/subject paired 95% CI下界均大于0。\n- Full972 − KD200：scan `+0.008783184`、subject `+0.012148574`、point `+0.014815`；scan/subject paired 95% CI下界均大于0。\n"
text=text.replace(key,key+gain,1)
report_marker="报告入口：\n"
text=text.replace(report_marker,report_marker+"\n- Full972：[报告](workspace/BraTS/sammed3d_t1ce_wt/voxel_kd_fulltrajectory_972_20260824/FULLTRAJECTORY_TEST270.json)，[归档](archive/research_records/2026-08-24_voxel_kd_fulltrajectory_972/RESULT.md)\n",1)
vstart=text.index("| 排名 | 方法 | Scan Dice | Subject Dice | Point hit | 是否进test | 结论 |",text.index("## Fixed validation108")); vend=text.index("\n\n## 非同口径",vstart)
oldrows=text[vstart:vend].splitlines()[2:]
reranked=[]
for line in oldrows:
 if line.startswith("| **1**"):
  line=line.replace("| **1** |","| 2 |",1).replace("**voxel-KD global epoch200**","voxel-KD global epoch200").replace("**`0.734384308`**","`0.734384308`")
 else:
  parts=line.split("|");
  if len(parts)>2:
   try: parts[1]=f" {int(parts[1].strip())+1} "; line="|".join(parts)
   except ValueError: pass
 reranked.append(line)
vtable="| 排名 | 方法 | Scan Dice | Subject Dice | Point hit | 是否进test | 结论 |\n|---:|---|---:|---:|---:|---|---|\n| **1** | **Full972 random-init KD epoch168** | **`0.740012535`** | **`0.731266217`** | **`0.953704`** | 用户授权门后探索性 | 当前development最高 |\n"+"\n".join(reranked)
text=text[:vstart]+vtable+text[vend:]
text=text.replace("- 最高内部test点估计：固定`0.75 history168 + 0.25 EMA168` logit ensemble。","- 最高内部test点估计：Full972 random-init voxel-KD epoch168（1×）；但证据为validation failed-gate后用户授权探索性test。")
text=text.replace("- 最佳单模型部署：2025/2026 voxel-KD global epoch200。","- 最佳单模型点估计：Full972 random-init voxel-KD epoch168，1×推理。")
mp.write_text(text)

rp=ROOT/"archive/research_records/2026-08-24_voxel_kd_fulltrajectory_972/RESULT.md"; text=rp.read_text();
if "## 用户授权failed-gate test270" not in text:
 text += '''\n## 用户授权failed-gate test270\n\n用户在得知validation Subject CI门失败后，明确授权对冻结checkpoint运行一次探索性reused internal test270。该结果不是pristine blind、外部确认或正式promotion，且禁止触发调参。\n\nFull972 Scan/Subject/Point=`0.751760/0.737327/262/270`，成为当前内部test总体和单模型最高点估计。相对旧ensemble为`+0.006705/+0.009982/+4 points`，Scan/Subject 95% CI分别`[+0.000784,+0.013006]`/`[+0.001524,+0.019404]`；相对KD200为`+0.008783/+0.012149/+4 points`，CI分别`[+0.002994,+0.015118]`/`[+0.003561,+0.021742]`。\n\nTest报告SHA `9827a75b88dc9239902c7e473a7d75e2d6058f4f515bfd2b4d93003ba7bd34cd`。Test已访问；仍标记failed-gate exploratory，不据此调epoch/loss/threshold/postprocess。\n'''
 rp.write_text(text)
rj=ROOT/"archive/research_records/2026-08-24_voxel_kd_fulltrajectory_972/RESULT.json"; x=json.loads(rj.read_text()); x["status"]="failed_gate_user_authorized_test_new_internal_high"; x["test270"]={"scan":s["locator_dsc_mean"],"subject":s["locator_subject_dsc_mean"],"point_hit":s["automatic_point_hit_rate"],"report_sha256":REPORT_SHA,"evidence_status":"failed_gate exploratory reused internal test; not pristine blind"}; x["access"]["test270"]=True; rj.write_text(json.dumps(x,indent=2,sort_keys=True)+"\n")
