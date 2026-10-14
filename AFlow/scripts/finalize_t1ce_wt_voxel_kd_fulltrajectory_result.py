#!/usr/bin/env python3
"""Update canonical handoff/catalog after the frozen fulltrajectory result."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

catalog = ROOT / "AFLOW_BRATS_METHOD_CATALOG.md"
text = catalog.read_text()
start = text.index("### 972病例随机初始化完整voxel-KD轨迹")
end = text.index("### 972病例配额一致voxel-KD", start)
card = '''### 972病例随机初始化完整voxel-KD轨迹 ⚠️development冠军/一项CI门失败

**一句话：** 新增216例不再期末补课，而是从随机初始化第一轮就参与；先用公共train-inner选epoch，再重训full756/full972。

**状态：** 两阶段全部完成。Full972 epoch168在fixed validation108达到`0.740013/0.731266/103`，同时超过matched full756和KD200点估计，是当前development单模型冠军；只失败candidate-vs-KD200 Subject CI下界门，不访问test。

**出发点：** budget968 continuation达到`0.734259/0.724956/100`，但新增病例只在history168后出现。完整轨迹测试新增病例从表示学习早期参与。

**具体做法：** Inner control648/candidate864共享108-case subject-clean tune，两臂同968 draws/121步跑200轮并由candidate选择epoch168；随后full756/full972以新seed、相同随机初始state和计算量重训到168。

**唯一变化/固定项：** 只改eligible病例池；inner tune、计算量、随机初始state、模型、teacher、loss、LR、增强、阈值与1×推理固定。

**成功/停止门：** Final必须超过full control与KD200 Scan/Subject、Point≥101、四项CI下界≥-0.003才授权test。

**观察结果：** Candidate=`0.740013/0.731266/103`；full756 control=`0.729070/0.723284/104`；KD200=`0.734384/0.724434/101`。相对control Scan/Subject=`+0.010943/+0.007982`，Scan CI整体为正；相对KD200=`+0.005628/+0.006832/+2 points`，但Subject CI下界`-0.004484`低于`-0.003`门。

**是否支持直觉：** 强支持。新增216例从随机初始化参与明显优于late continuation；失败的是保守subject CI授权，不是点估计或机制。

**实际代价：** Inner约4.42h、final约3.61h，两卡总墙钟约8h；每卡20.29GiB；推理仍1×。

**下一决定：** 保留为development champion，不在当前validation放宽CI门或调参，不进test。证据：[结果](archive/research_records/2026-08-24_voxel_kd_fulltrajectory_972/RESULT.md)、[机器结果](archive/research_records/2026-08-24_voxel_kd_fulltrajectory_972/RESULT.json)、[raw report](workspace/BraTS/sammed3d_t1ce_wt/voxel_kd_fulltrajectory_972_20260824/FULLTRAJECTORY_VALIDATION.json)。

'''
catalog.write_text(text[:start] + card + text[end:])

handoff = ROOT / "AFLOW_BRATS_HANDOFF_CURRENT.md"
text = handoff.read_text().replace("更新时间：2026-08-24 UTC", "更新时间：2026-08-25 UTC")
old_active = "- 2026-08-24 active experiment：972病例随机初始化完整voxel-KD轨迹阶段A已完成并通过inner门，选择epoch168。阶段B final双卡smoke已通过：full756/full972 seed20260932初始state exact、相同968 draws/121步、20.29GiB；正式重训即将/正在worker1061661后台运行。Final完成前禁止fixed validation108/test访问。证据`archive/research_records/2026-08-24_voxel_kd_fulltrajectory_972/INNER_SELECTION.json`。"
new_active = "- 2026-08-25最新结果：972病例随机初始化完整voxel-KD轨迹两阶段已完成。Full972 epoch168 fixed validation108 Scan/Subject/Point=`0.740013/0.731266/103/108`，matched full756=`0.729070/0.723284/104/108`，KD200=`0.734384/0.724434/101/108`。Candidate同时超过两者点估计，是当前development单模型冠军；相对KD200 Subject CI下界`-0.004484`失败冻结`-0.003`门，因此test270未访问。证据`archive/research_records/2026-08-24_voxel_kd_fulltrajectory_972/RESULT.md`。"
if old_active not in text:
    raise RuntimeError("active handoff line not found")
text = text.replace(old_active, new_active)
old_next = "当前没有满足全部门槛的自动候选。Alpha0.5仍是最高development单模型Scan（`0.734683`）但Point少1例；新budget968 candidate取得当前development单模型最高Subject `0.724956`，Scan只比KD200低`0.000125`，Point也只少1例。数据多样性和预算机制都获支持，但不能用本validation事后延长epoch或改Point门；下一步必须换独立证据或新目标，而不是继续沿同一validation微调。"
new_next = "当前没有满足全部门槛的test候选，但full972 random-init KD168已成为development单模型冠军：`0.740013/0.731266/103`，同时超过KD200点估计。唯一阻塞是相对KD200 Subject CI下界`-0.004484`未过`-0.003`安全门。不能事后放宽；需要独立subject-level确认或显式改变科研目标。"
if old_next not in text:
    raise RuntimeError("next-decision paragraph not found")
text = text.replace(old_next, new_next)
marker = "- 最新972预算机制正向/极近KD200但未晋级结果："
latest = "- 最新full972完整轨迹development冠军/CI门失败结果：[`archive/research_records/2026-08-24_voxel_kd_fulltrajectory_972/RESULT.md`](archive/research_records/2026-08-24_voxel_kd_fulltrajectory_972/RESULT.md)\n"
if latest not in text and marker in text:
    pos = text.index(marker)
    text = text[:pos] + latest + text[pos:]
handoff.write_text(text)
