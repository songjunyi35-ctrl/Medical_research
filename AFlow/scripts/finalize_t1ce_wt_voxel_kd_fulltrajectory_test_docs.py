#!/usr/bin/env python3
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
hp=ROOT/"AFLOW_BRATS_HANDOFF_CURRENT.md"; h=hp.read_text()
h=h.replace("| 总体最高 | `0.75 history168 + 0.25 EMA168` logit ensemble | `0.745054` | `0.727345` | `0.955556` | `2×` locator |\n| 最高单模型 | 2025/2026 voxel-KD global epoch200 | `0.742976` | `0.725179` | `0.955556` | `1×` locator |","| 总体最高（探索性） | Full972 random-init voxel-KD epoch168 | `0.751760` | `0.737327` | `0.970370` | `1×` locator |\n| 最高单模型（探索性） | Full972 random-init voxel-KD epoch168 | `0.751760` | `0.737327` | `0.970370` | `1×` locator |")
old="- 2026-08-25最新结果：972病例随机初始化完整voxel-KD轨迹两阶段已完成。Full972 epoch168 fixed validation108 Scan/Subject/Point=`0.740013/0.731266/103/108`，matched full756=`0.729070/0.723284/104/108`，KD200=`0.734384/0.724434/101/108`。Candidate同时超过两者点估计，是当前development单模型冠军；相对KD200 Subject CI下界`-0.004484`失败冻结`-0.003`门，因此test270未访问。证据`archive/research_records/2026-08-24_voxel_kd_fulltrajectory_972/RESULT.md`。"
new="- 2026-08-25最新结果：Full972 random-init KD168 validation一项Subject CI门失败后，用户明确授权一次failed-gate exploratory reused test270。Test Scan/Subject/Point=`0.751760/0.737327/262/270`，相对旧ensemble=`+0.006705/+0.009982/+4 points`、相对KD200=`+0.008783/+0.012149/+4 points`，两组Scan/Subject paired CI下界均大于0，成为当前内部test总体/单模最高点估计。证据不是pristine blind或正式promotion，禁止据此调参。证据`archive/research_records/2026-08-24_voxel_kd_fulltrajectory_972/RESULT.md`。"
if old in h: h=h.replace(old,new)
h=h.replace("当前没有满足全部门槛的test候选，但full972 random-init KD168已成为development单模型冠军：`0.740013/0.731266/103`，同时超过KD200点估计。唯一阻塞是相对KD200 Subject CI下界`-0.004484`未过`-0.003`安全门。不能事后放宽；需要独立subject-level确认或显式改变科研目标。","Full972 random-init KD168现为内部reused test总体/单模最高点估计：`0.751760/0.737327/262/270`，相对旧ensemble和KD200的Scan/Subject paired CI均整体为正。该test由用户在validation CI门失败后明确授权，必须持续标记failed-gate exploratory、非盲测、非外部确认；禁止后续test调参。")
hp.write_text(h)
cp=ROOT/"AFLOW_BRATS_METHOD_CATALOG.md"; c=cp.read_text()
c=c.replace("### 972病例随机初始化完整voxel-KD轨迹 ⚠️development冠军/一项CI门失败","### 972病例随机初始化完整voxel-KD轨迹 ✅探索性内部test新高/failed-gate标记")
c=c.replace("**状态：** 两阶段全部完成。Full972 epoch168在fixed validation108达到`0.740013/0.731266/103`，同时超过matched full756和KD200点估计，是当前development单模型冠军；只失败candidate-vs-KD200 Subject CI下界门，不访问test。","**状态：** 两阶段完成；validation只失败KD200 Subject CI门。用户随后明确授权一次failed-gate exploratory reused test270，取得`0.751760/0.737327/262/270`，相对旧ensemble和KD200的Scan/Subject CI均整体为正，成为当前内部总体/单模新高；不是盲测或正式promotion。")
c=c.replace("**下一决定：** 保留为development champion，不在当前validation放宽CI门或调参，不进test。证据：","**Test结果：** 相对ensemble=`+0.006705/+0.009982/+4 points`，相对KD200=`+0.008783/+0.012149/+4 points`，四项paired CI下界均大于0。禁止用test继续调参。\n\n**下一决定：** 保留为failed-gate exploratory内部新高，寻求外部/独立确认。证据：")
cp.write_text(c)
