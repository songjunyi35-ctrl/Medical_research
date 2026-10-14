#!/usr/bin/env python3
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=ROOT/"AFLOW_BRATS_METHOD_CATALOG.md"; s=p.read_text(); marker="## 当前推荐\n"
card='''## 当前推荐

### Full972多seed等权ensemble并蒸馏 🧪seed复现中

**一句话：** 再训练两个独立Full972模型，与当前seed32等权平均logits；若更稳，再把三模型共识蒸馏回1×student。

**状态：** 2026-08-25已冻结seed40/41、三模型各1/3公式和条件蒸馏门；3项合成测试及双卡smoke通过，两个168轮seed复现已授权。研究发生在test270揭示后，必须标post-test research。

**出发点：** Full972 seed32内部test为`0.751760/0.737327/262`，但只有一个随机轨迹；需评估随机方差并利用输出互补。旧parameter soup只否定跨seed参数平均，不否定logit平均。

**具体做法：** seed40/41完全复用Full972 epoch168协议；与seed32按raw logits各1/3。Validation不搜索seed或权重；ensemble过门才训练在线三teacher的1×student。

**唯一变化/固定项：** 训练只变seed；ensemble只加固定logit均值。数据、模型、epoch、draw、teacher、loss、阈值和后处理固定。

**成功/停止门：** Ensemble validation需超过seed32 Scan/Subject、Point≥101、两CI安全门；过门才蒸馏。任一训练>6h、24GiB、NaN或checkpoint失败即停。

**预计代价：** seed复现两卡约3.6h；ensemble3×；条件蒸馏再约3.6h并恢复1×；新增磁盘<1GB。

**下一决定：** 完成seed40/41后冻结三checkpoint并评validation108。证据：[七问](archive/research_records/2026-08-25_full972_multiseed_ensemble_distill/IDEA.md)、[预注册](archive/research_records/2026-08-25_full972_multiseed_ensemble_distill/PREREG.md)。
'''
if "### Full972多seed等权ensemble并蒸馏" not in s: s=s.replace(marker,card,1)
p.write_text(s)
h=ROOT/"AFLOW_BRATS_HANDOFF_CURRENT.md"; t=h.read_text(); line="- 2026-08-25 active post-test research：Full972 seed20260940/41双卡168轮复现已通过smoke并准备/正在运行；完成后与seed32按固定1/3等权logit ensemble，只用validation报告，不允许test选择seed/权重。归档`archive/research_records/2026-08-25_full972_multiseed_ensemble_distill/`。\n"
anchor="## 当前Leaderboard摘要\n"
if line not in t: t=t.replace(anchor,line+"\n"+anchor,1)
h.write_text(t)
