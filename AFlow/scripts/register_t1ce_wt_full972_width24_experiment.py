#!/usr/bin/env python3
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; p=ROOT/"AFLOW_BRATS_METHOD_CATALOG.md"; s=p.read_text(); marker="## 当前推荐\n"; card='''## 当前推荐

### Full972 width24完整轨迹 🧪inner训练中

**一句话：** Full972教材变厚后，把单模型从4.70M扩到10.57M参数，测试是否解除train-only欠容量。

**状态：** 2026-08-26已冻结width16/24单因素两阶段实验；4项合成测试和双卡smoke通过。Width24 peak30.28GiB，低于32GiB门；inner200轮已授权。

**出发点：** Full972 width16 train Dice低于Full756；此前160³和late adapter负结果未测试随机初始化的全程宽度扩展。

**具体做法：** 同candidate-fit864/common-inner108、seed20260960、968 draws/121步、teacher/loss/schedule；仅init_filters16→24。Inner过门后full972新seed重训同一epoch。

**唯一变化/固定项：** 只改模型宽度与参数量；数据、计算步数、teacher、loss、epoch选择、阈值与后处理固定。两者均1×forward。

**成功/停止门：** Inner需超过width16同epochScan/Subject、Point within1和CI门；final需超过matched width16及seed32 validation。Peak>32GiB、单臂>8h或工程失败即停。

**预计代价：** Inner约6h；过门后final再4–6h；新增磁盘<1GB。证据：[七问](archive/research_records/2026-08-26_full972_width24_fulltrajectory/IDEA.md)、[预注册](archive/research_records/2026-08-26_full972_width24_fulltrajectory/PREREG.md)。
''';
if "### Full972 width24完整轨迹" not in s: s=s.replace(marker,card,1)
p.write_text(s); h=ROOT/"AFLOW_BRATS_HANDOFF_CURRENT.md"; t=h.read_text(); line="- 2026-08-26 active post-test research：Full972 width16/24同candidate-fit864、common-inner108、seed20260960、968 draws/121步的200轮inner容量对照已通过smoke并准备/正在运行；width24 peak30.28GiB，硬门32GiB。完成前禁止新fixed validation/test访问。\n"; anchor="## 当前Leaderboard摘要\n";
if line not in t: t=t.replace(anchor,line+"\n"+anchor,1)
h.write_text(t)
