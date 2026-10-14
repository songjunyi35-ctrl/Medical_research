#!/usr/bin/env python3
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=ROOT/"AFLOW_BRATS_METHOD_CATALOG.md"; s=p.read_text(); start=s.index("### Full972多seed等权ensemble并蒸馏"); end=s.index("### 972病例随机初始化完整voxel-KD轨迹",start)
card='''### Full972多seed等权ensemble并蒸馏 ❌点估计正向/CI门失败

**一句话：** 新训练seed40/41并与seed32固定各1/3平均logits；若稳定过门再蒸馏回1×。

**状态：** 两个新seed168轮、工程门和固定ensemble validation已完成。Ensemble点估计明显超过seed32，但Subject CI安全门失败，按预注册不蒸馏、不访问新test。

**出发点：** Full972 seed32内部test为`0.751760/0.737327/262`，但只有一个随机轨迹；旧parameter soup不否定logit互补。

**具体做法：** seed40/41复用Full972协议；与seed32按raw logits各1/3，validation不搜索seed或权重。

**唯一变化/固定项：** 训练只变seed；ensemble只加固定logit均值。数据、模型、epoch、draw、teacher、loss、阈值和后处理固定。

**观察结果：** Ensemble=`0.745841/0.735684/104`；seed32=`0.740013/0.731266/103`；seed40=`0.733408/0.722941/101`；seed41=`0.736000/0.724980/103`。相对seed32 Scan/Subject=`+0.005829/+0.004418`，88/108 scans改善；Subject CI下界`-0.005660`失败`-0.003`门。

**是否支持直觉：** 部分支持。seed错误互补带来明显点估计收益，但50-subject层不确定性不足以授权条件蒸馏。

**实际代价：** 两卡约3.61h、每卡20.29GiB、workspace217MB；ensemble3×，未进入1×student阶段。

**下一决定：** 停止，不调权重、不剔除seed、不蒸馏、不访问新test。证据：[结果](archive/research_records/2026-08-25_full972_multiseed_ensemble_distill/RESULT.md)、[机器结果](archive/research_records/2026-08-25_full972_multiseed_ensemble_distill/RESULT.json)、[raw report](workspace/BraTS/sammed3d_t1ce_wt/full972_multiseed_20260825/THREE_SEED_ENSEMBLE_VALIDATION.json)。

'''
p.write_text(s[:start]+card+s[end:])
h=ROOT/"AFLOW_BRATS_HANDOFF_CURRENT.md"; t=h.read_text(); old="- 2026-08-25 active post-test research：Full972 seed20260940/41双卡168轮复现已通过smoke并准备/正在运行；完成后与seed32按固定1/3等权logit ensemble，只用validation报告，不允许test选择seed/权重。归档`archive/research_records/2026-08-25_full972_multiseed_ensemble_distill/`。"
new="- 2026-08-25 post-test multiseed结果：seed40/41完整168轮；seed32/40/41固定各1/3 ensemble validation=`0.745841/0.735684/104/108`，相对seed32=`+0.005829/+0.004418/+1`，但Subject CI下界`-0.005660`失败`-0.003`门。按预注册不调权重、不剔除seed、不蒸馏、不访问新test。证据`archive/research_records/2026-08-25_full972_multiseed_ensemble_distill/RESULT.md`。"
if old not in t: raise RuntimeError("active multiseed line missing")
h.write_text(t.replace(old,new))
