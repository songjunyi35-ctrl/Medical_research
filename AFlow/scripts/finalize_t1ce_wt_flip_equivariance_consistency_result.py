#!/usr/bin/env python3
"""Close the flip-consistency experiment and write the GPT-6 handoff state."""
from pathlib import Path
import tempfile

ROOT=Path(__file__).resolve().parents[1]

def atomic(path:Path,text:str)->None:
    with tempfile.NamedTemporaryFile("w",encoding="utf-8",dir=path.parent,delete=False) as h:
        h.write(text); tmp=Path(h.name)
    tmp.replace(path)

def once(text:str,old:str,new:str,label:str)->str:
    if text.count(old)!=1: raise RuntimeError(f"expected one {label}; found {text.count(old)}")
    return text.replace(old,new,1)

def prefix(text:str,start:str,replacement:str,label:str)->str:
    lines=text.splitlines(keepends=True); hits=[i for i,x in enumerate(lines) if x.startswith(start)]
    if len(hits)!=1: raise RuntimeError(f"expected one {label}; found {len(hits)}")
    lines[hits[0]]=replacement+("\n" if lines[hits[0]].endswith("\n") else "")
    return "".join(lines)

def update_catalog()->None:
    p=ROOT/"AFLOW_BRATS_METHOD_CATALOG.md"; s=p.read_text()
    s=prefix(s,"| 镜像前后答案一致训练 |","| 镜像前后答案一致训练 | 训练时让原图与镜像图翻回后的最终mask一致 | ❌ fresh C/D失败并关闭 | 一致性降35–36%，但8× Scan平均`-0.001918`、identity Scan平均`-0.000115` |","catalog row")
    start=s.index("## 镜像前后答案一致训练")
    end=s.index("## 全自动Direct Locator固定Flip8测试时增强",start)
    card="""## 镜像前后答案一致训练 ❌ fresh C/D失败并关闭

**一句话：** 训练时让同一MRI正着看和镜像看，镜像答案翻回来后必须一致，希望模型正着一次和8镜像平均都更准。

**状态：** 四条fixed168训练、四checkpoint预访问冻结和fresh C/D inner一次性评测全部完成；主门和跨复制门失败。没有访问fixed validation、outer、confirmation、test270或UCSD75。

**出发问题：** test访问前的Flip8 A/B让Scan/Subject平均提高`+0.006828/+0.007132`，说明方向敏感可复现，但8×推理昂贵。这个实验尝试只用新train-role证据把等变性学进模型。

**具体做法：** C/D各864 fit/108 inner、各49 inner subjects，彼此case/subject互斥并排除旧A/B inner的93个subjects。每复制Control/Candidate同初值、sampler、增强、teacher和两次student forward；只把平方分母sigmoid soft-Dice一致性梯度从0改为0.1。四模型全部完成后才打开inner，同时评identity1×与固定Flip8 8×。

**改变/固定因素：** 唯一变化是一致性weight0/0.1。模型4.70M、T1ce/WT、brain crop true128³、DiceCE+逐体素知识蒸馏、native-volume sampling、AdamW、168轮、reflection cycle、阈值0、无Segment Anything Model/后处理全部固定。

**预期机制：** 若Flip8收益来自模型不严格等变，直接减少最终mask镜像差异应同时改善1×与8×。旧KAS是晚期feature投影加三loss，未提前否定本实现。

**成功/停止门：** 每复制Candidate Flip8须严格提高Scan/Subject、Point最多少1、两项CI下界≥`-0.003`；identity也须严格提高Scan/Subject；末5轮一致性≤Control 0.90×。跨复制Flip8 Scan平均>`0.001`且四项Dice平均方向为正。任一失败停止，不调weight/formula/view/epoch/seed，不进旧cohort。

**观察结果：** C identity Candidate−Control=`+0.000316/+0.001386/+1`，Flip8=`-0.001773/+0.000643/+2`，Flip8 CI=`[-0.006636,+0.003320]/[-0.005893,+0.007807]`。D identity=`-0.000546/+0.000085/-1`，Flip8=`-0.002063/-0.003967/-1`，CI=`[-0.005778,+0.001629]/[-0.009182,+0.000867]`。跨复制平均Flip8 Scan/Subject=`-0.001918/-0.001662`，identity Scan/Subject=`-0.000115/+0.000735`；整体失败。

**直觉是否支持：** 不支持准确率主张。机制确实学到：末5轮Candidate/Control一致性loss比为C`0.636`、D`0.648`，即差异少约35–36%；但更一致会把错误也稳定重复，未变成更准。只否定当前weight0.1、平方soft-Dice、单reflection循环实现，不证明所有群等变架构不可能。

**代价与下一决定：** 四训练约20.873 GPU-hours/双卡两波10.484h，每卡峰值42.006GB；inner评测3,456 forwards；workspace110MB，无新cache。原普通soft-Dice公式混入锐化，在训练前以AMENDMENT_01透明修正，未看inner。关闭分支，不调参、不加seed、不进榜。C/D已经揭示，下一内部方法必须新E/F并排除A/B/C/D inner subjects。证据见[结果](archive/research_records/2026-09-04_flip_equivariance_consistency_stability/RESULT.md)、[机器结果](archive/research_records/2026-09-04_flip_equivariance_consistency_stability/RESULT.json)、[raw inner](archive/research_records/2026-09-04_flip_equivariance_consistency_stability/RAW_STAGE_A_INNER.json)、[preaccess](archive/research_records/2026-09-04_flip_equivariance_consistency_stability/INNER_PREACCESS.json)与[修订](archive/research_records/2026-09-04_flip_equivariance_consistency_stability/AMENDMENT_01.md)。

"""
    s=s[:start]+card+s[end:]; atomic(p,s)

def update_idea()->None:
    p=ROOT/"archive/research_records/2026-09-04_flip_equivariance_consistency_stability/IDEA.md"; s=p.read_text()
    s=once(s,"状态：七问已冻结；尚未生成split、实现、构造模型、训练或访问新inner","状态：完成并关闭；一致性机制成立但fresh C/D准确率门失败，未访问任何旧榜单cohort","IDEA status")
    start=s.index("## 8. 最终结果（完成后填写）")
    final="""## 8. 最终结果

- Fresh C identity Candidate−Control=`+0.000316/+0.001386/+1`，Flip8=`-0.001773/+0.000643/+2`；Fresh D identity=`-0.000546/+0.000085/-1`，Flip8=`-0.002063/-0.003967/-1`。
- C Flip8 Scan/Subject 95% CI=`[-0.006636,+0.003320]/[-0.005893,+0.007807]`；D=`[-0.005778,+0.001629]/[-0.009182,+0.000867]`。两复制平均Flip8 Scan/Subject=`-0.001918/-0.001662`，identity Scan/Subject=`-0.000115/+0.000735`。
- 机制门成立：最后5轮Candidate/Control一致性loss比例C=`0.636`、D=`0.648`；工程配对全部exact。但更一致没有更准，直觉的准确率部分不受支持。
- 四模型fixed168并冻结后才访问C/D inner；没有访问fixed validation、outer、confirmation、test270、UCSD75或其他外部cohort。
- 原公式在相等预测时loss非0，训练前透明AMENDMENT_01改为平方分母；没有用inner修公式。
- 关闭当前weight0.1实现，不调参、不加seed、不进榜。完整证据见[RESULT.md](RESULT.md)和[RESULT.json](RESULT.json)。
"""
    s=s[:start]+final; atomic(p,s)

def update_handoff()->None:
    p=ROOT/"AFLOW_BRATS_HANDOFF_CURRENT.md"; s=p.read_text()
    s=once(s,"更新时间：2026-09-04 UTC","更新时间：2026-09-05 UTC","handoff date")
    authorization="""## GPT-6接手：用户明确的30小时长周期授权

- 用户要求GPT-6接手后连续科研30小时，不因单次负结果或等待训练而提前停止；两张A100、充足credit和联网检索均获授权。
- 可自主完成仓库范围内的只读审计、联网一手文献检索、七问、预注册、隔离实现、测试、双卡训练、合法fresh-inner/新外部评测、归档和canonical文档更新；不需要为常规安全科研步骤逐项等待确认。
- “权限全开”不覆盖安全与证据规范：不得用已揭示test270、UCSD75、fixed validation、outer、confirmation或C/D inner选择loss、weight、epoch、view、阈值、checkpoint或后处理；不得把reused/post-test结果冒充blind/external；不得静默删除原始数据、代码、checkpoint或报告。
- 磁盘不足时，用户允许自主删除**已审计、可重建、已有哈希/归档证据的派生缓存**；删除前仍须精确核对目标，保留原始数据、checkpoint、代码和报告，并在交接记录删除内容与可恢复性。当前已知可回收候选是`workspace/BraTS/cache/t1ce_wt_n4_braincrop_resolution_20260818/formal_128`，审计证明756例完全包含于现972例cache，可释放约7.93GB；只有实际需要时再删。
- 执行循环：完整链路审视→用当前数字向初学者解释七问→冻结IDEA/PREREG→新entry point与合成测试→比例GPU smoke→formal→预访问receipt→合法评测→正负结果归档/更新handoff+catalog；若失败，关闭该实现并立即转向下一条未被否定的单因素路线，直到30小时结束或出现真正外部阻碍。

"""
    marker="## 交接给下一session：当前停点\n\n"
    s=once(s,marker,authorization+marker,"GPT-6 authorization section")
    s=prefix(s,"- **新活动实验：镜像前后答案一致训练formal C pair训练中。**","- **镜像前后答案一致训练fresh C/D失败并关闭：** C/D各864 fit/108 inner、各49 subjects，彼此case/subject互斥且排除旧A/B的93个inner subjects。四条fixed168训练全部完成并冻结后才访问inner；两pair初值、sampling、168轮sampler、增强RNG、reflection、首batch/首loss全部exact。Candidate确实把末5轮一致性loss降到Control的`0.636/0.648`，但C Flip8 Scan/Subject/Point=`-0.001773/+0.000643/+2`，D=`-0.002063/-0.003967/-1`；两复制平均Flip8 Scan/Subject=`-0.001918/-0.001662`，identity Scan/Subject=`-0.000115/+0.000735`。机制成立但准确率直觉不成立，`all_gates_passed=false`；不调weight/formula/view/epoch/seed，不访问fixed validation/outer/confirmation/test270/UCSD75。四训练20.873 GPU-hours/双卡10.484h、峰值42.006GB、inner3,456 forwards、workspace110MB。证据见[结果](archive/research_records/2026-09-04_flip_equivariance_consistency_stability/RESULT.md)、[机器结果](archive/research_records/2026-09-04_flip_equivariance_consistency_stability/RESULT.json)、[raw inner](archive/research_records/2026-09-04_flip_equivariance_consistency_stability/RAW_STAGE_A_INNER.json)、[preaccess](archive/research_records/2026-09-04_flip_equivariance_consistency_stability/INNER_PREACCESS.json)与[修订](archive/research_records/2026-09-04_flip_equivariance_consistency_stability/AMENDMENT_01.md)。","handoff result line")
    s=prefix(s,"- 当前待续任务只有镜像前后答案一致训练：","- 当前没有运行中的训练或已授权待执行的旧cohort评测。GPT-6应按上面的30小时循环从全链路审视和一手文献检索开始，提出新方法后先七问/预注册；不得重启任何旧runner。","handoff active state")
    s=prefix(s,"- 当前内部test最高Scan点估计checkpoint：","- 当前内部test总体三项最高点估计是Full972 KD168固定Flip8（8×）：`0.757270/0.741254/265/270`，checkpoint SHA-256 `f7df2a1678564268e899d20411e55c0163df480498273721a7e5b1b71c80d17b`。1×范围最高Scan仍是GCSAM fit864 epoch184；1×最高Subject/Point仍是Full972 KD168 identity。","handoff best checkpoint")
    lines=s.splitlines(keepends=True); hits=[i for i,x in enumerate(lines) if x.startswith("6. 当前唯一活动方法是镜像前后答案一致训练；")]
    if len(hits)!=1: raise RuntimeError("expected one handoff step6")
    lines[hits[0]]="6. 镜像前后答案一致训练已完成负结果并关闭；C/D已揭示，不得用于下一方法选择。若继续内部研究，建立新的E/F并排除A/B/C/D全部inner subjects且彼此患者隔离；否则优先解决RHUH等新人工标签外部cohort的官方inventory/hash。\n"
    s="".join(lines)
    s=once(s,"5. 新idea必须离开已关闭的容量、全图160³、旧loss/feature-KD、难例聚合、固定/自适应多seed融合以及本session已判负的LAWA6/Lookahead/Gamma轴；仍需七问与预注册后才实施。","5. 新idea必须离开已关闭的容量、全图160³、旧loss/feature-KD、难例聚合、固定/自适应多seed融合、LAWA6/Lookahead/Gamma，以及本次weight0.1平方soft-Dice镜像一致性轴；仍需七问与预注册后才实施。","handoff closed axes")
    atomic(p,s)

def main()->None:
    update_catalog(); update_idea(); update_handoff(); print("finalized_flip_consistency_and_gpt6_handoff")

if __name__=="__main__": main()
