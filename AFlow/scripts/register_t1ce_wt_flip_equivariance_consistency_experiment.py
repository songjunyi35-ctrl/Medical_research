#!/usr/bin/env python3
"""Register the preregistered flip-equivariance consistency experiment."""
from __future__ import annotations

from pathlib import Path
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def atomic(path: Path, text: str) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    temporary.replace(path)


def once(text: str, old: str, new: str, label: str) -> str:
    if text.count(old) != 1:
        raise RuntimeError(f"expected one {label}; found {text.count(old)}")
    return text.replace(old, new, 1)


def main() -> None:
    catalog_path = ROOT / "AFLOW_BRATS_METHOD_CATALOG.md"
    catalog = catalog_path.read_text(encoding="utf-8")
    row = "| Direct locator固定Flip8 TTA | 同一模型把MRI按8种镜像看完，翻回后平均 | ✅ 完成；reused test270三项点估计第1 | fresh A/B平均Scan/Subject `+0.006828/+0.007132`；test=`0.757270/0.741254/265` |"
    new_row = "| 镜像前后答案一致训练 | 训练时让原图与镜像图翻回后的最终mask一致 | 🧪 预注册完成；尚未实现 | fresh C/D、配对w0/w0.1、四模型完成后才访问inner；不进任何旧榜单cohort |"
    if new_row in catalog:
        raise RuntimeError("catalog row already registered")
    catalog = once(catalog, row, row + "\n" + new_row, "catalog anchor row")
    marker = "## 全自动Direct Locator固定Flip8测试时增强"
    card = """## 镜像前后答案一致训练 🧪 预注册完成；尚未实现

**一句话：** 同一道MRI正着看和镜像看，镜像答案翻回来后必须一致，希望模型最后正着只看一次也更稳。

**出发问题：** test访问前的Flip8 fresh A/B让Scan/Subject平均提高`+0.006828/+0.007132`，说明方向敏感可复现；但部署8×昂贵。下一步不在已揭示test搜融合，而是用全新train-only证据把等变性写进训练。

**具体做法：** 新C/D inner彼此subject互斥，并排除已访问A/B-inner subjects；每个复制训练完全配对的Control与Candidate。历史增强后，两臂都让同一student看原方向及按固定7种非identity镜像循环的一份输入，并把第二份logit翻回。历史DiceCE与知识蒸馏不变；唯一差别是soft-Dice一致性权重Control=`0`、Candidate=`0.1`。

**改变/固定因素：** 只改变一致性梯度权重。两臂同初值、sampler、增强、reflection schedule、两次student forward、teacher、模型、AdamW、late-cosine、epoch168、阈值0；无Segment Anything Model或后处理。

**预期机制：** 若Flip8收益来自模型未严格等变，直接惩罚最终mask的镜像前后差异应同时改善identity与Flip8。TCSM和Bortsova等一手论文支持这种逆变换后一致性；旧KAS匹配深层feature投影且混合三loss/晚期继续训练，不否定本实现。

**成功/停止门：** 每个复制Candidate Flip8须严格提高Scan/Subject、Point最多少1、两项CI下界≥`-0.003`；Candidate identity也须严格提高Scan/Subject且Point最多少1；最后5轮train-only一致性须≤Control的0.90×。跨复制Flip8 Scan平均>`0.001`且四个Dice平均方向均正。任一失败停止，不调weight/loss/view/epoch/seed，不访问fixed validation、outer、confirmation、test270或UCSD75。

**当前状态、代价与下一决定：** 七问与预注册已冻结；64个train-only病例的一致性尺度均值`0.08029`，据此在看新inner前固定weight0.1。预计四模型两波双A100、10–14h墙钟/20–28 GPU-hours、峰值35–50GiB且硬门60GiB、磁盘<150MB；部署参数不变，报告1×与8×。下一步只做隔离实现、合成测试和train-only GPU smoke。证据见[IDEA](archive/research_records/2026-09-04_flip_equivariance_consistency_stability/IDEA.md)、[预注册](archive/research_records/2026-09-04_flip_equivariance_consistency_stability/PREREG.md)、[train-only尺度审计](archive/research_records/2026-09-04_flip_equivariance_consistency_stability/TRAIN_ONLY_SCALE_AUDIT.json)、[TCSM](https://pubmed.ncbi.nlm.nih.gov/32479407/)与[Bortsova等](https://arxiv.org/abs/1911.01218)。

"""
    if card.splitlines()[0] in catalog:
        raise RuntimeError("catalog card already registered")
    if catalog.count(marker) != 1:
        raise RuntimeError("unexpected Flip8 card marker")
    catalog = catalog.replace(marker, card + marker, 1)

    handoff_path = ROOT / "AFLOW_BRATS_HANDOFF_CURRENT.md"
    handoff = handoff_path.read_text(encoding="utf-8")
    top_prefix = "- **Direct locator固定Flip8 TTA已完成并刷新reused test270全面点估计：**"
    lines = handoff.splitlines(keepends=True)
    matches = [i for i, line in enumerate(lines) if line.startswith(top_prefix)]
    if len(matches) != 1:
        raise RuntimeError("unexpected handoff Flip8 top line")
    active = "- **新活动实验：镜像前后答案一致训练已预注册、尚未实现。** 依据test访问前的Flip8 A/B平均增益`+0.006828/+0.007132`，用全新且对旧A/B-inner患者隔离、彼此互斥的C/D inner，完全配对训练Control w0与Candidate w0.1；两臂都执行固定第二镜像forward，只改变一致性梯度。四模型全部fixed168并冻结前禁止inner；任一门失败即停，明确禁止fixed validation/outer/confirmation/test270/UCSD75。预计双A100两波10–14h墙钟，先做合成合同与train-only GPU smoke。证据见[IDEA](archive/research_records/2026-09-04_flip_equivariance_consistency_stability/IDEA.md)与[预注册](archive/research_records/2026-09-04_flip_equivariance_consistency_stability/PREREG.md)。\n"
    if active in lines:
        raise RuntimeError("handoff active line already registered")
    lines.insert(matches[0] + 1, active)
    handoff = "".join(lines)
    handoff = once(
        handoff,
        "- 当前没有待续跑的训练，也没有已经授权但尚未执行的validation/test。下一session不能把任何旧分支误当成“继续跑完即可”。",
        "- 当前待续任务只有镜像前后答案一致训练：预注册已冻结但尚未实现/启动；任何旧分支仍不能误当成“继续跑完即可”。",
        "handoff active-state sentence",
    )
    handoff = once(
        handoff,
        "6. 当前没有活动训练或已预注册的下一方法。固定1–99 percentile clipping已完成双fresh-seed负检验并关闭；不要在已揭示inner上搜索其他百分位、顺序、epoch或teacher接线。",
        "6. 当前唯一活动方法是已冻结的镜像前后答案一致训练；先完成隔离实现、合成测试、train-only GPU smoke，再决定是否启动四模型。不要在任何已揭示inner/test上选择loss、weight、view、epoch或teacher接线。",
        "handoff next-session step6",
    )
    atomic(catalog_path, catalog)
    atomic(handoff_path, handoff)
    print("registered_flip_equivariance_consistency")


if __name__ == "__main__":
    main()
