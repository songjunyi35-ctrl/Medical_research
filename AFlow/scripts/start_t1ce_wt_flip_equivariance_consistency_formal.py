#!/usr/bin/env python3
"""Record implementation/smoke success immediately before formal launch."""
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


def replace_prefix(text: str, prefix: str, replacement: str, label: str) -> str:
    lines = text.splitlines(keepends=True)
    matches = [i for i, line in enumerate(lines) if line.startswith(prefix)]
    if len(matches) != 1:
        raise RuntimeError(f"expected one {label}; found {len(matches)}")
    lines[matches[0]] = replacement + ("\n" if lines[matches[0]].endswith("\n") else "")
    return "".join(lines)


def main() -> None:
    catalog_path = ROOT / "AFLOW_BRATS_METHOD_CATALOG.md"
    catalog = catalog_path.read_text(encoding="utf-8")
    catalog = replace_prefix(
        catalog,
        "| 镜像前后答案一致训练 |",
        "| 镜像前后答案一致训练 | 训练时让原图与镜像图翻回后的最终mask一致 | 🧪 实现/smoke通过；formal准备启动 | C/D inner互斥且排除旧A/B；配对smoke全exact、峰值42.01GB、inner未访问 |",
        "catalog table row",
    )
    catalog = once(catalog, "## 镜像前后答案一致训练 🧪 预注册完成；尚未实现", "## 镜像前后答案一致训练 🧪 实现/smoke通过；formal准备启动", "catalog header")
    old = "**当前状态、代价与下一决定：** 七问与预注册已冻结；64个train-only病例的一致性尺度均值`0.08029`，据此在看新inner前固定weight0.1。预计四模型两波双A100、10–14h墙钟/20–28 GPU-hours、峰值35–50GiB且硬门60GiB、磁盘<150MB；部署参数不变，报告1×与8×。下一步只做隔离实现、合成测试和train-only GPU smoke。证据见[IDEA](archive/research_records/2026-09-04_flip_equivariance_consistency_stability/IDEA.md)、[预注册](archive/research_records/2026-09-04_flip_equivariance_consistency_stability/PREREG.md)、[train-only尺度审计](archive/research_records/2026-09-04_flip_equivariance_consistency_stability/TRAIN_ONLY_SCALE_AUDIT.json)、[TCSM](https://pubmed.ncbi.nlm.nih.gov/32479407/)与[Bortsova等](https://arxiv.org/abs/1911.01218)。"
    new = "**当前状态、代价与下一决定：** 静态审查发现原soft-Dice分母在预测相等时不为0，会混入置信度锐化；训练前透明冻结AMENDMENT_01，改为平方分母，相等loss严格0。V2 train-only尺度均值`0.04784`，保留weight0.1。真实C/D各864/108、各49 inner subjects，彼此case/subject overlap0并排除旧A/B的93个inner subjects。29项合同/回归测试和双A100两步配对smoke全过；初值、sampling、batch、existing/consistency loss、sampler/RNG/reflection exact，Candidate参数按0.1项分叉，峰值42.01GB<60GiB，inner未访问。预计四模型两波10–14h墙钟/20–28 GPU-hours、磁盘<150MB；下一步启动formal C pair→D pair，四checkpoint全部完成才冻结inner。证据见[IDEA](archive/research_records/2026-09-04_flip_equivariance_consistency_stability/IDEA.md)、[预注册](archive/research_records/2026-09-04_flip_equivariance_consistency_stability/PREREG.md)、[修订](archive/research_records/2026-09-04_flip_equivariance_consistency_stability/AMENDMENT_01.md)、[实现凭证](archive/research_records/2026-09-04_flip_equivariance_consistency_stability/IMPLEMENTATION_RECEIPT.json)与[V2尺度审计](archive/research_records/2026-09-04_flip_equivariance_consistency_stability/TRAIN_ONLY_SCALE_AUDIT_V2.json)。"
    catalog = once(catalog, old, new, "catalog current status")

    handoff_path = ROOT / "AFLOW_BRATS_HANDOFF_CURRENT.md"
    handoff = handoff_path.read_text(encoding="utf-8")
    handoff = replace_prefix(
        handoff,
        "- **新活动实验：镜像前后答案一致训练已预注册、尚未实现。**",
        "- **新活动实验：镜像前后答案一致训练实现/smoke通过，formal准备启动。** 方法由test访问前Flip8 A/B增益驱动；C/D各864 fit/108 inner、各49 inner subjects，C/D互斥且排除旧A/B的93个inner subjects。原公式静态审查发现相等预测loss不为0，训练前AMENDMENT_01改为平方分母；V2 train-only均值`0.04784`、weight仍0.1。Control/Candidate每步都跑相同两次student forward，只改变一致性梯度0/0.1。29项测试和双卡两步smoke全过：初值/sampling/batch/首existing与consistency/sampler/RNG/reflection exact，Candidate参数分叉，峰值42.01GB，inner未访问。下一步两波双A100 C pair→D pair；四模型fixed168并冻结前禁止inner，且全程禁止fixed validation/outer/confirmation/test270/UCSD75。证据见[IDEA](archive/research_records/2026-09-04_flip_equivariance_consistency_stability/IDEA.md)、[预注册](archive/research_records/2026-09-04_flip_equivariance_consistency_stability/PREREG.md)、[修订](archive/research_records/2026-09-04_flip_equivariance_consistency_stability/AMENDMENT_01.md)与[实现凭证](archive/research_records/2026-09-04_flip_equivariance_consistency_stability/IMPLEMENTATION_RECEIPT.json)。",
        "handoff active line",
    )
    handoff = once(
        handoff,
        "- 当前待续任务只有镜像前后答案一致训练：预注册已冻结但尚未实现/启动；任何旧分支仍不能误当成“继续跑完即可”。",
        "- 当前待续任务只有镜像前后答案一致训练：实现/smoke已通过，formal尚未启动；任何旧分支仍不能误当成“继续跑完即可”。",
        "handoff active-state",
    )
    handoff = once(
        handoff,
        "6. 当前唯一活动方法是已冻结的镜像前后答案一致训练；先完成隔离实现、合成测试、train-only GPU smoke，再决定是否启动四模型。不要在任何已揭示inner/test上选择loss、weight、view、epoch或teacher接线。",
        "6. 当前唯一活动方法是镜像前后答案一致训练；实现/smoke已过，下一步按固定runner启动C pair→D pair，四checkpoint全部完成后才打开inner。不要在任何已揭示inner/test上选择loss、weight、view、epoch或teacher接线。",
        "handoff step6",
    )
    atomic(catalog_path, catalog)
    atomic(handoff_path, handoff)
    print("recorded_flip_consistency_formal_ready")


if __name__ == "__main__":
    main()
