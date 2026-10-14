#!/usr/bin/env python3
"""Mark the paired consistency experiment as formally running."""
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]

def atomic(path: Path, text: str) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as h:
        h.write(text); tmp=Path(h.name)
    tmp.replace(path)

def prefix(text: str, start: str, replacement: str) -> str:
    lines=text.splitlines(keepends=True); hits=[i for i,x in enumerate(lines) if x.startswith(start)]
    if len(hits)!=1: raise RuntimeError(f"expected one prefix {start!r}, found {len(hits)}")
    lines[hits[0]]=replacement+("\n" if lines[hits[0]].endswith("\n") else "")
    return "".join(lines)

def once(text: str, old: str, new: str) -> str:
    if text.count(old)!=1: raise RuntimeError(f"expected one exact state, found {text.count(old)}")
    return text.replace(old,new,1)

def main() -> None:
    cpath=ROOT/"AFLOW_BRATS_METHOD_CATALOG.md"; c=cpath.read_text()
    c=prefix(c,"| 镜像前后答案一致训练 |","| 镜像前后答案一致训练 | 训练时让原图与镜像图翻回后的最终mask一致 | 🧪 formal C pair训练中 | 首轮配对全exact、121步/968 draws/2 forwards；峰值约42GB，inner未访问 |")
    c=once(c,"## 镜像前后答案一致训练 🧪 实现/smoke通过；formal准备启动","## 镜像前后答案一致训练 🧪 formal C pair训练中")
    c=once(c,"下一步启动formal C pair→D pair，四checkpoint全部完成才冻结inner。","formal C Control/Candidate已在双A100启动；epoch1 sampler/RNG/reflection/首batch/首existing与consistency exact，121步/968 draws/2 forwards，inner为空。C完成后runner自动启动D；四checkpoint全部完成才冻结inner。")
    hpath=ROOT/"AFLOW_BRATS_HANDOFF_CURRENT.md"; h=hpath.read_text()
    h=prefix(h,"- **新活动实验：镜像前后答案一致训练实现/smoke通过，formal准备启动。**","- **新活动实验：镜像前后答案一致训练formal C pair训练中。** C/D各864 fit/108 inner且彼此/对旧A/B-inner患者隔离；Control/Candidate每步同两次student forward，只改变一致性梯度0/0.1。AMENDMENT_01平方分母、29项测试和双卡smoke已过。C epoch1两臂sampler、增强RNG、reflection、首batch、首existing/consistency exact，每臂121步/968 draws/2 forwards；Candidate平均一致性`0.001063` vs Control`0.001478`，只作train-only机制健康信号。双卡约42GB，inner产物为空。runner将C完成后自动跑D，四模型fixed168并冻结前禁止inner；全程禁止fixed validation/outer/confirmation/test270/UCSD75。workdir=`workspace/BraTS/sammed3d_t1ce_wt/flip_equivariance_consistency_20260904`，证据见[预注册](archive/research_records/2026-09-04_flip_equivariance_consistency_stability/PREREG.md)、[修订](archive/research_records/2026-09-04_flip_equivariance_consistency_stability/AMENDMENT_01.md)、[实现凭证](archive/research_records/2026-09-04_flip_equivariance_consistency_stability/IMPLEMENTATION_RECEIPT.json)与[launch](archive/research_records/2026-09-04_flip_equivariance_consistency_stability/FORMAL_LAUNCH_RECEIPT.json)。")
    h=once(h,"- 当前待续任务只有镜像前后答案一致训练：实现/smoke已通过，formal尚未启动；任何旧分支仍不能误当成“继续跑完即可”。","- 当前待续任务只有镜像前后答案一致训练：formal C pair正在双A100运行，之后自动D pair；任何旧分支仍不能误当成“继续跑完即可”。")
    h=once(h,"6. 当前唯一活动方法是镜像前后答案一致训练；实现/smoke已过，下一步按固定runner启动C pair→D pair，四checkpoint全部完成后才打开inner。不要在任何已揭示inner/test上选择loss、weight、view、epoch或teacher接线。","6. 当前唯一活动方法是镜像前后答案一致训练；固定runner已启动C pair，之后自动D pair，四checkpoint全部完成后才打开inner。不要重启或重复；不要在任何已揭示inner/test上选择loss、weight、view、epoch或teacher接线。")
    atomic(cpath,c); atomic(hpath,h); print("marked_flip_consistency_running")

if __name__=="__main__": main()
