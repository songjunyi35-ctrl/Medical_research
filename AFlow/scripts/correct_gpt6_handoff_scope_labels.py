#!/usr/bin/env python3
"""Remove stale overall-best wording from the GPT-6 handoff/catalog."""
from pathlib import Path
import tempfile
ROOT=Path(__file__).resolve().parents[1]
def atomic(p:Path,s:str)->None:
    with tempfile.NamedTemporaryFile("w",encoding="utf-8",dir=p.parent,delete=False) as h: h.write(s); t=Path(h.name)
    t.replace(p)
def repl(s:str,a:str,b:str)->str:
    if s.count(a)!=1: raise RuntimeError(f"expected one stale scope label, found {s.count(a)}")
    return s.replace(a,b,1)
def main()->None:
    p=ROOT/"AFLOW_BRATS_HANDOFF_CURRENT.md"; s=p.read_text()
    s=repl(s,"它现在是最高Subject/Point点估计、Scan第2；仍是failed-gate exploratory reused test","它现在是1×范围最高Subject/Point点估计、1× Scan第2（总体含8×时rank3）；仍是failed-gate exploratory reused test")
    s=repl(s,"它只在Scan点估计上排第1；Subject/Point仍由Full972 KD168领先。","它只在1×范围Scan点估计上排第1；1× Subject/Point仍由Full972 KD168领先，总体三项已由固定Flip8刷新。")
    s=repl(s,"Full972完整轨迹版本仍是当前最高Subject/Point且综合最稳的内部单模型点估计。","Full972完整轨迹identity仍是当前1×最高Subject/Point；同checkpoint固定Flip8是8×总体点估计最高。")
    atomic(p,s)
    p=ROOT/"AFLOW_BRATS_METHOD_CATALOG.md"; s=p.read_text()
    s=repl(s,"| Logit ensemble | 强模型为主，混入25% EMA输出 | ✅ | 当前总体test最高`0.745054`，但推理2× |","| Logit ensemble | 强模型为主，混入25% EMA输出 | ✅ | 历史总体test曾达`0.745054`，后被Full972/Flip8超过；推理2× |")
    s=repl(s,"| Voxel-KD | 让单模型模仿成功ensemble的每个体素概率 | ✅ | 当前最高单模型test`0.742976`，推理1× |","| Voxel-KD | 让单模型模仿成功ensemble的每个体素概率 | ✅ | 旧版本单模`0.742976`；Full972 KD168成为现1×主干，固定Flip8刷新总体 |")
    s=repl(s,"| GCSAM Full972稳定优化 | 每步先轻推参数，再要求被推后也答得好 | ❌/探索性 | inner CI门失败；授权test Scan点估计第1但CI跨0、Point少10，非Full972 final |","| GCSAM Full972稳定优化 | 每步先轻推参数，再要求被推后也答得好 | ❌/探索性 | inner CI门失败；授权test仅1× Scan点估计第1，CI跨0、Point少10，非Full972 final |")
    atomic(p,s); print("corrected_gpt6_scope_labels")
if __name__=="__main__": main()
