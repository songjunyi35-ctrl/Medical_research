"""Publish a primary-source limitation without modifying any experimental result."""
from datetime import datetime,timezone
from scripts.structured_background_protocol import ROOT,sha,write_new
from scripts.publish_research_card import publish


def main():
    rec=ROOT/'archive/research_records/2026-09-10_resumed_eight_hour';source=rec/'PRIMARY_SOURCE_CHECK.md'
    synthesis=rec/'RESEARCH_SYNTHESIS.md';old=synthesis.read_text()
    with (rec/'RESEARCH_SYNTHESIS.md.BEFORE_PRIMARY_SOURCE_NOTE').open('x') as f:f.write(old)
    synthesis.write_text(old+'\n## 一手依据补充：不把已完成的WT直接监督再当新方法\n\n原始区域训练对照从类别softmax改为重叠区域sigmoid，而当前基线已直接优化WT；这不是追加辅助区域收益的直接证据。下一优先级是匹配对照设计，不是立即照搬预期涨幅。完整来源与适用边界见[一手核查](PRIMARY_SOURCE_CHECK.md)。\n')
    labels=ROOT/'archive/research_records/2026-09-10_fit_subregion_labels';card=labels/'CARD.md'
    card.write_text(card.read_text()+'\n**一手依据边界：** 原论文的区域对照并非当前WT-only基线追加辅助区域；本工程只准备数据，不证明新增区域目标会提高指标。下一步先做匹配对照设计。[原文适用边界](archive/research_records/2026-09-10_resumed_eight_hour/PRIMARY_SOURCE_CHECK.md)。\n')
    publish(str(labels.relative_to(ROOT)))
    write_new(rec/'PRIMARY_SOURCE_NOTE_PUBLICATION.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'note_sha256':sha(source),'synthesis_sha256':sha(synthesis),'data_engineering_card_sha256':sha(card),'experimental_metrics_and_gates_unchanged':True,'new_forwards':0})


if __name__=='__main__':main()
