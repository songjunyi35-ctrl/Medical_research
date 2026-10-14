"""Index late source checks and descriptive audits without touching model results."""
from datetime import datetime,timezone
from scripts.structured_background_protocol import ROOT,sha,write_new


def main():
    rec=ROOT/'archive/research_records/2026-09-10_resumed_eight_hour'
    files=[rec/'PRIMARY_SOURCE_CHECK.md',rec/'UNIQUE_COHORT_ACCOUNTING.json',rec/'TRIALS_CLOSED.json',rec/'POINT_RECORD_CORRECTION.json',ROOT/'archive/research_records/2026-09-10_small_rotation/SAMPLING_INTERACTION.json',ROOT/'archive/research_records/2026-09-10_small_rotation/SAMPLING_INTERACTION.md',ROOT/'archive/research_records/2026-09-10_image_blur/SCAN_TRADEOFFS.json',ROOT/'archive/research_records/2026-09-10_image_blur/SCAN_TRADEOFFS.md']
    write_new(rec/'SUPPLEMENTAL_EVIDENCE.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'files':{str(p.relative_to(ROOT)):sha(p) for p in files},'all_posthoc_items_descriptive_not_new_promotion_rules':True,'new_forwards':0})
    synthesis=rec/'RESEARCH_SYNTHESIS.md';old=synthesis.read_text()
    with (rec/'RESEARCH_SYNTHESIS.md.BEFORE_SUPPLEMENT_INDEX').open('x') as f:f.write(old)
    synthesis.write_text(old+'\n## 零额外前向的补充核查\n\n- [唯一队列计数](UNIQUE_COHORT_ACCOUNTING.json)：1800次前向仍是同180扫描/78患者，不是1800份独立验证。\n- [采样与旋转的事后交互](../2026-09-10_small_rotation/SAMPLING_INTERACTION.md)：Dice交互均值两seed为负，Point交互符号不一致；无新增检验/门。\n- [模糊逐扫描取舍](../2026-09-10_image_blur/SCAN_TRADEOFFS.md)：seed11有119例收缩，其中83例Dice下降；seed12扩张/收缩混合，不把总体体素趋势泛化为所有病例。\n- [补充证据SHA索引](SUPPLEMENTAL_EVIDENCE.json)。\n\n以上补充不改变任何原始分数、晋级门、未准入测试结论或当前模型参数。\n')
    print('supplemental evidence indexed',flush=True)


if __name__=='__main__':main()
