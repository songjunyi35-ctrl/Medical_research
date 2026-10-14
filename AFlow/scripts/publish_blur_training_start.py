"""Artifact-driven start status for the predeclared conditional blur experiment."""
from datetime import datetime,timezone
import json
from scripts.structured_background_protocol import ROOT,sha,write_new
from scripts.publish_research_card import publish


def main():
    rec=ROOT/'archive/research_records/2026-09-10_image_blur'
    assert (rec/'TRAINING_STARTED.json').exists() and not (rec/'RESULT.json').exists()
    admission=json.loads((rec/'ENGINEERING_ADMISSION.json').read_text());assert admission['status']=='passed'
    card=rec/'CARD.md';old=card.read_text()
    new=old.replace('◻️ 条件备用，未训练','🟡 已准入，双种子训练中').replace('**结果：** 尚未训练。','**结果：** 两固定种子正式168轮训练已启动，暂无其开发结果。')
    new+='\n**正式启动：** 旋转完整开发与实际mask核验后无入选层；模糊fit8 GPU smoke/代码及数据冻结通过，双卡运行seed2026090711/12。其训练标签、患者抽样、基础随机流、损失与1×推理不变。预计约两小时，计算截止21:24:55 UTC。[训练前冻结](archive/research_records/2026-09-10_image_blur/TRAIN_PREACCESS.json)、[工程准入](archive/research_records/2026-09-10_image_blur/ENGINEERING_ADMISSION.json)。\n'
    card.write_text(new);publish(str(rec.relative_to(ROOT)))
    write_new(rec/'TRAINING_START_PUBLICATION.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'training_started_sha256':sha(rec/'TRAINING_STARTED.json'),'admission_sha256':sha(rec/'ENGINEERING_ADMISSION.json'),'card_sha256':sha(card),'development_access':False})


if __name__=='__main__':main()
