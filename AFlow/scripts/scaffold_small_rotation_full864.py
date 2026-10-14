"""New conditional full-data entries; closed historical branch stays byte-bound."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REC=ROOT/'archive/research_records/2026-09-10_small_rotation_full864'


def main():
    files=['scripts/weak_elastic_full864_protocol.py','scripts/train_weak_elastic_full864.py','scripts/smoke_weak_elastic_full864.py','scripts/freeze_weak_elastic_full864.py','scripts/run_weak_elastic_full864.py','scripts/prepare_weak_elastic_full864_manifest.py','tests/test_weak_elastic_full864.py']
    receipt={}
    for name in files:
        source=ROOT/name;target=ROOT/name.replace('weak_elastic','small_rotation')
        if target.exists():raise FileExistsError(target)
        text=source.read_text().replace('weak_elastic','small_rotation').replace('WeakElastic','SmallRotation').replace('ELASTIC','ROTATION').replace('elastic','rotation').replace('2026091031','2026091071')
        text=text.replace('datetime(2026,9,10,11,28,19,','datetime(2026,9,10,21,24,55,').replace('2026-09-10T11:28:19+00:00','2026-09-10T21:24:55+00:00')
        # Admission reserves 20 minutes to the actual session end, supervisor
        # independently reserves the final 15-minute archival period.
        if name.endswith('freeze_weak_elastic_full864.py'):text=text.replace('datetime(2026,9,10,21,24,55,','datetime(2026,9,10,21,39,55,')
        with target.open('x') as f:f.write(text)
        receipt[name]={'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'destination':str(target.relative_to(ROOT)),'destination_sha256':hashlib.sha256(target.read_bytes()).hexdigest()}
    with (REC/'SCAFFOLD_RECEIPT.json').open('x') as f:json.dump(receipt,f,indent=2)
    with (REC/'PREREG.md').open('x') as f:f.write('# 条件全864预注册\n\nIDEA.md七问构成协议。主开发全门通过、时间/工程准入前不训练；训练固定seed2026091071，两臂唯一旋转差异，原dev180一旦参与训练就不再作为验证。测试需另固定实现/输出/统计收据，所有结果入榜。\n')


if __name__=='__main__':main()
