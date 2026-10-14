"""Auditable mechanical derivation into NEW files; historical files untouched."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REC=ROOT/'archive/research_records/2026-09-10_small_rotation'


def main():
    names=['scripts/weak_elastic_protocol.py','scripts/train_weak_elastic.py',
           'scripts/smoke_weak_elastic.py','scripts/freeze_weak_elastic.py',
           'scripts/run_weak_elastic_training.py','scripts/evaluate_weak_elastic.py',
           'scripts/finalize_weak_elastic.py','scripts/audit_weak_elastic_training.py',
           'tests/test_weak_elastic_evaluation.py']
    receipt={}
    for name in names:
        src=ROOT/name;dst=ROOT/name.replace('weak_elastic','small_rotation')
        if dst.exists():raise FileExistsError(dst)
        text=src.read_text().replace('weak_elastic','small_rotation').replace('WeakElastic','SmallRotation').replace('ELASTIC','ROTATION').replace('elastic','rotation')
        text=text.replace('max_lipschitz_bound','max_observed_degrees').replace('max_axis_displacement_voxels','max_observed_degrees').replace('max_displacement_voxels','max_degrees')
        text=text.replace('<=.250001','<=15.000001').replace('<=2.000001','<=15.000001').replace("e['max_degrees']==2.","e['max_degrees']==15.")
        text=text.replace('2026091011','2026091051').replace('2026091012','2026091052')
        text=text.replace('seed+100000','seed+200000').replace('torch.rand(8,3,7,7,7,generator=generator)*2-1','(torch.rand(8,3,generator=generator)*2-1)*15.')
        text=text.replace('2026-09-10T11:28:19+00:00','2026-09-10T21:24:55+00:00').replace('datetime(2026,9,10,11,28,19,','datetime(2026,9,10,21,24,55,')
        text=text.replace('弱弹性形变','小角度输入网格旋转').replace('p0.2/最大2输入体素的平滑形变','p0.2/每轴±15度的整体旋转').replace('无形变','无追加旋转')
        with dst.open('x') as f:f.write(text)
        receipt[name]={'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'destination':str(dst.relative_to(ROOT)),'destination_sha256':hashlib.sha256(dst.read_bytes()).hexdigest()}
    with (REC/'SCAFFOLD_RECEIPT.json').open('x') as f:json.dump(receipt,f,indent=2)
    with (REC/'PREREG.md').open('x') as f:f.write('# 固定预注册\n\nIDEA.md七问构成完整预注册；训练前绑定其SHA以及所有训练实现、数据、对照和GPU smoke。开发前绑定评估/独立核验实现。四候选全部完成后统一开发，既定统计随机种子2026091051/52；未授权失败方案测试。\n')
    with (REC/'SESSION.md').open('x') as f:f.write('# 续研窗口\n\n2026-09-10 13:39:55至21:39:55 UTC，正式计算须21:24:55前结束。用户新授权；不沿用旧窗口时间。历史候选脚本只读，派生入口的精确机械变更见SCAFFOLD_RECEIPT及scaffold脚本。\n')


if __name__=='__main__':main()
