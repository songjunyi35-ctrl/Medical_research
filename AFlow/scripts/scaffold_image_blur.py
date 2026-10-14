"""Mechanical NEW-file derivation from the frozen rotation framework."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REC=ROOT/'archive/research_records/2026-09-10_image_blur'


def main():
    names=['scripts/small_rotation_protocol.py','scripts/train_small_rotation.py','scripts/smoke_small_rotation.py','scripts/freeze_small_rotation.py','scripts/evaluate_small_rotation.py','scripts/finalize_small_rotation.py','scripts/audit_small_rotation_training.py','tests/test_small_rotation_evaluation.py']
    receipt={}
    for name in names:
        src=ROOT/name;dst=ROOT/name.replace('small_rotation','image_blur')
        if dst.exists():raise FileExistsError(dst)
        text=src.read_text().replace('small_rotation','image_blur').replace('SmallRotation','ImageBlur').replace('ROTATION','BLUR').replace('rotation','blur')
        text=text.replace("ARMS=('native_blur','subject_blur')","ARMS=('subject_blur',)")
        text=text.replace('2026091051','2026091061').replace('2026091052','2026091062')
        text=text.replace('max_observed_degrees','max_observed_sigma').replace('<=15.000001','<=1.000001')
        text=text.replace("e['max_degrees']==15.","e['sigma_range']==[.5,1.]")
        text=text.replace('seed+200000','seed+300000').replace('(torch.rand(8,3,generator=generator)*2-1)*15.','torch.rand(8,generator=generator)*.5+.5')
        text=text.replace("'all_four_complete'","'both_complete'").replace('forwards==720','forwards==360').replace("raw['new_forwards']==720","raw['new_forwards']==360")
        text=text.replace("len(replay['results'])==4","len(replay['results'])==2").replace('len(results)==4','len(results)==2')
        text=text.replace("        assert blur[f'{seed}_native_blur']['independent_random_stream_sha256']==blur[f'{seed}_subject_blur']['independent_random_stream_sha256']\n",'')
        text=text.replace('四训练','两训练').replace('开发720 forwards','开发360 forwards').replace('四候选','两候选').replace('four fixed168','two fixed168')
        text=text.replace('小角度输入网格旋转双抽样层','图像模糊患者抽样层').replace('p0.2/每轴±15度的整体旋转','p0.2/sigma0.5—1输入体素的仅图像模糊')
        text=text.replace("['scripts/run_image_blur_training.py'","['scripts/run_image_blur_training.py'")
        if name.endswith('protocol.py'):
            text=text.replace('    source.check_freeze()\n    frozen=',"    source.check_freeze()\n    parent=ROOT/'archive/research_records/2026-09-10_small_rotation'\n    failed=(parent/'TRAINING_FAILURE.json').exists()\n    closed=(parent/'RESULT.json').exists() and json.loads((parent/'RESULT.json').read_text())['selected_arm'] is None\n    assert failed or closed, 'rotation retains GPU priority'\n    frozen=")
        if name.endswith('evaluate_small_rotation.py'):
            needle="            'export_exact':candidate['export_max_abs_error']==0}"
            text=text.replace(needle,"            'same_first_batch_labels':all(x['first_batch']['labels_sha256']==y['first_batch']['labels_sha256'] for x,y in zip(a,b,strict=True)),\n"+needle)
        if name.endswith('smoke_small_rotation.py'):
            text=text.replace('            if blur:x,y=deform(x,y)',"            original_labels=y.clone()\n            if blur:x,y=deform(x,y)\n            assert torch.equal(y,original_labels)\n            del original_labels")
        if name.startswith('tests/'):
            # Single recipe selection contracts replace the old two-arm tie tests.
            text=text[:text.index('def fixture():')]+'''def test_single_recipe_and_failed_seed_not_dropped():
    r={f'{s}_subject_blur':{'subject_dice':.75} for s in p.SEEDS}
    c={k:{'versus_native':delta()} for k in r}
    g={k:{'pass':True} for k in r}
    assert select_arm(r,c,g)[1]=='subject_blur'
    g['2026090711_subject_blur']['pass']=False
    assert select_arm(r,c,g)[1] is None


def test_average_gain_required():
    r={f'{s}_subject_blur':{'subject_dice':.75} for s in p.SEEDS}
    c={k:{'versus_native':delta(scan=.0029)} for k in r}
    g={k:{'pass':True} for k in r}
    assert select_arm(r,c,g)[1] is None
'''
        with dst.open('x') as f:f.write(text)
        receipt[name]={'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'destination':str(dst.relative_to(ROOT)),'destination_sha256':hashlib.sha256(dst.read_bytes()).hexdigest()}
    with (REC/'SCAFFOLD_RECEIPT.json').open('x') as f:json.dump(receipt,f,indent=2)
    with (REC/'PREREG.md').open('x') as f:f.write('# 预注册\n\nIDEA.md完整七问构成固定协议；两subject_blur种子，唯一训练图像模糊因素，标签原样、既定native与自身对照联合门。统计随机种子2026091061/62，360开发前向。旋转优先，失败才条件执行；既定截止21:24:55 UTC。评估与独立核验在正式训练前冻结。\n')
    with (REC/'SESSION.md').open('x') as f:f.write('# 条件备用\n\n新八小时窗口2026-09-10 13:39:55—21:39:55 UTC，计算截止21:24:55。训练准入最晚19:00 UTC，并需实测训练耗时估计足够。旋转没有晋级资格才执行，不重开旋转失败配方。\n')


if __name__=='__main__':main()
