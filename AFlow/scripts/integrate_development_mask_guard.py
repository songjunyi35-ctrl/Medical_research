"""Pre-access mechanical integration of the declared archive-only engineering guard."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
from scripts.publish_research_card import publish

ROOT=Path(__file__).resolve().parents[1]
SESSION=ROOT/'archive/research_records/2026-09-10_resumed_eight_hour'


def main():
    for study in ('small_rotation','image_blur'):
        rec=ROOT/f'archive/research_records/2026-09-10_{study}'
        assert not (rec/'DEV_PREACCESS.json').exists() and not (rec/'TEST_PREACCESS.json').exists()
    assert not (ROOT/'archive/research_records/2026-09-10_small_rotation_full864/TRAIN_PREACCESS.json').exists()
    edits={
      'scripts/smoke_small_rotation_full864.py':("    return result\n","    from scripts.require_development_mask_verification import require_masks\n    require_masks(p.source.RECORD)\n    return result\n"),
      'scripts/freeze_image_blur_test270.py':("    assert all(all(g.values()) for g in result['gates'].values())\n","    assert all(all(g.values()) for g in result['gates'].values())\n    from scripts.require_development_mask_verification import require_masks\n    require_masks(p.RECORD)\n"),
      'scripts/publish_training_study_result.py':("    selected=r['selected_arm'];details=interpret(r)\n","    from scripts.require_development_mask_verification import require_masks\n    require_masks(rec)\n    selected=r['selected_arm'];details=interpret(r)\n"),
      'scripts/image_blur_protocol.py':("    assert failed or closed, 'rotation retains GPU priority'\n","    assert failed or closed, 'rotation retains GPU priority'\n    if closed:\n        from scripts.require_development_mask_verification import require_masks\n        require_masks(parent)\n"),
    }
    changes={}
    for name,(before,after) in edits.items():
        path=ROOT/name;old=path.read_text();assert old.count(before)==1,name
        new=old.replace(before,after);path.write_text(new)
        changes[name]={'before_sha256':hashlib.sha256(old.encode()).hexdigest(),'after_sha256':hashlib.sha256(new.encode()).hexdigest()}
    path=ROOT/'scripts/freeze_small_rotation_full864.py';old=path.read_text()
    needle="    p.write_new(p.RECORD/'IMPLEMENTATION_FREEZE.json',"
    addition="    files += [p.source.RECORD/n for n in ['DEV_MASK_VERIFICATION.json','DEV_MASK_INDEX.json','DEV_MASK_PREACCESS.json']]\n    files += [p.ROOT/n for n in ['scripts/require_development_mask_verification.py','scripts/verify_resumed_development_masks.py','sammed3d/prediction_recording.py']]\n"
    assert old.count(needle)==1;new=old.replace(needle,addition+needle);path.write_text(new)
    changes[str(path.relative_to(ROOT))]={'before_sha256':hashlib.sha256(old.encode()).hexdigest(),'after_sha256':hashlib.sha256(new.encode()).hexdigest()}
    runbook=SESSION/'EXECUTION_ORDER.md';old=runbook.read_text()
    new=old.replace('`python -m scripts.evaluate_small_rotation`','`python -m scripts.evaluate_resumed_with_masks --study small_rotation`')
    new=new.replace('3. `python -m scripts.finalize_small_rotation`','3. `python -m scripts.finalize_small_rotation`，随后`python -m scripts.verify_resumed_development_masks --study small_rotation`')
    new=new.replace('→ evaluate_image_blur → finalize_image_blur →','→ `evaluate_resumed_with_masks --study image_blur` → finalize_image_blur → `verify_resumed_development_masks --study image_blur` →')
    new+='\n新增工程观察器在正式开发访问前明确补充，原训练和数值评估代码仍hash-bound。两个原evaluate入口只由透明包装调用，不另行重复执行；额外保存180/90MiB实际mask且0额外forward。先原统计finalizer，再实际mask核验，再发布/晋级。\n'
    runbook.write_text(new);changes[str(runbook.relative_to(ROOT))]={'before_sha256':hashlib.sha256(old.encode()).hexdigest(),'after_sha256':hashlib.sha256(new.encode()).hexdigest()}
    blur=ROOT/'archive/research_records/2026-09-10_image_blur/CARD.md'
    text=blur.read_text()+'\n**执行前归档补充：** 开发用透明观察器保存相同掩膜（90MiB，0额外forward），不改预测/参数/门；实际掩膜独立核对是后续工程准入要求。[补充](archive/research_records/2026-09-10_image_blur/DEV_MASK_ADDENDUM.md)。\n';blur.write_text(text)
    with (SESSION/'DEV_MASK_INTEGRATION.json').open('x') as f:json.dump({'created_utc':datetime.now(timezone.utc).isoformat(),'changes':changes,'new_development_forwards':0,'scientific_prediction_and_gates_unchanged':True,'new_engineering_evidence_required':True},f,indent=2)
    publish('archive/research_records/2026-09-10_small_rotation');publish('archive/research_records/2026-09-10_image_blur')


if __name__=='__main__':main()
