"""Bind new engineering guard inputs in still-prospective blur admissions."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REC=ROOT/'archive/research_records/2026-09-10_image_blur'


def main():
    assert not (REC/'TRAIN_PREACCESS.json').exists() and not (REC/'TEST_PREACCESS.json').exists()
    changes={}
    path=ROOT/'scripts/freeze_image_blur.py';old=path.read_text();needle="    p.write_new(p.RECORD/'IMPLEMENTATION_FREEZE.json',"
    addition="    files += [p.ROOT/n for n in ['scripts/require_development_mask_verification.py','scripts/verify_resumed_development_masks.py','scripts/evaluate_resumed_with_masks.py','sammed3d/prediction_recording.py','tests/test_prediction_recording.py']]\n    parent=p.ROOT/'archive/research_records/2026-09-10_small_rotation'\n    if (parent/'RESULT.json').exists():\n        files += [parent/n for n in ['RESULT.json','VERIFICATION_RECEIPT.json','DEV_MASK_VERIFICATION.json','DEV_MASK_INDEX.json','DEV_MASK_PREACCESS.json']]\n    else:\n        assert (parent/'TRAINING_FAILURE.json').exists()\n        files += [parent/'TRAINING_FAILURE.json']\n"
    assert old.count(needle)==1;new=old.replace(needle,addition+needle);path.write_text(new)
    changes[str(path.relative_to(ROOT))]={'before':hashlib.sha256(old.encode()).hexdigest(),'after':hashlib.sha256(new.encode()).hexdigest()}
    path=ROOT/'scripts/freeze_image_blur_test270.py';old=path.read_text();needle='    p.write_new(output,'
    addition="    files += ['scripts/require_development_mask_verification.py','scripts/verify_resumed_development_masks.py','archive/research_records/2026-09-10_image_blur/DEV_MASK_VERIFICATION.json','archive/research_records/2026-09-10_image_blur/DEV_MASK_INDEX.json','archive/research_records/2026-09-10_image_blur/DEV_MASK_PREACCESS.json']\n"
    assert old.count(needle)==1;new=old.replace(needle,addition+needle);path.write_text(new)
    changes[str(path.relative_to(ROOT))]={'before':hashlib.sha256(old.encode()).hexdigest(),'after':hashlib.sha256(new.encode()).hexdigest()}
    with (REC/'MASK_GUARD_DEPENDENCIES.json').open('x') as f:json.dump({'changes':changes,'formal_training_and_development_not_started':True,'scientific_gates_unchanged':True},f,indent=2)


if __name__=='__main__':main()
