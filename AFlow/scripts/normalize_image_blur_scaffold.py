"""Pre-freeze mechanical vocabulary/contracts cleanup of newly derived entries."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REC=ROOT/'archive/research_records/2026-09-10_image_blur'


def main():
    assert not (REC/'TRAIN_PREACCESS.json').exists() and not (REC/'SMOKE_RECEIPT.json').exists()
    names=['scripts/evaluate_image_blur.py','scripts/finalize_image_blur.py','scripts/audit_image_blur_training.py','scripts/freeze_image_blur.py']
    changes={}
    for name in names:
        path=ROOT/name;old=path.read_text()
        new=old.replace('all_four','both').replace('all four','both').replace('four weak-blur candidates','two image-blur candidates')
        new=new.replace('eligible_for_preregistered_full864_refit','eligible_for_preregistered_fixed_seed11_test_panel').replace('conditional_full864_refit_requires_own_admission','fixed_seed11_test_requires_own_admission')
        new=new.replace('close_both_fixed_image_blur_recipes_no_strength_seed_epoch_search','close_fixed_image_blur_recipe_no_sigma_probability_seed_epoch_search')
        new=new.replace("assert e['max_observed_sigma']<=1.000001 and e['max_observed_sigma']<=1.000001","assert .5<=e['min_observed_sigma']<=e['max_observed_sigma']<=1.000001 and e['kernel_size']==7 and e['padding']=='reflect'")
        path.write_text(new)
        changes[name]={'before_sha256':hashlib.sha256(old.encode()).hexdigest(),'after_sha256':hashlib.sha256(new.encode()).hexdigest()}
    with (REC/'PRE_FREEZE_SCAFFOLD_CLEANUP.json').open('x') as f:json.dump(changes,f,indent=2)


if __name__=='__main__':main()
