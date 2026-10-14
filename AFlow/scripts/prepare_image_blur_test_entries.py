"""New seed11 image-blur test entries, prepared before candidate development."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REC=ROOT/'archive/research_records/2026-09-10_image_blur'


def main():
    receipt={}
    for name in ['scripts/freeze_fractional_occupancy_test270.py','scripts/evaluate_fractional_occupancy_test270.py']:
        src=ROOT/name;dst=ROOT/name.replace('fractional_occupancy','image_blur')
        if dst.exists():raise FileExistsError(dst)
        text=src.read_text().replace('fractional_occupancy','image_blur').replace('weak_elastic_test270_helpers','resumed_test270_helpers').replace('weak_elastic_full864','small_rotation_full864')
        text=text.replace('from scripts.evaluate_image_blur import pairing','from scripts.evaluate_image_blur import training_pairing as pairing')
        text=text.replace("result['all_gates_passed'] and result['fixed_test_seed_if_eligible']==2026090711","result['selected_arm']=='subject_blur' and result['arm_gates']['subject_blur']['eligible']")
        text=text.replace("p.source.WORK/'seed2026090711_control/TRAIN_DONE.json'","p.control_path(2026090711,'subject_blur')")
        text=text.replace("p.WORK/'seed2026090711_image_blur/TRAIN_DONE.json'","p.WORK/'seed2026090711_subject_blur/TRAIN_DONE.json'")
        text=text.replace('datetime(2026,9,10,11,28,19,','datetime(2026,9,10,21,24,55,')
        text=text.replace('verify_saved_test270_panel','verify_resumed_test270_panel').replace('test_saved_test270_panel_verification','test_resumed_test270_panel_verification')
        text=text.replace('archive_verified_test270_panel','archive_resumed_test270_panel').replace('test_test270_panel_archival_rendering','test_resumed_test270_panel_archival_rendering')
        text=text.replace('both soft-target development seeds','both image-blur development seeds').replace('old-hard versus occupancy target','patient-sampling control versus image-blur candidate')
        if name.startswith('scripts/freeze'):
            text=text.replace('    p.write_new(output,',"    files+=['scripts/weak_elastic_test270_helpers.py','scripts/resumed_test270_helpers.py']\n    p.write_new(output,",1)
        with dst.open('x') as f:f.write(text)
        receipt[name]={'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'destination':str(dst.relative_to(ROOT)),'destination_sha256':hashlib.sha256(dst.read_bytes()).hexdigest()}
    with (REC/'TEST_ENTRY_DERIVATION.json').open('x') as f:json.dump(receipt,f,indent=2)


if __name__=='__main__':main()
