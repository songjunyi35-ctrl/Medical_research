"""Pre-admission mechanical normalization of prospective receipt time fields."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REC=ROOT/'archive/research_records/2026-09-10_resumed_eight_hour'


def main():
    for study in ('small_rotation_full864','image_blur'):
        record=ROOT/f'archive/research_records/2026-09-10_{study}'
        assert not (record/'TEST_PREACCESS.json').exists()
        if study=='small_rotation_full864':assert not (record/'TRAIN_PREACCESS.json').exists()
    changes={}
    for name in ['scripts/freeze_small_rotation_full864.py','scripts/freeze_small_rotation_test270.py','scripts/freeze_image_blur_test270.py']:
        path=ROOT/name;before=path.read_text()
        after=before.replace("'deadline_utc':'2026-09-10T21:24:55+00:00'","'deadline_utc':'2026-09-10T21:39:55+00:00','training_supervisor_deadline_utc':'2026-09-10T21:24:55+00:00'")
        after=after.replace("'created_utc':datetime.now(timezone.utc).isoformat(),","'created_utc':datetime.now(timezone.utc).isoformat(),'session_end_utc':'2026-09-10T21:39:55+00:00','compute_deadline_utc':'2026-09-10T21:24:55+00:00',")
        assert before!=after
        path.write_text(after)
        changes[name]={'before_sha256':hashlib.sha256(before.encode()).hexdigest(),'after_sha256':hashlib.sha256(after.encode()).hexdigest()}
    with (REC/'BUDGET_RECEIPT_NORMALIZATION.json').open('x') as f:json.dump({'changes':changes,'scientific_gates_unchanged':True,'all_affected_entries_not_admitted_yet':True},f,indent=2)


if __name__=='__main__':main()
