"""One pre-freeze mechanical addition of the independent full864 audit gate."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REC=ROOT/'archive/research_records/2026-09-10_small_rotation_full864'


def main():
    assert not (REC/'TEST_PREACCESS.json').exists()
    path=ROOT/'scripts/freeze_small_rotation_test270.py';old=path.read_text()
    needle="    p.check_freeze()\n"
    assert old.count(needle)==1
    new=old.replace(needle,needle+"    audit=json.loads((p.RECORD/'TRAINING_REPLAY_FINAL.json').read_text())\n    assert audit['status']=='passed_full864_fixed168' and audit['same_initial_and_all_base_rng']\n    p.checked(p.ROOT/'scripts/audit_small_rotation_full864_training.py',audit['verifier_sha256'])\n    for r in audit['reports'].values():\n        p.checked(r['report_path'],r['report_sha256'])\n        p.checked(r['checkpoint']['path'],r['checkpoint']['sha256'])\n")
    new=new.replace("    p.write_new(output,", "    files+=['scripts/audit_small_rotation_full864_training.py','tests/test_small_rotation_full864_replay.py','archive/research_records/2026-09-10_small_rotation_full864/TRAINING_REPLAY_FINAL.json']\n    p.write_new(output,",1)
    path.write_text(new)
    with (REC/'TEST_REPLAY_BINDING.json').open('x') as f:json.dump({'before_sha256':hashlib.sha256(old.encode()).hexdigest(),'after_sha256':hashlib.sha256(new.encode()).hexdigest(),'new_test_access':False},f,indent=2)


if __name__=='__main__':main()
