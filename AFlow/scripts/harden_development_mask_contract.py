"""Pre-access mechanical wiring of independently tested shape/path contracts."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REC=ROOT/'archive/research_records/2026-09-10_resumed_eight_hour'


def main():
    for study in ('small_rotation','image_blur'):assert not (ROOT/f'archive/research_records/2026-09-10_{study}/DEV_MASK_PREACCESS.json').exists()
    changes={}
    path=ROOT/'scripts/verify_resumed_development_masks.py';old=path.read_text()
    new=old.replace('from sammed3d.packed_prediction_masks import read_packed_case','from sammed3d.packed_prediction_masks import read_packed_case\nfrom sammed3d.development_mask_contract import validate_index,validate_case')
    needle="    for info in index['files'].values():p.checked(info['path'],info['sha256'])"
    assert old.count(needle)==1;new=new.replace(needle,"    validate_index(index,pre['case_ids'],pre['keys'],rec/'dev_prediction_masks')\n"+needle)
    needle="            assert np.array_equal(np.isin(cl,[1,2,3]),truth)"
    assert old.count(needle)==1;new=new.replace(needle,"            validate_case(row,pred,truth)\n"+needle);path.write_text(new)
    changes[str(path.relative_to(ROOT))]={'before':hashlib.sha256(old.encode()).hexdigest(),'after':hashlib.sha256(new.encode()).hexdigest()}
    path=ROOT/'scripts/evaluate_resumed_with_masks.py';old=path.read_text();needle='    p.write_new(prepath,'
    assert old.count(needle)==1;new=old.replace(needle,"    files+=['sammed3d/development_mask_contract.py','tests/test_development_mask_contract.py']\n"+needle);path.write_text(new)
    changes[str(path.relative_to(ROOT))]={'before':hashlib.sha256(old.encode()).hexdigest(),'after':hashlib.sha256(new.encode()).hexdigest()}
    with (REC/'DEV_MASK_CONTRACT_BINDING.json').open('x') as f:json.dump({'changes':changes,'formal_candidate_development_not_started':True,'no_prediction_or_gate_change':True},f,indent=2)


if __name__=='__main__':main()
