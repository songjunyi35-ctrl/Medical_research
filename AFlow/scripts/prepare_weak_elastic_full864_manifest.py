"""Metadata-only deterministic union; this does not admit training or test."""
from datetime import datetime,timezone
import hashlib
import json
from scripts import weak_elastic_protocol as p


def main():
    p.check_freeze();fit,dev=p.check_roles()
    rows=sorted(fit+dev,key=lambda r:r['case_id'])
    assert len(rows)==864 and len({r['case_id'] for r in rows})==864 and len({r['subject_id'] for r in rows})==391
    record=p.ROOT/'archive/research_records/2026-09-10_weak_elastic_full864'
    payload=''.join(json.dumps(row,sort_keys=True,separators=(',',':'))+'\n' for row in rows).encode()
    with (record/'FIT864.jsonl').open('xb') as f:f.write(payload)
    p.write_new(record/'MANIFEST_RECEIPT.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'manifest_sha256':hashlib.sha256(payload).hexdigest(),'source_fit_sha256':p.FIT_SHA,'source_development_sha256':p.DEV_SHA,'scans':864,'subjects':391,'pixels_accessed':False,'training_admitted':False,'source_roles_checked':True,'old_confirmation_included':False})
    print(hashlib.sha256(payload).hexdigest())


if __name__=='__main__':main()
