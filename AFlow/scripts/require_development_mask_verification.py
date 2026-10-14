"""A shared engineering guard; it does not modify any scientific promotion gate."""
import json
from pathlib import Path
from scripts.structured_background_protocol import ROOT,checked


def require_masks(record):
    record=Path(record);proof=json.loads((record/'DEV_MASK_VERIFICATION.json').read_text())
    assert proof['status']=='passed_actual_masks' and proof['independent_counts_and_point_hit']
    checked(record/'RESULT.json',proof['result_sha256']);checked(record/'RAW_DEVELOPMENT.json',proof['raw_sha256'])
    checked(record/'DEV_MASK_INDEX.json',proof['mask_index_sha256']);checked(record/'DEV_MASK_PREACCESS.json',proof['observer_preaccess_sha256'])
    checked(ROOT/'scripts/verify_resumed_development_masks.py',proof['verifier_sha256'])
    index=json.loads((record/'DEV_MASK_INDEX.json').read_text())
    assert proof['actual_masks_checked']==index['cases']*len(index['files'])
    for entry in index['files'].values():checked(entry['path'],entry['sha256'])
    return proof
