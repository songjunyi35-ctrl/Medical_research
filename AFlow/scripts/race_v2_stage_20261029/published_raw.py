"""Bind worker mask-audit receipts to the exact published integer records."""
import json
from pathlib import Path

from .governance import sha


def read(path):
    return json.loads(Path(path).read_text())


def verified(path, selection_sha256=None):
    path = Path(path)
    if not path.exists():
        return None
    raw = read(path)
    required = ('per_case', 'candidate_id', 'selection_sha256',
                'masks_audit_receipt_path', 'masks_audit_receipt_sha256')
    if any(field not in raw for field in required):
        return None
    receipt_path = Path(raw['masks_audit_receipt_path'])
    if not receipt_path.exists():
        return None
    assert sha(receipt_path) == raw['masks_audit_receipt_sha256']
    receipt = read(receipt_path)
    assert receipt['status'] == 'passed' and receipt['scans'] == 270
    assert receipt['all_masks_counts_points_exact'] is True
    assert receipt['selection_sha256'] == raw['selection_sha256']
    if selection_sha256 is not None:
        assert raw['selection_sha256'] == selection_sha256
    # Legacy B wrappers predate the explicit canonical path fields. Their
    # immutable canonical report lives in this same candidate directory.
    canonical = Path(raw.get('canonical_raw_path',
        path.parent / 'candidates' / raw['candidate_id'] / 'RAW_TEST270.json'))
    if not canonical.exists():
        return None
    digest = sha(canonical)
    assert receipt['raw_sha256'] == digest
    if 'canonical_raw_sha256' in raw:
        assert raw['canonical_raw_sha256'] == digest
    if 'raw_path' in receipt:
        assert Path(receipt['raw_path']).resolve() == canonical.resolve()
    original = read(canonical)
    assert original['selection_sha256'] == raw['selection_sha256']
    assert len(raw['per_case']) == 270 and raw['per_case'] == original['per_case']
    audits = receipt.get('per_case')
    if audits is None:
        configurations = receipt['per_configuration']
        assert len(configurations) == 1
        audits = next(iter(configurations.values()))['per_case']
    assert len(audits) == 270
    for audit, row in zip(audits, raw['per_case']):
        for key in ('case_id', 'tp', 'fp', 'fn'):
            assert audit[key] == row[key]
        if 'prediction_sha256' in audit:
            assert audit['prediction_sha256'] == row['prediction_sha256']
    return raw
