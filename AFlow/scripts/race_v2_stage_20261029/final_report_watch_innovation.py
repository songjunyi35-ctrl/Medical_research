"""Wait for fully audited published wrappers, not a bare raw-report pathname."""
from __future__ import annotations

import json
import time

from .governance import RECORD, ROOT, atomic_json, guard, sha
from . import final_report_innovation as final_report


def ready(path):
    if not path.exists():
        return False
    value = json.loads(path.read_text())
    fields = ('per_case', 'candidate_id', 'selection_sha256', 'masks_audit_receipt_path',
              'masks_audit_receipt_sha256')
    if any(key not in value for key in fields):
        return False
    from pathlib import Path
    receipt = Path(value['masks_audit_receipt_path'])
    if not receipt.exists():
        return False
    assert sha(receipt) == value['masks_audit_receipt_sha256']
    audit = json.loads(receipt.read_text())
    assert audit['status'] == 'passed' and audit['scans'] == 270
    assert audit['selection_sha256'] == value['selection_sha256']
    assert audit['all_masks_counts_points_exact'] is True
    return True


def main():
    frozen = json.loads((RECORD/'FINAL_PUBLICATION_INNOVATION_FREEZE.json').read_text())
    for path, digest in frozen['source_sha256'].items():
        assert sha(ROOT/path) == digest, path
    while True:
        guard(1154121)
        if ((RECORD/'FINAL_TEST_SELECTION.json').exists()
                and ready(final_report.A/'final270/RAW_SINGLE.json')
                and ready(final_report.B/'final270/RAW_FUSION.json')):
            final_report.complete()
            return
        time.sleep(30)


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        atomic_json(RECORD/'FINAL_PUBLICATION_FAILURE.json', {'at_unix': time.time(), 'error': repr(error),
                    'saved_predictions_retained': True, 'no_automatic_neural_retry': True})
        raise
