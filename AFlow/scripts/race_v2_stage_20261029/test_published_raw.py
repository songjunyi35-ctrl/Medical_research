"""Metadata regression: reject an audit receipt attached to different scores."""
import json
from pathlib import Path
import tempfile
import unittest

from .governance import sha
from .published_raw import verified


class PublishedRawBinding(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='aflow_receipt_binding_')
        self.addCleanup(self.directory.cleanup)
        self.folder = Path(self.directory.name)
        self.canonical = self.folder / 'candidates/single/RAW_TEST270.json'
        self.canonical.parent.mkdir(parents=True)
        self.rows = [dict(case_id=f'synthetic-{i}', tp=1, fp=0, fn=1) for i in range(270)]
        self.canonical.write_text(json.dumps(dict(selection_sha256='frozen-selection', per_case=self.rows)))
        self.audit = self.folder / 'AUDIT.json'
        self.receipt = dict(status='passed', scans=270, all_masks_counts_points_exact=True,
            selection_sha256='frozen-selection', raw_sha256=sha(self.canonical), per_case=self.rows)
        self.raw = dict(candidate_id='single', selection_sha256='frozen-selection', per_case=self.rows,
            masks_audit_receipt_path=str(self.audit))
        self.published = self.folder / 'RAW_SINGLE.json'

    def save(self):
        self.audit.write_text(json.dumps(self.receipt))
        self.raw['masks_audit_receipt_sha256'] = sha(self.audit)
        self.published.write_text(json.dumps(self.raw))

    def test_valid_legacy_path_binding(self):
        self.save()
        self.assertEqual(verified(self.published, 'frozen-selection')['per_case'], self.rows)

    def test_same_selection_wrong_receipt_raw_digest_rejected(self):
        self.receipt['raw_sha256'] = 'different-canonical'
        self.save()
        with self.assertRaises(AssertionError):
            verified(self.published, 'frozen-selection')

    def test_same_selection_mutated_wrapper_scores_rejected(self):
        self.raw['per_case'] = [dict(row) for row in self.rows]
        self.raw['per_case'][0]['tp'] = 7
        self.save()
        with self.assertRaises(AssertionError):
            verified(self.published, 'frozen-selection')

    def test_new_a_explicit_canonical_and_configuration_receipt(self):
        self.receipt.pop('per_case')
        self.receipt['per_configuration'] = {'single': dict(scans=270, per_case=self.rows)}
        self.receipt['raw_path'] = str(self.canonical)
        self.raw.update(canonical_raw_path=str(self.canonical), canonical_raw_sha256=sha(self.canonical))
        self.save()
        self.assertIsNotNone(verified(self.published, 'frozen-selection'))


if __name__ == '__main__':
    unittest.main()
