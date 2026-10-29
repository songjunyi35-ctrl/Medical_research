"""Freeze receipt binding and archived-source publication after CPU review."""
import datetime
import subprocess
import sys

from .governance import ROOT, RECORD, atomic_json, sha


def main():
    assert not (RECORD / 'FINAL_TEST_SELECTION.json').exists()
    path = RECORD / 'FINAL_PUBLICATION_INNOVATION_FREEZE_V2.json'
    assert not path.exists()
    test = subprocess.run([sys.executable, '-m', 'scripts.race_v2_stage_20261029.test_published_raw'],
                          cwd=ROOT, check=True, capture_output=True, text=True)
    names = ['final_report_innovation_v2.py', 'final_report_watch_innovation_v2.py',
             'published_raw.py', 'test_published_raw.py', 'publish_innovation_v2.py',
             'publish_innovation.py', 'publish_snapshot_v2.py', 'audit_validation.py', 'governance.py']
    directory = ROOT / 'scripts/race_v2_stage_20261029'
    atomic_json(path, dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_sha256={str((directory / name).relative_to(ROOT)): sha(directory / name) for name in names},
        prior_version_path=str(RECORD / 'FINAL_PUBLICATION_INNOVATION_FREEZE.json'),
        prior_version_sha256=sha(RECORD / 'FINAL_PUBLICATION_INNOVATION_FREEZE.json'),
        prior_sources_and_freezes_preserved=True,
        mask_audit_receipt_canonical_and_wrapper_records_bound=True,
        archived_python_sources_included_in_text_publication=True,
        synthetic_metadata_regressions=dict(returncode=test.returncode, stdout=test.stdout, stderr=test.stderr),
        no_arrays_or_clinical_pixels_accessed=True, no_training_change=True, new270_access=False))
    path.chmod(0o444)


if __name__ == '__main__':
    main()
