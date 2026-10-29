"""Explicitly authorized deletion of two closed, SHA-audited cache payloads.

No weight, source image, report, current-worker cache, or Git object is removed.
The whitelist is exact; indexes and original numerical builders are preserved.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
import shutil

from .governance import ROOT, RECORD, atomic_json, sha

REVIEW = ROOT/'archive/reviews/2026-10-29_cache_cleanup'
OUT = RECORD/'cache_cleanup_20261029'
AUDIT = REVIEW/'b/PAYLOAD_SHA_AUDIT.json'
ALLOWED = {
    (ROOT/'archive/reviews/2026-09-16_remove_684_branch/closed_material/runtime/graymean_pair_20260916/cache').resolve(),
    (ROOT/'workspace/BraTS/nnunet_t1ce_wt/resenc_20261013/test_logits').resolve(),
}


def read(path):
    return json.loads(Path(path).read_text())


def main(apply):
    audit = read(AUDIT)
    assert audit['status'] == 'passed_exact_candidate_payloads'
    files = audit['files']
    assert len(files) == len({row['path'] for row in files}) == 1134
    assert {Path(row['path']).resolve().parent for row in files} == ALLOWED
    assert (REVIEW/'b/PASSED_RECEIPT.json').is_file()
    for path, digest in audit['evidence_files'].items():
        assert sha(path) == digest, path
    for row in files:
        path = Path(row['path'])
        assert path.resolve().is_relative_to(ROOT.resolve())
        assert not path.is_symlink() and path.suffix == '.npz' and path.is_file()
        stat = path.stat()
        assert (stat.st_dev, stat.st_ino, stat.st_nlink, stat.st_size) == (
            row['device'], row['inode'], 1, row['bytes'])
        assert sha(path) == row['sha256'] == row['expected_sha256'], str(path)
    for directory in ALLOWED:
        assert {str(path) for path in directory.glob('*.npz')} == {
            row['path'] for row in files if Path(row['path']).parent == directory}
    OUT.mkdir(parents=True, exist_ok=True)
    plan = dict(at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
                authorization='user explicitly: 没磁盘容量就清，之前缓存保留脚本就行',
                status='exact_whitelist_ready', files=len(files), logical_bytes=audit['total_logical_bytes'],
                allocated_bytes=audit['total_allocated_bytes'], payload_audit_sha256=sha(AUDIT),
                protected_weights_manifest_sha256=sha(REVIEW/'a/WEIGHTS_BEFORE.json'),
                protected_sources_manifest_sha256=sha(REVIEW/'a/SOURCE_METADATA_BEFORE.json'),
                evidence_sha256=audit['evidence_files'], groups=audit['groups'],
                active_io_audit='Known AFlow processes: no references; unrelated system/sshd permission exceptions preserved in receipt',
                exclusions='all weights/original sources/final reports and masks/current1154259 train/current1154121 108+270/Git',
                arrays_decoded=False, model_forwards=0, payload_transfer=False)
    if not apply:
        atomic_json(OUT/'PLAN.json', plan)
        print(json.dumps({key: plan[key] for key in ('status','files','logical_bytes','allocated_bytes')}))
        return
    assert not (OUT/'DELETE_STARTED.json').exists(), 'no implicit repeated deletion attempt'
    assert read(OUT/'PLAN.json')['payload_audit_sha256'] == sha(AUDIT)
    before = shutil.disk_usage(ROOT).free
    atomic_json(OUT/'DELETE_STARTED.json', dict(at=plan['at'], plan_sha256=sha(OUT/'PLAN.json'),
                                               free_bytes_before=before, user_authorized=True))
    removed = 0
    try:
        with (OUT/'DELETION_JOURNAL.jsonl').open('x') as journal:
            for row in files:
                path = Path(row['path'])
                stat = path.stat()
                assert stat.st_ino == row['inode'] and stat.st_nlink == 1 and stat.st_size == row['bytes']
                path.unlink()
                assert not path.exists()
                removed += 1
                journal.write(json.dumps(dict(path=row['path'], bytes=row['bytes'], sha256=row['sha256']))+'\n')
            journal.flush()
            os.fsync(journal.fileno())
    except BaseException as error:
        atomic_json(OUT/'DELETE_FAILURE.json', dict(error=repr(error), files_removed=removed,
                                                   no_other_paths_touched=True))
        raise
    assert removed == 1134 and all(not Path(row['path']).exists() for row in files)
    for path, digest in audit['evidence_files'].items():
        assert sha(path) == digest, path
    for group in audit['groups']:
        marker = dict(at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
                      status='payload_removed_by_user_authorization', files_removed=group['files'],
                      bytes_removed=group['logical_bytes'], original_index_and_builder_retained=True,
                      arrays_not_present=True, index_does_not_imply_payload_complete=True,
                      receipt_directory=str(OUT), no_model_or_scoring_run=True)
        path = Path(group['directory'])/'PAYLOAD_REMOVED_20261029.json'
        assert not path.exists()
        atomic_json(path, marker)
        atomic_json(OUT/(group['group']+'_REMOVED.json'), dict(marker, original_marker_path=str(path)))
    after = shutil.disk_usage(ROOT).free
    atomic_json(OUT/'DELETION_RESULT.json', dict(at=plan['at'], status='payloads_deleted_protection_audit_pending',
                files_removed=removed, logical_bytes_removed=audit['total_logical_bytes'],
                allocated_bytes_removed=audit['total_allocated_bytes'], free_bytes_before=before,
                free_bytes_after=after, measured_free_delta=after-before,
                concurrent_filesystem_writes_possible=True, retained_evidence_sha256=audit['evidence_files'],
                no_original_source_or_weight_deleted=True, no_neural_forwards=True, no_new_scores=True,
                no_clinical_payload_transfer=True))
    print(json.dumps(dict(status='deleted', files=removed, allocated_GiB=audit['total_allocated_bytes']/2**30,
                          free_GiB_after=after/2**30)))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    main(parser.parse_args().apply)
