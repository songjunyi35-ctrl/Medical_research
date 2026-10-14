"""Frozen contract for the UCSD BraTS-GLI 2024 public test-package panel (56 scans / 41 subjects)."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / 'archive/research_records/2026-09-06_ucsd_brats2024test_frozen_panel'
REVIEW = ROOT / 'archive/reviews/2026-09-06_ucsd_brats2024test_source_audit'
SOURCE = ROOT / 'workspace/BraTS/ucsd_brats2024test_source_20260906'
COPY_A = SOURCE / 'copy_a'
COPY_B = Path('/tmp/aflow-ucsd-brats2024test/copy_b')
CACHE = Path('/tmp/aflow-ucsd-brats2024test/cache_128')
COHORT = REVIEW / 'FROZEN_COHORT.json'
AUDIT_V2 = REVIEW / 'SOURCE_AUDIT_V2.json'
INVENTORY = REVIEW / 'PACKAGE1221_INVENTORY.json'
SOURCE_RECEIPT = RECORD / 'SOURCE_RECEIPT.json'
CACHE_RECEIPT = RECORD / 'CACHE_RECEIPT.json'
FIRST_PROBE = RECORD / 'FIRST_CASE_PROBE.json'
MODEL_PREACCESS = RECORD / 'MODEL_PREACCESS.json'

IDEA_SHA = '73709613d0a0c04313165ad895f1d8e12df1863e31e31abab4a1954fdf03d693'
PREREG_SHA = 'acefe70505d5a233038a0180b4fcbdbbc40e13157e9c0851e10d9a5121be84d8'
AUDIT_V2_SHA = '9fbc2467e84bbf536676f056ffd581c4807b259453a47cb9e49692c9b06fbbf4'
INVENTORY_SHA = 'ec78b96cd7b5f0eee5a45441ee9ae8b38177f9cc42b040e0dee7109ff272d4a8'
COHORT_SHA = 'aed4cfcb611e639d3421cbf95c0eb3afc2871a1eb46837c5e71c2c470d6b2dd0'

PACKAGE_ID = 1221
CASES = 56
SUBJECTS = 41
BOOTSTRAP_SAMPLES = 10000
SEED_ABS_FULL_IDENTITY = 20261601
SEED_ABS_FULL_FLIP8 = 20261611
SEED_ABS_GCSAM = 20261621
SEED_PAIRED_FLIP8 = 20261701
SEED_PAIRED_GCSAM = 20261801
AXES = ((), (-1,), (-2,), (-2, -1), (-3,), (-3, -1), (-3, -2), (-3, -2, -1))

FULL = ROOT / 'workspace/BraTS/sammed3d_t1ce_wt/voxel_kd_fulltrajectory_972_20260824/final_formal_full972_candidate/plain_fulltrajectory_full972_candidate_epoch168.pth'
GCSAM = ROOT / 'workspace/BraTS/sammed3d_t1ce_wt/gcsam_full972_stability_20260831/inner_formal_gcsam_candidate/post100_best.pth'
FULL_SHA = 'f7df2a1678564268e899d20411e55c0163df480498273721a7e5b1b71c80d17b'
GCSAM_SHA = 'acf2fce112370e3cdb58fe295642680d5db1db3ceb619defdc47824ef4016d0d'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def checked(path, digest):
    if sha(path) != digest:
        raise ValueError('SHA mismatch: ' + str(path))


def docs():
    checked(RECORD / 'IDEA.md', IDEA_SHA)
    checked(RECORD / 'PREREG.md', PREREG_SHA)
    checked(AUDIT_V2, AUDIT_V2_SHA)
    checked(INVENTORY, INVENTORY_SHA)
    checked(COHORT, COHORT_SHA)


def cases():
    docs()
    d = json.load(open(COHORT))
    rows = d['cases']
    if len(rows) != CASES or d['case_count'] != CASES or d['subject_count'] != SUBJECTS:
        raise ValueError('cohort mismatch')
    if len({r['case_id'] for r in rows}) != CASES or len({r['subject_id'] for r in rows}) != SUBJECTS:
        raise ValueError('cohort identity')
    return rows


def write_new(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write('\n')
