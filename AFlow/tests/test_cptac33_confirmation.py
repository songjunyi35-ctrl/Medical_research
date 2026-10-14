from __future__ import annotations
import json
from pathlib import Path
import pytest,torch
from scripts import cptac33_protocol as p
import scripts.download_cptac33_source as dl
import scripts.evaluate_brats_africa95_frozen_baselines as base
def test_cohort33_public_labeled_contract():
 rows=p.rows();assert len(rows)==len({x['brats2021_id'] for x in rows})==33;assert len({x['cptac_patient_id'] for x in rows})==33;assert dl.expected(rows) and sum(dl.expected(rows).values())==78013574;assert all(x['original_role']=='Training' for x in rows)
def test_official_sums_cover_selected66():
 r=json.load(open(p.SUMS_RECEIPT));assert r['status']=='passed' and r['selected_rows']==66 and not r['missing'] and not r['empty_file_checksums'];assert r['selected_checksum_length']==32
def test_hypothesis_rule_frozen():
 r=json.load(open(p.RECORD/'PREREGISTRATION.json'));assert r['primary_interpretation']['strong_directional_support'].endswith('paired CI lower >0');assert r['performance_promotion_gate'] is None;assert r['result_policy']=='one run, always archive, no tuning or repeat'
def test_reflections_and_forward_budget():
 assert len(set(p.AXES))==8;v=torch.arange(24).reshape(1,1,2,3,4)
 for axes in p.AXES:assert torch.equal(torch.flip(torch.flip(v,axes),axes),v)
 source=Path('scripts/evaluate_cptac33_confirmation.py').read_text();assert 'forwards!=297' in source and "performance_promotion_gate':None" in source
