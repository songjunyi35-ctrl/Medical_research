import ast,json
from pathlib import Path
from scripts import motum_hgg29_protocol as p
from scripts.evaluate_motum_hgg29_panel import interp
ROOT=Path(__file__).resolve().parents[1]
def test_frozen_source_contract():
 p.docs();cohort=json.load(open(p.COHORT));audit=json.load(open(p.AUDIT));assert cohort['case_count']==cohort['subject_count']==p.CASES==29;assert len({x['case_id'] for x in cohort['cases']})==29;assert audit['official_snapshot']['dual_transfer_byte_exact'] is True;assert audit['hgg29_arrays']['native_geometry_exact'] is True;assert audit['participants']['metastases_excluded_before_model']==38;assert audit['model_or_cache']['model_forwards']==0
def test_view_and_bootstrap_contract():
 assert len(p.AXES)==len(set(p.AXES))==8 and p.AXES[0]==();assert p.BOOTSTRAP_SAMPLES==10000 and p.BOOTSTRAP_SEED==20261481
def test_entrypoints_parse_and_do_not_reference_old_raw():
 forbidden=('RAW_WAW152.json','RAW_CPTAC33.json','RAW_EXTERNAL95.json','RAW_TEST270.json')
 for name in ('build_motum_hgg29_cache.py','smoke_motum_hgg29_panel.py','evaluate_motum_hgg29_panel.py'):
  text=(ROOT/'scripts'/name).read_text();ast.parse(text);assert not any(x in text for x in forbidden)
def test_interpretation_boundaries():
 def c(d,lo,hi):return {'x':{'scan_dice':d},'paired_scan_dice_95ci':[lo,hi]}
 assert interp(c(.1,.01,.2),'x')=='supported_on_MOTUM';assert interp(c(.1,-.01,.2),'x')=='positive_but_uncertain';assert interp(c(0,-.1,.1),'x')=='not_supported_or_reversed';assert interp(c(-.1,-.2,-.01),'x')=='not_supported_or_reversed'
def test_cache_when_present():
 if not p.CACHE_RECEIPT.exists():return
 r=json.load(open(p.CACHE_RECEIPT));i=json.load(open(p.CACHE/'index.json'));assert r['all_gates_passed'] is True and i['cases']==i['subjects']==29;assert all(i['gates'].values());p.checked(p.CACHE/'index.json',r['index_sha256']);p.checked(p.CACHE/'images.f32',r['images_sha256']);p.checked(p.CACHE/'labels.u8',r['labels_sha256'])
def test_raw_when_present():
 path=p.RECORD/'RAW_MOTUM_HGG29.json'
 if not path.exists():return
 d=json.load(open(path));assert d['forwards']==261 and d['result_must_not_trigger_tuning_or_repeat'] is True;assert d['access']['motum_hgg29'] is True;assert all(v is False for k,v in d['access'].items() if k!='motum_hgg29')
