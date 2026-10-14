import ast,json
from pathlib import Path
from scripts import waw_postop_only31_protocol as p
from scripts.evaluate_waw_postop_only31_panel import interp
ROOT=Path(__file__).resolve().parents[1]
def test_cohort_is_patient_disjoint_and_earliest():
 p.docs();c=json.load(open(p.COHORT));a=json.load(open(p.AUDIT));assert c['case_count']==c['subject_count']==31;assert len({x['subject_id'] for x in c['cases']})==31;assert a['frozen']['preop_patient_overlap']==0;assert min(x['days_postop'] for x in c['cases'])==0 and max(x['days_postop'] for x in c['cases'])==22;assert a['patient_difference']['later_numeric_scans_excluded_by_earliest_per_subject']==17;assert a['model_or_large_image_access']['model_forwards']==0
def test_source_receipt_and_target():
 r=json.load(open(p.SOURCE_RECEIPT));assert r['status']=='passed' and r['selected']['cases']==31 and r['preop_patient_overlap']==0;assert set(r['selected']['global_raw_union'])=={0,1,2,3,4};assert r['model_forwards']==0
def test_views_stats_and_interpretation():
 assert len(p.AXES)==len(set(p.AXES))==8 and p.AXES[0]==();assert p.BOOTSTRAP_SEED==20261531 and p.BOOTSTRAP_SAMPLES==10000
 def c(d,lo,hi):return {'x':{'scan_dice':d},'paired_scan_dice_95ci':[lo,hi]}
 assert interp(c(.1,.01,.2),'x')=='supported_on_WAW_postop31';assert interp(c(.1,-.01,.2),'x')=='positive_but_uncertain';assert interp(c(0,-.1,.1),'x')=='not_supported_or_reversed'
def test_entrypoints_isolated():
 bad=('RAW_WAW152.json','RAW_CPTAC33.json','RAW_EXTERNAL95.json','RAW_TEST270.json')
 for n in ('build_waw_postop_only31_cache.py','smoke_waw_postop_only31_panel.py','evaluate_waw_postop_only31_panel.py'):
  t=(ROOT/'scripts'/n).read_text();ast.parse(t);assert not any(x in t for x in bad)
def test_cache_when_present():
 if not p.CACHE_RECEIPT.exists():return
 r=json.load(open(p.CACHE_RECEIPT));i=json.load(open(p.CACHE/'index.json'));assert r['all_gates_passed'] and i['cases']==i['subjects']==31 and all(i['gates'].values());p.checked(p.CACHE/'index.json',r['index_sha256']);p.checked(p.CACHE/'images.f32',r['images_sha256']);p.checked(p.CACHE/'labels.u8',r['labels_sha256'])
def test_raw_when_present():
 path=p.RECORD/'RAW_WAW_POSTOP31.json'
 if not path.exists():return
 d=json.load(open(path));assert d['forwards']==279 and d['access']['waw_postop_only31'] is True;assert all(v is False for k,v in d['access'].items() if k!='waw_postop_only31');assert d['result_must_not_trigger_tuning_or_repeat'] is True
