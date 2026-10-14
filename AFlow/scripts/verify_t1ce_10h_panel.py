"""Check immutable recipe membership and exact cached single-model identity."""
import sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'archive/research_records/2026-09-21_t1ce_test270_10h'
def read(p):return json.loads(Path(p).read_text())
def main():
 f=read(R/'FINAL_PANEL_FREEZE.json');assert len(f['selected'])<=3 and len(f['recipes'])<=19
 sources={d['id']:read(R/'reports'/(d['id']+'.json')) for d in f['selected']};branches=[d.get('arm') for d in sources.values() if d.get('arm')]
 assert len(set(branches))==len(branches);cap=f.get('compute_cap',8)
 if cap!=8:
  a=read(R/'INFERENCE_COST_AMENDMENT.json');assert cap==11 and a['explicit_user_authorization'] is True
 identities=[]
 for recipe in f['recipes']:
  d=read(R/'reports'/(recipe['id']+'.json'));assert d['components']==recipe['components'] and d['threshold']==recipe['threshold']
  cost=sum(sources[k]['forward_equivalents'] for k in d['components']);assert abs(cost-d['forward_equivalents'])<1e-12 and cost<=cap+1e-10
  assert d['threshold'] in [.4,.5,.6] and (len(d['components'])<3 or d['threshold']==.5)
  if len(d['components'])==1 and d['threshold']==.5:
   src=sources[d['components'][0]];a={c['case_id']:c for c in src['per_case']};assert len(a)==len(d['per_case'])==270
   err=max(abs(c['locator_dsc']-a[c['case_id']]['locator_dsc']) for c in d['per_case']);assert err<1e-12
   assert all(c['automatic_point_hits_gt']==a[c['case_id']]['automatic_point_hits_gt'] and c['automatic_point']==a[c['case_id']]['automatic_point'] for c in d['per_case'])
   identities.append({'panel':d['id'],'source':src['id'],'max_dice_error':err})
 assert len(identities)==len(sources)
 result={'status':'passed','recipe_count':len(f['recipes']),'compute_cap':cap,'exact_single_model_checks':identities,'freeze_sha256':hashlib.sha256((R/'FINAL_PANEL_FREEZE.json').read_bytes()).hexdigest()}
 (R/'FINAL_PANEL_CHECKS.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
