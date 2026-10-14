"""Parallelize byte verification only; preserve frozen image arrays and data roles."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.sammed3d_graymean_single972 import *
import concurrent.futures

def verify(row):
 source=row['source'];p=ROOT/row['path'];q={k:v for k,v in row.items() if k!='source'}
 prior=q if 'sha256' in q else json.loads((WORK/'cache'/f"{row['case_id']}.source.json").read_text())
 hashes={'image_sha256':sha(Path(source['image_path'])),'label_sha256':sha(Path(source['label_path'])),'sha256':sha(p)}
 for name,h in hashes.items():
  if prior[name]!=h:raise RuntimeError(f"source/cache changed: {row['case_id']} {name}")
 q.update(prior);q.update(role=row['role'],path=row['path'],subject_id=row['subject_id'],bytes=p.stat().st_size,reused_cache=('sha256' in row));return q
if __name__=='__main__':
 try:
  plan=json.loads((RECORD/'INPUT_PLAN.json').read_text());t=time.monotonic();items=[]
  write(RECORD/'PARALLEL_SOURCE_PREACCESS.json',{'utc':now(),'workers':8,'scope':'same1080 train/validation source hashes, no model evaluation/test','input_plan_sha256':sha(RECORD/'INPUT_PLAN.json')})
  with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
   for row in pool.map(verify,plan['cases']):
    items.append(row)
    if len(items)%108==0:print(f'verified {len(items)}/1080',flush=True)
  if len(items)!=1080:raise RuntimeError('incomplete audit')
  write(RECORD/'CACHE_INDEX.json',{'created_utc':now(),'cases':items,'count':1080,'bytes':sum(r['bytes'] for r in items),'new_bytes':sum(r['bytes'] for r in items if not r['reused_cache']),'preaccess_sha256':sha(RECORD/'CACHE_PREACCESS.json'),'parallel_source_preaccess_sha256':sha(RECORD/'PARALLEL_SOURCE_PREACCESS.json'),'verification_seconds':time.monotonic()-t})
  print('ALL1080 SOURCE AND CACHE HASHES VERIFIED',flush=True)
 except Exception as e:write(RECORD/'FAILURE_cache_parallel.json',{'utc':now(),'error':repr(e),'traceback':traceback.format_exc()});raise
