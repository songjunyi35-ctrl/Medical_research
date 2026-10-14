"""Original 972/108 split contract; test is manifest metadata only."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'data/datasets/brats2024_t1ce_va_hdfs_20260812_v1'
FIT=BASE/'train_t1ce_wt.jsonl';DEV=BASE/'validation_t1ce_wt.jsonl';TEST=BASE/'sealed_test/test_cases_t1ce.jsonl'
FIT_SHA='6728d8d5e6846f90f6e5d0ad47acc4f2706eed42763ace4e228abcbbd6a4cf65'
DEV_SHA='f85f7df19908c7ab5bc7aafc2ce16fab1ff9786dbf3ea4ee72b97721888d35fb'
def validate_rows(fit,dev,test):
 for rows,n,ns in [(fit,972,444),(dev,108,50),(test,270,119)]:
  if len(rows)!=n or len({r['case_id'] for r in rows})!=n or len({r['subject_id'] for r in rows})!=ns:raise ValueError('count/duplicate mismatch')
 for a,b in [(fit,dev),(fit,test),(dev,test)]:
  if {r['case_id'] for r in a}&{r['case_id'] for r in b} or {r['subject_id'] for r in a}&{r['subject_id'] for r in b}:raise ValueError('split overlap')
 for rows,split in [(fit,'train'),(dev,'validation')]:
  for r in rows:
   if r['split']!=split or r['modality']!='t1ce' or r['foreground_labels']!=[1,2,3] or r['excluded_labels']!=[4]:raise ValueError('data semantics changed')
 return fit,dev

def check_roles():
 for p,h in [(FIT,FIT_SHA),(DEV,DEV_SHA)]:
  if hashlib.sha256(p.read_bytes()).hexdigest()!=h:raise ValueError('original manifest changed')
 rows=[[json.loads(x) for x in p.read_text().splitlines() if x.strip()] for p in [FIT,DEV,TEST]]
 return validate_rows(*rows)
