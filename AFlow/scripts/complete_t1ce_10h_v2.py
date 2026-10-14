"""Archival completion wrapper; unchanged training, selection and monitor deadline."""
import sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import complete_t1ce_10h as original
from scripts.archive_t1ce_10h_candidates import main as archive_candidates
from scripts.t1ce_10h_common import *
publish=original.publish;evidence=original.evidence

def publish_all(final=False):publish(final);archive_candidates()
def evidence_all():
 evidence();p=R/'EVIDENCE_SHA256.json';d=read(p)
 for arm in 'ABC':
  for f in ['RESULT.md','RESULT.json']:
   q=R/arm/f
   if q.exists():d['files'][str(q.relative_to(ROOT))]=sha(q)
 write(p,d)
original.publish=publish_all;original.evidence=evidence_all

if __name__=='__main__':
 try:original.main()
 except BaseException:write(R/f'COMPLETION_V2_FATAL_{int(time.time())}.json',{'utc':now(),'traceback':traceback.format_exc()});raise
