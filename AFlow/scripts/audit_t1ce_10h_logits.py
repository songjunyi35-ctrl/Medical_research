"""Finite-value, native-shape and file-hash audit of already generated logits."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.t1ce_10h_common import *

def main():
 idx=read(OUT/'label_relay/test/index.json');results=[]
 for p in sorted((R/'reports').glob('*.json')):
  d=read(p)
  if 'logits_dir' not in d:continue
  path=Path(d['logits_dir'])
  if not path.exists():continue
  hashes={}
  for cid in idx['case_ids']:
   file=path/(cid+'.npy');x=np.load(file,allow_pickle=False)
   assert list(x.shape)==idx['geometry'][cid]['extent'] and x.dtype==np.float32
   assert np.isfinite(x).all(),(d['id'],cid,'nonfinite prediction')
   hashes[cid]=sha(file)
  assert len(hashes)==270;results.append({'id':d['id'],'status':'passed','case_sha256':hashes,'report_sha256':sha(p)})
  print(json.dumps({'utc':now(),'audited':d['id'],'cases':len(hashes)}),flush=True)
 suffix=sys.argv[1];write(R/('LOGIT_AUDIT_'+suffix+'.json'),{'utc':now(),'status':'passed','new_model_forward':False,'new_patient_image_or_label_read':False,'models':results})

if __name__=='__main__':main()
