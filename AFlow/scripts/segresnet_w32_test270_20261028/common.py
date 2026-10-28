import os,json,hashlib,datetime,shutil,time
from pathlib import Path
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[2]
R=ROOT/'archive/research_records/2026-10-28_segresnet_w32_e171_test270'
TMP=Path('/tmp/aflow_w32_e171_test270_20261028');CACHE=TMP/'native';CANON=TMP/'canonical128'
PY=ROOT.parent/'sammed3d-venv/bin/python'
TRAIN=ROOT/'archive/research_records/2026-10-27_segresnet_width_resume'
SOURCE=ROOT/'workspace/BraTS/segresnet_t1ce_wt/width_resume_20261027/w32/best.pth'
CHECKPOINT=R/'checkpoint_epoch171.pth'
CHECKPOINT_SHA='02a60b2d37da0d8de8018c221e1c00566106bd21d0d713b95ac3d9f94d87fade'
WORKER=1152470;EPOCH=171;PARAMETERS=18795969;CANVAS=(160,208,168)
ID='segresnet_width32_native_epoch171_test270_20261028'
CANON_INDEX=ROOT/'workspace/BraTS/cache/t1ce_wt_n4_braincrop_128_20260817/test270/index.json'
NATIVE_REF=ROOT/'archive/research_records/2026-09-20_segresnet_native_res_e173_test270/CACHE_RECEIPT.json'
from scripts.followup_test270_common import MANIFEST,SCHEMA,validate_rows,summary_ci,paired,numeric_mode
from scripts.segresnet_native_res_test_common import pad_to_canvas,case_metrics,NativeTestDataset

def now():return datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def write(p,data):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(tmp,p)
def write_new(p,data):
 with Path(p).open('x') as f:json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False)
def rows(split):return [json.loads(s) for s in MANIFEST[split].read_text().splitlines() if s.strip()]
def env():
 e=os.environ.copy();e.update(PYTHONPATH=str(ROOT),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MPLCONFIGDIR=str(TMP/'mpl'));return e
def verify():
 for name in ['FREEZE.json','EXECUTION_FREEZE.json']:
  if not (R/name).exists():continue
  for p,h in read(R/name)['files'].items():
   if sha(ROOT/p)!=h:raise RuntimeError('frozen source changed '+p)
 assert sha(CHECKPOINT)==CHECKPOINT_SHA
 deadline();disk()
def deadline():
 if time.time()>read(R/'AUTHORIZATION.json')['deadline_unix']:raise RuntimeError('four-hour evaluation budget exceeded')
def disk(extra=0):
 if shutil.disk_usage(ROOT).free-extra<20*2**30:raise RuntimeError('persistent reserve below20GiB')
 used=sum(p.stat().st_size for p in R.rglob('*') if p.is_file())
 if used+extra>2*2**30:raise RuntimeError('new persistent storage above2GiB')
def status(stage,**kw):write(R/'PIPELINE_STATUS.json',{'at':now(),'state':stage,'worker':WORKER,**kw})
def validate_manifests():
 expected={'train':(972,444),'validation':(108,50),'test':(270,119)};subjects={}
 for split,p in MANIFEST.items():
  data=rows(split);subjects[split]={d['subject_id'] for d in data}
  assert (len(data),len(subjects[split]))==expected[split] and len({d['case_id'] for d in data})==len(data)
 assert not any(subjects[a]&subjects[b] for a,b in [('train','validation'),('train','test'),('validation','test')])
 assert read(CANON_INDEX)['case_ids']==[d['case_id'] for d in rows('test')]
def load_model(device):
 import torch
 from sammed3d.t1ce_wt_capg3d import AutomaticPromptLocator
 c=torch.load(CHECKPOINT,map_location='cpu',weights_only=False);assert c['epoch']==EPOCH
 m=AutomaticPromptLocator(center_aware=False,init_filters=32,dropout=.1).to(device)
 assert sum(p.numel() for p in m.parameters())==PARAMETERS
 m.load_state_dict(c['model_state_dict'],strict=True)
 for k,v in m.state_dict().items():assert torch.equal(v.detach().cpu(),c['model_state_dict'][k]) and torch.isfinite(v).all(),k
 m.eval().requires_grad_(False);return m
