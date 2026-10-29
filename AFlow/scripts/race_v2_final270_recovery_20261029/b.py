"""Bounded fixed-final recovery on1154259 GPU1; no training or recipe search."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from zoneinfo import ZoneInfo
import numpy as np
import torch
from scripts.race_v2_final270_recovery_20261029 import common as recovery_governance
from scripts.race_v2_inference_b_20261029.appearance_b.core import deploy
from scripts.race_v2_inference_a_20261029.statistics import summary_ci
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'archive/research_records/2026-10-29_race_v2_final270_recovery';R=BASE/'b';AR=BASE/'a'
TMP=Path('/tmp/aflow_race_v2_final270_recovery_20261029');BTMP=TMP/'b'
BOLD=ROOT/'archive/research_records/2026-10-29_race_v2_inference_b';OLD=ROOT/'archive/research_records/2026-10-29_race_e164_inference_test270'
SELECTION=ROOT/'archive/research_records/2026-10-29_race_v2_stage/FINAL_TEST_SELECTION.json'
SELECTION_SHA='5b74021964fe8b6d0f34fbdb35303d7d05be34d6796caec2adc6e6a68c6579bc'
CHECKPOINT=ROOT/'archive/research_records/2026-10-28_segresnet_w32_e171_test270/checkpoint_epoch171.pth'
CHECKPOINT_SHA='02a60b2d37da0d8de8018c221e1c00566106bd21d0d713b95ac3d9f94d87fade'
RACE_SHA='4bd6c35f7e12ecdf1540bf88128f9d3f9853340d801b32e77779fc4ba56eb3d7'
MANIFEST=ROOT/'data/datasets/brats2024_t1ce_va_hdfs_20260812_v1/sealed_test/test_cases_t1ce.jsonl'
UUID='GPU-e7e4ead3-6461-4fac-ac02-2e1e666fa462';CANVAS=(160,208,168)
WREADY=R/'W32_ATOMS_READY.json'

def now():return datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
def read(path):return json.loads(Path(path).read_text())
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as handle:
  for value in iter(lambda:handle.read(8<<20),b''):h.update(value)
 return h.hexdigest()
def write(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
 tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)
def exclusive(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('x') as handle:json.dump(value,handle,ensure_ascii=False,indent=2,allow_nan=False)
def slices(g):return tuple(slice(int(o),int(o)+int(e)) for o,e in zip(g['canvas_offset'],g['extent']))
def selection():
 assert sha(SELECTION)==SELECTION_SHA,'original FINAL remains immutable'
 root=read(SELECTION);assert root['registered_by']=='root' and root['approval_scope']=='final_frozen_test270_only'
 assert len(root['candidates'])==2
 candidate=next(c for c in root['candidates'] if c['kind']=='fusion')
 assert [m['model_id'] for m in candidate['models']]==['race164','w32e171']
 assert candidate['fusion']['space']=='image_appearance_unary'
 recipe=candidate['fusion']['algorithm_recipe'];bound=BOLD/'appearance_b/DEPLOYMENT_RECIPE.json'
 assert recipe==read(bound)['candidate'] and recipe['id']=='appearance_k1_lambda0.5'
 assert candidate['fusion']['algorithm_recipe_sha256']==sha(bound)
 assert read(bound)['source_sha256']==candidate['fusion']['algorithm_source_sha256']==sha(ROOT/'scripts/race_v2_inference_b_20261029/appearance_b/core.py')
 assert root['cohort']['scans']==270 and root['cohort']['subjects']==119 and root['cohort']['canvas']==list(CANVAS)
 assert sha(MANIFEST)==root['cohort']['manifest_sha256']
 for path,digest in root['entry_source_sha256'].items():assert sha(ROOT/path)==digest,path
 return root,candidate,recipe

def guard():
 auth=recovery_governance.guard();allocation=read(BASE/'RESOURCE_ALLOCATION.json')
 queue=allocation['queues']['b'];assert queue['worker']==1154259 and queue['gpus']==[1] and queue['gpu_uuid']==UUID
 done=read(ROOT/'archive/research_records/2026-10-29_race_v2_training/DONE.json');assert done['state']=='complete' and done['new_epoch']==done['target_new_epochs']==80
 assert sum(p.stat().st_size for p in R.rglob('*') if p.is_file())<=64*2**20
 return auth

def verify():
 guard();frozen=read(R/'EXECUTION_FREEZE.json')
 for path,digest in frozen['source_sha256'].items():assert sha(ROOT/path)==digest,path
 for path,digest in frozen.get('evidence_sha256',{}).items():assert sha(ROOT/path)==digest,path
 selection();return frozen

def lock():
 handle=recovery_governance.gpu_lock('b')
 write(R/'GPU_QUEUE.json',dict(at=now(),worker=1154259,physical_gpu=1,gpu_uuid=UUID,pid=os.getpid(),
  canonical_lock_path=str(Path(handle.name)),selection_sha256=SELECTION_SHA))
 return handle

def numeric_mode():
 torch.set_num_threads(2);torch.manual_seed(20261029);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True

def load_cpu():
 from sammed3d.t1ce_wt_capg3d import AutomaticPromptLocator
 assert sha(CHECKPOINT)==CHECKPOINT_SHA
 saved=torch.load(CHECKPOINT,map_location='cpu',weights_only=False);assert saved['epoch']==171
 model=AutomaticPromptLocator(center_aware=False,init_filters=32,dropout=.1);model.load_state_dict(saved['model_state_dict'],strict=True)
 assert sum(p.numel() for p in model.parameters())==18795969
 assert all(torch.equal(v,saved['model_state_dict'][k]) and bool(v.isfinite().all()) for k,v in model.state_dict().items())
 return model.eval().requires_grad_(False)

def cpu_info():
 import importlib.metadata
 model=load_cpu();assert not torch.cuda.is_initialized()
 write(R/'CPU_STRICT_RELOAD.json',dict(at=now(),status='passed',checkpoint_path=str(CHECKPOINT),checkpoint_sha256=CHECKPOINT_SHA,
  epoch=171,parameters=sum(p.numel() for p in model.parameters()),strict_cpu_reload=True,all_saved_tensors_equal_and_finite=True,
  architecture='sammed3d.t1ce_wt_capg3d.AutomaticPromptLocator',construction=dict(center_aware=False,init_filters=32,dropout=.1),
  python=sys.version,torch=torch.__version__,monai=importlib.metadata.version('monai'),CUDA_initialized=False,patient_model_forwards=0,
  original_selection_sha256=SELECTION_SHA,original_worker1154121_persistent_patient_receipts_absent=True,
  absence_is_not_absolute_proof_of_no_unrecorded_execution=True))

@torch.inference_mode()
def gate():
 verify();handle=lock();numeric_mode();torch.cuda.set_per_process_memory_fraction(.94);model=load_cpu().cuda();probes=[]
 for batch in (4,8):
  torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
  try:
   image=torch.zeros((batch,1,*CANVAS),device='cuda')
   with torch.autocast('cuda',dtype=torch.bfloat16):value=model(image).segmentation_logits
   assert value.shape==image.shape and bool(value.isfinite().all());torch.cuda.synchronize();begun=time.monotonic()
   for _ in range(3):
    with torch.autocast('cuda',dtype=torch.bfloat16):value=model(image).segmentation_logits
   torch.cuda.synchronize();speed=3*batch/(time.monotonic()-begun);peak=torch.cuda.max_memory_reserved()/2**30
   probes.append(dict(batch=batch,passed=peak<=38,cases_per_second=speed,peak_reserved_gib=peak));del image,value
  except torch.cuda.OutOfMemoryError as error:
   probes.append(dict(batch=batch,passed=False,error=repr(error),patient_forwards=0))
   if 'image' in locals():del image
   if 'value' in locals():del value
   torch.cuda.empty_cache()
 viable=[p for p in probes if p['passed']];assert viable
 fastest=max(p['cases_per_second'] for p in viable);batch=min(p['batch'] for p in viable if p['cases_per_second']>=.97*fastest);tail=270%batch
 torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
 if tail:
  image=torch.zeros((tail,1,*CANVAS),device='cuda')
  with torch.autocast('cuda',dtype=torch.bfloat16):value=model(image).segmentation_logits
  assert value.shape==image.shape and bool(value.isfinite().all());del image,value
 assert torch.cuda.max_memory_reserved()/2**30<=38
 speed=min(p['cases_per_second'] for p in viable if p['batch']==batch);estimate=2160/speed*1.5+120
 assert time.time()+estimate<guard()['deadline_unix']
 write(R/'NUMERIC_GATE.json',dict(at=now(),status='passed',worker=1154259,gpu_uuid=UUID,canvas=list(CANVAS),batch=batch,tail=tail,
  probes=probes,estimated_forward_and_decode_seconds=estimate,strict_cpu_reload=True,pure_synthetic=True,patient_forwards=0,selection_sha256=SELECTION_SHA))

def data(scoring=False):
 ready=read(AR/'CACHE_READY.json');assert ready['status']=='passed' and ready['all_four_payloads_exact']
 assert ready['worker']==1154259 and ready['selection_sha256']==SELECTION_SHA
 cache=Path(ready['cache_path']);canonical=Path(ready['canonical_cache_path']);assert cache==TMP/'native' and canonical==TMP/'canonical128'
 expected=read(OLD/'CACHE_READY.json')['payload_sha256'];assert ready['payload_sha256']==expected
 files={'native_image':(cache/'images.f32',270*np.prod(CANVAS)*4),'native_label':(cache/'labels.u8',270*np.prod(CANVAS)),
  'canon_image':(canonical/'images.f32',270*128**3*4),'canon_label':(canonical/'labels.u8',270*128**3)}
 actual={}
 for key,(path,size) in files.items():assert path.is_file() and path.stat().st_size==size;actual[key]=sha(path)
 assert actual==expected
 assert sha(cache/'index.json')==ready['index_sha256'];index=read(cache/'index.json')
 cohort=[json.loads(s) for s in MANIFEST.read_text().splitlines() if s.strip()]
 assert len(cohort)==270 and len({r['subject_id'] for r in cohort})==119
 assert index['case_ids']==[r['case_id'] for r in cohort] and index['canvas']==list(CANVAS)
 assert ready['manifest_sha256']==sha(MANIFEST)
 canonical_geometry=read(ROOT/'workspace/BraTS/cache/t1ce_wt_n4_braincrop_128_20260817/test270/index.json')['crop_geometry']
 for row in cohort:
  current=index['geometry'][row['case_id']];previous=canonical_geometry[row['case_id']]
  assert all(current[k]==previous[k] for k in ('origin','extent','native_shape'))
  assert current['canvas_offset']==[(c-e)//2 for c,e in zip(CANVAS,current['extent'])]
 write(R/'ACTUAL_DATA_VERIFIED.json',dict(at=now(),status='passed',worker=1154259,all_four_actual_payloads_exact=True,payload_sha256=actual,
  new_index_sha256=ready['index_sha256'],historical_index_sha256=ready['historical_index_sha256'],index_byte_sha_reuse_claim=False,
  semantic_crop_geometry_exact=True,selection_sha256=SELECTION_SHA))
 images=np.memmap(cache/'images.f32',np.float32,'r',shape=(270,*CANVAS))
 labels=np.memmap(cache/'labels.u8',np.uint8,'r',shape=(270,*CANVAS)) if scoring else None
 return ready,index,cohort,images,labels

@torch.inference_mode()
def forward():
 verify();assert not WREADY.exists() and not (R/'MODEL_PREACCESS.json').exists(),'no automatic patient neural retry'
 handle=lock();numeric_mode();numeric=read(R/'NUMERIC_GATE.json');assert numeric['status']=='passed';batch=numeric['batch']
 ready,index,cohort,images,_=data(False);model=load_cpu().cuda();torch.cuda.set_per_process_memory_fraction(.94)
 exclusive(R/'MODEL_PREACCESS.json',dict(at=now(),worker=1154259,gpu_uuid=UUID,selection_sha256=SELECTION_SHA,
  recovery_authorization_sha256=sha(BASE/'AUTHORIZATION.json'),checkpoint_sha256=CHECKPOINT_SHA,maximum_case_forwards=2160,
  cache_ready_sha256=sha(AR/'CACHE_READY.json'),new_index_sha256=ready['index_sha256'],automatic_neural_retry=False,
  old_persistent_model_preaccess_absent=True,absence_is_not_absolute_proof_of_no_unrecorded_execution=True))
 entries=[];attempts=0;calls=0;started=time.monotonic()
 for first in range(0,270,batch):
  guard();slots=list(range(first,min(first+batch,270)));image=torch.from_numpy(np.array(images[slots],np.float32))[:,None].cuda()
  for view in range(8):
   guard();dims=tuple(a+2 for a in range(3) if view&(1<<a));changed=torch.flip(image,dims) if dims else image
   attempts+=len(slots);assert attempts<=2160
   write(R/'FORWARD_ATTEMPT_JOURNAL.json',dict(at=now(),case_forward_attempt_budget_consumed=attempts,view=view,slots=slots,completed_atoms=len(entries),call_pending=True))
   with torch.autocast('cuda',dtype=torch.bfloat16):value=model(changed).segmentation_logits
   calls+=1;value=value.float();value=torch.flip(value,dims) if dims else value;assert bool(value.isfinite().all())
   for k,slot in enumerate(slots):
    array=value[(k,0)+slices(index['geometry'][cohort[slot]['case_id']])].contiguous().cpu().numpy()
    path=BTMP/'atoms'/f'{slot:03}'/f'flip{view}.npy';path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_name(path.name+'.tmp')
    with temporary.open('xb') as output:np.save(output,array,allow_pickle=False)
    assert not path.exists();os.replace(temporary,path)
    entries.append(dict(slot=slot,view=view,path=str(path),shape=list(array.shape),sha256=sha(path)))
   write(R/'FORWARD_JOURNAL.json',dict(at=now(),case_forwards_completed=len(entries),case_forward_attempts=attempts,batch_calls=calls,maximum=2160,atoms=entries,call_pending=False))
  del image,changed,value
 assert len(entries)==attempts==2160 and len({(e['slot'],e['view']) for e in entries})==2160
 output=dict(at=now(),status='complete',worker=1154259,model_id='w32e171',checkpoint_path=str(CHECKPOINT),checkpoint_sha256=CHECKPOINT_SHA,
  architecture='sammed3d.t1ce_wt_capg3d.AutomaticPromptLocator',construction=dict(center_aware=False,init_filters=32,dropout=.1),
  selection_sha256=SELECTION_SHA,recovery_authorization_sha256=sha(BASE/'AUTHORIZATION.json'),manifest_sha256=sha(MANIFEST),
  index_sha256=ready['index_sha256'],historical_index_sha256=ready['historical_index_sha256'],cache_path=ready['cache_path'],canvas=list(CANVAS),
  case_forwards=2160,new_model_forwards=2160,batch=batch,tail=270%batch,batch_calls=calls,seconds=time.monotonic()-started,
  atoms=entries,old_worker_missing=True,new_training_epochs=0,independent_confirmation=False)
 write(WREADY,output)
 for entry in entries:Path(entry['path']).chmod(0o444)
 for directory in (BTMP/'atoms').iterdir():directory.chmod(0o555)
 (BTMP/'atoms').chmod(0o555)

@torch.inference_mode()
def score():
 verify();assert not (R/'RAW_TEST270.json').exists(),'complete raw: statistics/audit only'
 handle=lock();numeric_mode();root,candidate,recipe=selection();ready,index,cohort,images,labels=data(True)
 race=read(AR/'RACE_ATOMS_READY.json');w32=read(WREADY)
 old_race=read(ROOT/'archive/research_records/2026-10-29_race_v2_inference_a/final270/reuse_race164/ATOMS_READY.json')
 old_map={(e['slot'],e['view']):e for e in old_race['atoms']};sources=[]
 for info,expected_sha in [(race,RACE_SHA),(w32,CHECKPOINT_SHA)]:
  assert info['checkpoint_sha256']==expected_sha and info['manifest_sha256']==sha(MANIFEST)
  assert info['index_sha256']==ready['index_sha256'] and info['case_forwards']==2160
  atoms={(e['slot'],e['view']):e for e in info['atoms']};assert len(atoms)==2160
  if expected_sha==RACE_SHA:assert all(e['sha256']==old_map[key]['sha256'] for key,e in atoms.items()),'RACE reconstruction differs: report before scoring'
  sources.append(atoms)
 exclusive(R/'SCORING_PREACCESS.json',dict(at=now(),selection_sha256=SELECTION_SHA,recovery_authorization_sha256=sha(BASE/'AUTHORIZATION.json'),
  RACE_ready_sha256=sha(AR/'RACE_ATOMS_READY.json'),W32_ready_sha256=sha(WREADY),fixed_recipe=recipe,GT_only_scorer=True,current_test_search=False))
 records=[];begun=time.monotonic()
 for slot,row in enumerate(cohort):
  guard();sl=slices(index['geometry'][row['case_id']]);image=np.array(images[slot][sl],np.float32);loaded=[]
  for source in sources:
   views=[]
   for view in range(8):
    atom=source[slot,view];path=Path(atom['path']);assert sha(path)==atom['sha256'];array=np.load(path,allow_pickle=False)
    assert array.dtype==np.float32 and array.shape==image.shape and list(array.shape)==atom['shape'] and np.isfinite(array).all()
    views.append(torch.from_numpy(array).cuda())
   loaded.append(views)
  prediction=deploy(*loaded,image,recipe);mask=prediction['mask'];point=prediction['automatic_point']
  # Full-crop label first enters only after all seed fitting/score/mask/Point decisions.
  target=np.array(labels[slot][sl]>0);tp=int((mask&target).sum());pred=int(mask.sum());positive=int(target.sum());den=pred+positive
  path=BTMP/'predictions'/f'{slot:03}.npz';path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_name(path.name+'.tmp')
  with temporary.open('xb') as output:np.savez_compressed(output,mask_bits=np.packbits(mask.ravel()),shape=np.array(mask.shape),point=np.array(point))
  assert not path.exists();os.replace(temporary,path)
  records.append(dict(case_id=row['case_id'],subject_id=row['subject_id'],source_slot=slot,tp=tp,fp=pred-tp,fn=positive-tp,
   locator_dsc=float(np.float32(2*tp)/np.float32(den)) if den else 1.,automatic_point=point,automatic_point_hits_gt=bool(target[tuple(point)]),
   prediction_path=str(path),prediction_sha256=sha(path),prediction_shape=list(mask.shape),appearance_trust=prediction['trust'],GMM=prediction['diagnostics']))
  write(R/'SCORING_JOURNAL.json',dict(at=now(),completed=slot+1,total=270,new_model_forwards_by_decoder=0))
  del loaded,views,image,prediction
 write(R/'RAW_TEST270.json',dict(at=now(),candidate_id=candidate['candidate_id'],per_case=records,models=candidate['models'],fusion=candidate['fusion'],
  selection_sha256=SELECTION_SHA,recovery_authorization_sha256=sha(BASE/'AUTHORIZATION.json'),execution_worker=1154259,
  actual_atom_ready_paths=[str(AR/'RACE_ATOMS_READY.json'),str(WREADY)],actual_native_cache=ready['cache_path'],new_native_index_sha256=ready['index_sha256'],
  new_model_forwards=2160,RACE_reconstruction_forwards=2160,total_recovery_case_forwards=4320,current_test_search=False,
  inherited_test_tuned_history=True,independent_confirmation=False,decode_seconds=time.monotonic()-begun))


def audit():
 verify();root,candidate,recipe=selection();raw=read(R/'RAW_TEST270.json');assert raw['selection_sha256']==SELECTION_SHA
 ready,index,cohort,_,labels=data(True);records=raw['per_case'];assert [(r['case_id'],r['subject_id']) for r in records]==[(r['case_id'],r['subject_id']) for r in cohort]
 audits=[]
 for slot,row in enumerate(records):
  path=Path(row['prediction_path']);assert sha(path)==row['prediction_sha256'];saved=np.load(path,allow_pickle=False);shape=tuple(saved['shape'])
  assert list(shape)==row['prediction_shape']==index['geometry'][row['case_id']]['extent']
  mask=np.unpackbits(saved['mask_bits'])[:int(np.prod(shape))].reshape(shape).astype(bool);point=saved['point'].tolist();target=np.array(labels[slot][slices(index['geometry'][row['case_id']])]>0)
  tp=int((mask&target).sum());fp=int((mask&~target).sum());fn=int((~mask&target).sum());den=2*tp+fp+fn;dice=float(np.float32(2*tp)/np.float32(den)) if den else 1.
  assert (tp,fp,fn,dice,point,bool(target[tuple(point)]))==(row['tp'],row['fp'],row['fn'],row['locator_dsc'],row['automatic_point'],row['automatic_point_hits_gt'])
  audits.append(dict(case_id=row['case_id'],tp=tp,fp=fp,fn=fn,locator_dsc=dice,point=point,prediction_sha256=sha(path)))
 receipt=R/'MASK_AUDIT_FUSION.json';write(receipt,dict(at=now(),status='passed',worker=1154259,scans=270,selection_sha256=SELECTION_SHA,
  raw_path=str(R/'RAW_TEST270.json'),raw_sha256=sha(R/'RAW_TEST270.json'),all_masks_counts_points_exact=True,per_case=audits,new_model_forwards=0))
 wrapper=dict(raw,masks_audit_receipt_path=str(receipt),masks_audit_receipt_sha256=sha(receipt),canonical_raw_path=str(R/'RAW_TEST270.json'),canonical_raw_sha256=sha(R/'RAW_TEST270.json'))
 write(R/'RAW_FUSION.json',wrapper)
 write(R/'RESULT.json',dict(at=now(),state='complete',candidate_id=candidate['candidate_id'],kind='fusion',metrics=summary_ci(records),selection_sha256=SELECTION_SHA,
  worker=1154259,models=2,forwards_per_scan=16,EM_iterations_per_class=5,new_W32_case_forwards=2160,new_RACE_reconstruction_case_forwards=2160,
  total_new_recovery_case_forwards=4320,masks_audit_receipt_path=str(receipt),masks_audit_receipt_sha256=sha(receipt),raw_fusion_path=str(R/'RAW_FUSION.json'),raw_fusion_sha256=sha(R/'RAW_FUSION.json'),independent_confirmation=False))

def watch():
 verify();exclusive(R/'WATCH_START.json',dict(at=now(),pid=os.getpid(),worker=1154259,physical_gpu=1,selection_sha256=SELECTION_SHA,automatic_neural_retry=False))
 def run(stage):subprocess.run([sys.executable,'-u','-m','scripts.race_v2_final270_recovery_20261029.b',stage],check=True)
 if not (R/'NUMERIC_GATE.json').exists():run('gate')
 while not (AR/'CACHE_READY.json').exists():guard();time.sleep(15)
 if not WREADY.exists():run('forward')
 while not (AR/'RACE_ATOMS_READY.json').exists():guard();time.sleep(15)
 if not (R/'RAW_TEST270.json').exists():run('score')
 run('audit');write(R/'WATCH_STATUS.json',dict(at=now(),state='complete',worker=1154259,selection_sha256=SELECTION_SHA,result_sha256=sha(R/'RESULT.json')))

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('cpu','gate','forward','score','audit','watch'));args=parser.parse_args()
 try:{'cpu':cpu_info,'gate':gate,'forward':forward,'score':score,'audit':audit,'watch':watch}[args.stage]()
 except BaseException as error:
  import traceback
  write(R/f'FAILURE_{args.stage}_{time.time_ns()}.json',dict(at=now(),stage=args.stage,error=repr(error),traceback=traceback.format_exc(),automatic_neural_retry=False));raise
