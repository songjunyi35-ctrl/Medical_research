"""Audit, build only the new native test cache, and image-only target weights."""
import sys, os, time, multiprocessing as mp
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.t1ce_10h_common import *

def audit():
 verify();sets={};receipts={}
 for split,(n,ns) in {'train':(972,444),'validation':(108,50),'test':(270,119)}.items():
  rr=rows(split);assert len(rr)==len({r['case_id'] for r in rr})==n
  sets[split]={r['subject_id'] for r in rr};assert len(sets[split])==ns
  idx=read(CANON[split]/'index.json');assert idx['count']==n and set(idx['case_ids'])=={r['case_id'] for r in rr}
  assert not idx['preprocessing']['uses_gt_for_localization']
  rec={'manifest_sha256':sha(MANIFEST[split]),'index_sha256':sha(CANON[split]/'index.json'),'scans':n,'subjects':ns}
  for name,width in [('images.f32',4),('labels.u8',1)]:
   path=CANON[split]/name;assert path.stat().st_size==n*128**3*width
   h=sha(path);assert h==idx[name.split('.')[0]+'_sha256'];rec[name]=h
  receipts[split]=rec
 assert not sets['train']&sets['validation'] and not sets['train']&sets['test'] and not sets['validation']&sets['test']
 for name,(_,h,_) in SOURCES.items():assert sha(source(name))==h
 for split,n in [('train',972),('validation',108)]:
  idx=read(NATIVE/split/'index.json');assert idx['count']==n and idx['manifest_sha256']==MANIFEST_SHA[split]
  assert idx['test_access'] is False and idx['preprocessing']['uses_gt_for_localization'] is False
  for name,width in [('images.f32',4),('labels.u8',1)]:
   path=NATIVE/split/name;assert path.stat().st_size==n*math.prod(idx['canvas'])*width
   assert sha(path)==idx[name.split('.')[0]+'_sha256']
  receipts['native_'+split]={'index_sha256':sha(NATIVE/split/'index.json'),'images_sha256':idx['images_sha256'],'labels_sha256':idx['labels_sha256']}
 write(R/'DATA_AUDIT.json',{'utc':now(),'status':'passed','receipts':receipts,'patient_isolation':True,'pixel_hash_reads':True,'target_features_or_labels_analyzed':False,'disk':disk()})

def build_test():
 verify();target=SCRATCH/'test_native'
 if (target/'index.json').exists():
  idx=read(target/'index.json')
  for name in ['images.f32','labels.u8']:assert sha(target/name)==idx[name.split('.')[0]+'_sha256']
  return
 from scripts import build_segresnet_native_res_test_cache as builder
 rr=rows('test');canon=read(CANON['test']/'index.json');n=270;shape=(160,208,168);vox=math.prod(shape)
 disk(n*vox*5);target.mkdir(parents=True,exist_ok=True)
 if any(target.iterdir()):raise RuntimeError('partial cache exists: manual engineering recovery needed')
 write(R/'TEST_PIXEL_PREACCESS.json',{'utc':now(),'test_tuned':True,'purpose':'new campaign native target cache and diagnostics','segmentation_training_uses_test_labels':False,'protocol_sha256':sha(R/'PROTOCOL.json'),'source_manifest_sha256':sha(MANIFEST['test'])})
 for name,width in [('images.f32',4),('labels.u8',1)]:
  with (target/name).open('xb') as f:f.truncate(n*vox*width)
 config={'images':str(target/'images.f32'),'labels':str(target/'labels.u8'),'canon_dir':str(CANON['test'])}
 tasks=[(i,r,canon['crop_geometry'][r['case_id']]) for i,r in enumerate(rr)]
 geo={};tick=time.monotonic()
 with mp.get_context('spawn').Pool(12,initializer=builder.init,initargs=(config,)) as pool:
  for k,(slot,cid,g,err) in enumerate(pool.imap_unordered(builder.worker,tasks,chunksize=1),1):
   if err:raise RuntimeError(cid+': '+err)
   assert g['canonical_label_exact'] and g['canonical_geometry_exact'] and g['canonical_image_within_tolerance'];geo[cid]=g
   write(R/'CACHE_PROGRESS.json',{'utc':now(),'cases':k,'total':n,'seconds':time.monotonic()-tick})
   if k%20==0:print(json.dumps({'utc':now(),'cache_done':k,'seconds':time.monotonic()-tick}),flush=True)
 idx={'count':n,'case_ids':[r['case_id'] for r in rr],'canvas':list(shape),'geometry':geo,'split':'test','test_access':True,'test_tuned':True,'manifest_sha256':sha(MANIFEST['test']),'preprocessing':{'uses_gt_for_localization':False,'native_1mm':True,'N4':True}}
 for name in ['images.f32','labels.u8']:idx[name.split('.')[0]+'_sha256']=sha(target/name)
 prior=read(ROOT/'archive/research_records/2026-09-20_segresnet_native_res_e173_test270/CACHE_RECEIPT.json')
 for key in ['images_sha256','labels_sha256']:assert idx[key]==prior[key],('rebuild differs from historical native test',key)
 write(target/'index.json',idx);write(R/'NATIVE_TEST_CACHE_RECEIPT.json',{'utc':now(),'path':str(target),'index_sha256':sha(target/'index.json'),'images_sha256':idx['images_sha256'],'labels_sha256':idx['labels_sha256'],'historical_payload_exact':True,'seconds':time.monotonic()-tick})

def label_relay():
 for split,path in [('validation',NATIVE/'validation'),('test',SCRATCH/'test_native')]:
  idx=read(path/'index.json');shape=idx['canvas'];labels=np.memmap(path/'labels.u8',mode='r',dtype='uint8',shape=(idx['count'],*shape))
  target=OUT/'label_relay'/split;target.mkdir(parents=True,exist_ok=True)
  arrays={cid:np.array(labels[i][crop_slices(idx['geometry'][cid])]) for i,cid in enumerate(idx['case_ids'])}
  disk(sum(a.nbytes for a in arrays.values()));np.savez_compressed(target/'labels.npz',**arrays)
  write(target/'index.json',{'case_ids':idx['case_ids'],'geometry':idx['geometry'],'source_index_sha256':sha(path/'index.json'),'labels_sha256':sha(target/'labels.npz'),'split':split,'test_tuned':True})
 write(R/'LABEL_RELAY_RECEIPT.json',{'utc':now(),'bytes':used(OUT/'label_relay'),'evaluation_only':True,'test_labels_in_training':False})

def features_and_weights():
 verify();from sklearn.cluster import KMeans
 values={};ids={}
 write(R/'FEATURE_PREACCESS.json',{'utc':now(),'splits':['train','test'],'image_only':True,'test_tuned':True,'purpose':'target image distribution weights','segmentation_training_uses_test_labels':False})
 for split in ['train','test']:
  idx=read(CANON[split]/'index.json');ids[split]=idx['case_ids'];x=np.memmap(CANON[split]/'images.f32',mode='r',dtype='float32',shape=(idx['count'],128,128,128));out=[]
  for i,cid in enumerate(ids[split]):
   nz=x[i][x[i]!=0];assert len(nz) and np.isfinite(nz).all()
   out.append(list(idx['crop_geometry'][cid]['extent'])+np.quantile(nz,[.01,.05,.25,.5,.75,.95,.99]).tolist())
  values[split]=np.asarray(out,dtype=np.float64)
 mean=values['train'].mean(0);std=values['train'].std(0);std[std<1e-8]=1
 model=KMeans(n_clusters=8,n_init=10,random_state=SEED).fit((values['train']-mean)/std)
 tr=model.labels_;te=model.predict((values['test']-mean)/std);ntr=np.bincount(tr,minlength=8);nte=np.bincount(te,minlength=8)
 ratio=((nte+1)/(270+8))/((ntr+1)/(972+8));clipped=np.clip(ratio,.5,2);w=clipped[tr];w/=w.mean()
 assert np.isfinite(w).all() and np.all(w>0) and abs(w.sum()-972)<1e-8
 write(R/'B/WEIGHTS.json',{'utc':now(),'case_weights':dict(zip(ids['train'],w.tolist())),'image_only':True,'feature_names':['extent_x','extent_y','extent_z','q01','q05','q25','q50','q75','q95','q99'],'standardization_mean':mean.tolist(),'standardization_std':std.tolist(),'centers':model.cluster_centers_.tolist(),'train_counts':ntr.tolist(),'target_counts':nte.tolist(),'raw_ratio':ratio.tolist(),'clipped_ratio':clipped.tolist(),'min_weight':float(w.min()),'max_weight':float(w.max()),'sum':float(w.sum()),'effective_sample_size':float(w.sum()**2/(w*w).sum()),'test_tuned':True,'test_label_access':False,'seed':SEED})
 write(R/'B/READY.json',{'utc':now(),'weights_sha256':sha(R/'B/WEIGHTS.json')})

def main():
 if sys.argv[1]=='dual':audit();build_test();label_relay();write(R/'DATA_READY.json',{'utc':now(),'status':'passed'})
 elif sys.argv[1]=='weights':features_and_weights()
 else:raise ValueError('dual or weights')

if __name__=='__main__':
 try:main()
 except BaseException:
  import traceback
  write(R/f'PREP_FAILURE_{sys.argv[1]}_{int(time.time())}.json',{'utc':now(),'traceback':traceback.format_exc()});raise
