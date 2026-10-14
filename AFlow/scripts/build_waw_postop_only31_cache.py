#!/usr/bin/env python3
"""Build canonical WAW postop-only31 cache."""
from __future__ import annotations
import argparse,hashlib,json,multiprocessing as mp,os,shutil,time
from datetime import datetime,timezone
from pathlib import Path
import nibabel as nib,numpy as np,torch
from sammed3d import t1ce_binary
from scripts.build_no_n4_clean_cache import arrays
from scripts import waw_postop_only31_protocol as p
CFG={};ORIG=t1ce_binary._case_id
def init(cfg):
 CFG.update(cfg);CFG['allowed']=set(cfg['allowed']);torch.set_num_threads(1);import SimpleITK as sitk;sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1);t1ce_binary._case_id=lambda v:v if v in CFG['allowed'] else ORIG(v);CFG['ifd']=os.open(cfg['images'],os.O_WRONLY);CFG['lfd']=os.open(cfg['labels'],os.O_WRONLY)
def arrsha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def smap():return {x['case_id']:x for x in json.load(open(p.SOURCE_RECEIPT))['selected']['files']}
def prepare(row,sources):
 s=sources[row['case_id']];ip=Path(s['image_path']);mpath=Path(s['mask_path']);p.checked(ip,s['image_sha256']);p.checked(mpath,s['mask_sha256']);ni=nib.as_closest_canonical(nib.load(str(ip),mmap=True));nm=nib.as_closest_canonical(nib.load(str(mpath),mmap=True));err=float(np.max(np.abs(ni.affine-nm.affine)))
 if ni.shape!=nm.shape or err>1e-5:raise ValueError('geometry')
 image=np.ascontiguousarray(np.asanyarray(ni.dataobj));raw=np.ascontiguousarray(np.asanyarray(nm.dataobj));rounded=np.rint(raw);values=sorted(int(x) for x in np.unique(rounded))
 if not np.isfinite(image).all() or not (image>0).any() or not np.allclose(raw,rounded,rtol=0,atol=1e-5) or not set(values)<={0,1,2,3,4}:raise ValueError('domain')
 label=np.ascontiguousarray(np.isin(rounded,[1,2,3]),dtype=np.uint8);oi,ol,geo=arrays(row['case_id'],image,label,nib.affines.voxel_sizes(ni.affine),use_n4=True);crop=tuple(slice(o,o+e) for o,e in zip(geo['origin'],geo['extent']));cropped=int(label[crop].sum())
 if cropped!=int(label.sum()) or not ol.any():raise ValueError('WT crop')
 return oi,ol,{'native_shape':list(ni.shape),'spacing_mm':[float(x) for x in nib.affines.voxel_sizes(ni.affine)],'axis_codes':list(nib.aff2axcodes(ni.affine)),'affine_error':err,'raw_values':values,'raw_wt':int(label.sum()),'cropped_wt':cropped,'resized_wt':int(ol.sum()),'geometry':geo,'image_sha256':arrsha(oi),'label_sha256':arrsha(ol)}
def worker(task):
 slot,row,sources=task;i,l,d=prepare(row,sources);vox=128**3
 if os.pwrite(CFG['ifd'],i.tobytes(),slot*vox*4)!=vox*4 or os.pwrite(CFG['lfd'],l.tobytes(),slot*vox)!=vox:raise IOError('short')
 return slot,row['case_id'],row['subject_id'],d
def probe():
 if p.FIRST_PROBE.exists():raise FileExistsError(p.FIRST_PROBE)
 rows=p.cases();sources=smap();torch.set_num_threads(1);import SimpleITK as sitk;sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1);first=rows[0]['case_id'];t1ce_binary._case_id=lambda v:v if v==first else ORIG(v);i,l,d=prepare(rows[0],sources);p.write_new(p.FIRST_PROBE,{'schema':'aflow.waw_postop_only31.first_probe','created_at_utc':datetime.now(timezone.utc).isoformat(),'status':'passed','case_id':first,**d,'output_image_shape':list(i.shape),'output_label_values':sorted(int(x) for x in np.unique(l)),'model_inference':False,'other_cases_preprocessed':False,'runner_sha256':p.sha(Path(__file__))});print(json.dumps(json.load(open(p.FIRST_PROBE)),indent=2))
def build(workers):
 rows=p.cases();sources=smap();probe=json.load(open(p.FIRST_PROBE));needed=p.CASES*128**3*5
 if probe['status']!='passed' or p.CACHE.exists() or p.CACHE_RECEIPT.exists():raise ValueError('state')
 if shutil.disk_usage(p.CACHE.parent).free<needed+1024**3:raise RuntimeError('disk')
 p.CACHE.mkdir();images=p.CACHE/'images.f32';labels=p.CACHE/'labels.u8'
 for path,size in ((images,needed*4//5),(labels,needed//5)):
  with path.open('xb') as f:f.truncate(size)
 cfg={'allowed':[r['case_id'] for r in rows],'images':str(images),'labels':str(labels)};start=time.monotonic();out=[]
 with mp.get_context('spawn').Pool(workers,initializer=init,initargs=(cfg,)) as pool:
  for x in pool.imap_unordered(worker,[(i,r,sources) for i,r in enumerate(rows)],chunksize=1):out.append(x);print(f'cached WAW postop {len(out)}/{p.CASES}',flush=True) if len(out)%5==0 or len(out)==p.CASES else None
 out.sort();details={c:d for _,c,_,d in out};im=np.memmap(images,dtype=np.float32,mode='r',shape=(p.CASES,128,128,128));la=np.memmap(labels,dtype=np.uint8,mode='r',shape=(p.CASES,128,128,128));fd=details[rows[0]['case_id']];gates={'all_finite_binary_nonempty':all(np.isfinite(im[i]).all() and np.all((la[i]==0)|(la[i]==1)) and la[i].any() for i in range(p.CASES)),'first_probe_image_exact':fd['image_sha256']==probe['image_sha256'],'first_probe_label_exact':fd['label_sha256']==probe['label_sha256'],'no_raw_wt_crop_loss':all(x['raw_wt']==x['cropped_wt'] for x in details.values()),'all_resized_wt_nonempty':all(x['resized_wt']>0 for x in details.values())}
 if not all(gates.values()):raise RuntimeError(str(gates))
 index={'schema':'aflow.waw_postop_only31.cache','schema_version':1,'created_at_utc':datetime.now(timezone.utc).isoformat(),'split':'public_waw_postop_only31','cases':p.CASES,'subjects':p.CASES,'case_ids':[r['case_id'] for r in rows],'subject_ids':[r['subject_id'] for r in rows],'images_sha256':p.sha(images),'labels_sha256':p.sha(labels),'model_shape':[128]*3,'details':details,'target':'postop WT raw labels1/2/3 exclude RC4','preprocessing':{'canonical_ras':True,'n4_shrink':2,'iterations':[50]*4,'threads':1,'image_only_crop':True,'positive_zscore':True,'resolution':128},'gates':gates,'cohort_sha256':p.COHORT_SHA,'source_receipt_sha256':p.SOURCE_RECEIPT_SHA,'first_probe_sha256':p.sha(p.FIRST_PROBE),'builder_sha256':p.sha(Path(__file__)),'elapsed_seconds':time.monotonic()-start,'model_inference':False};p.write_new(p.CACHE/'index.json',index);receipt={'schema':'aflow.waw_postop_only31.cache_receipt','status':'passed','all_gates_passed':True,'cases':p.CASES,'index_sha256':p.sha(p.CACHE/'index.json'),'images_sha256':index['images_sha256'],'labels_sha256':index['labels_sha256'],'gates':gates,'elapsed_seconds':index['elapsed_seconds'],'model_inference':False};p.write_new(p.CACHE_RECEIPT,receipt);print(json.dumps(receipt,indent=2))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=('probe','build'));ap.add_argument('--workers',type=int,default=16);a=ap.parse_args();p.docs();probe() if a.action=='probe' else build(a.workers)
if __name__=='__main__':main()
