#!/usr/bin/env python3
"""Build canonical-N4 cache for CPTAC33 with a required first-case probe."""
from __future__ import annotations
import argparse,hashlib,json,multiprocessing as mp,os,shutil,time
from datetime import datetime,timezone
from pathlib import Path
import nibabel as nib,numpy as np,torch
from sammed3d import t1ce_binary
from scripts.build_no_n4_clean_cache import arrays
from scripts import cptac33_protocol as p
CFG={};ORIG=t1ce_binary._case_id
def init(cfg):
 CFG.update(cfg);CFG['allowed']=set(cfg['allowed']);torch.set_num_threads(1);import SimpleITK as sitk;sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1);t1ce_binary._case_id=lambda v:v if v in CFG['allowed'] else ORIG(v);CFG['ifd']=os.open(cfg['images'],os.O_WRONLY);CFG['lfd']=os.open(cfg['labels'],os.O_WRONLY)
def arrsha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def prepare(row,source):
 bid=row['brats2021_id'];root=p.SOURCE/'copy_a'/bid;ip=root/f'{bid}_t1ce.nii.gz';lp=root/f'{bid}_seg.nii.gz';irow=source[f'{bid}/{bid}_t1ce.nii.gz'];lrow=source[f'{bid}/{bid}_seg.nii.gz']
 if p.sha(ip)!=irow['sha256'] or p.sha(lp)!=lrow['sha256']:raise ValueError('source SHA')
 ni=nib.as_closest_canonical(nib.load(str(ip),mmap=True));nl=nib.as_closest_canonical(nib.load(str(lp),mmap=True));aff=float(np.max(np.abs(ni.affine-nl.affine)))
 if ni.shape!=nl.shape or aff>1e-5:raise ValueError('geometry')
 image=np.ascontiguousarray(np.asanyarray(ni.dataobj));raw=np.ascontiguousarray(np.asanyarray(nl.dataobj));rounded=np.rint(raw);values=sorted(int(x) for x in np.unique(rounded))
 if not np.isfinite(image).all() or not (image>0).any() or not np.isfinite(raw).all() or not np.allclose(raw,rounded,rtol=0,atol=1e-5) or not set(values)<={0,1,2,4}:raise ValueError('array/domain '+str(values))
 binary=np.ascontiguousarray(rounded>0,dtype=np.uint8)
 if not binary.any() or not (binary==0).any():raise ValueError('WT')
 oi,ol,geo=arrays(bid,image,binary,nib.affines.voxel_sizes(ni.affine),use_n4=True)
 return oi,ol,{'native_shape':list(ni.shape),'spacing':[float(x) for x in nib.affines.voxel_sizes(ni.affine)],'axis_codes':list(nib.aff2axcodes(ni.affine)),'affine_error':aff,'raw_values':values,'raw_wt':int(binary.sum()),'resized_wt':int(ol.sum()),'geometry':geo,'image_sha':arrsha(oi),'label_sha':arrsha(ol)}
def worker(task):
 slot,row,source=task;i,l,d=prepare(row,source);vox=128**3
 if os.pwrite(CFG['ifd'],i.tobytes(),slot*vox*4)!=vox*4 or os.pwrite(CFG['lfd'],l.tobytes(),slot*vox)!=vox:raise IOError('short')
 return slot,row['brats2021_id'],d
def probe():
 if p.FIRST_PROBE.exists():raise FileExistsError(p.FIRST_PROBE)
 rows=p.rows();source={x['relative_path']:x for x in json.load(open(p.SOURCE_RECEIPT))['comparisons']};torch.set_num_threads(1);import SimpleITK as sitk;sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1);t1ce_binary._case_id=lambda v:v if v==rows[0]['brats2021_id'] else ORIG(v);i,l,d=prepare(rows[0],source);r={'schema':'aflow.cptac33.first_probe','created_at_utc':datetime.now(timezone.utc).isoformat(),'status':'passed','case_id':rows[0]['brats2021_id'],**d,'output_image_shape':list(i.shape),'output_label_values':sorted(int(x) for x in np.unique(l)),'model_inference':False,'other_cases':False,'runner_sha256':p.sha(Path(__file__))};p.write_new(p.FIRST_PROBE,r);print(json.dumps(r,indent=2))
def build(workers):
 rows=p.rows();source={x['relative_path']:x for x in json.load(open(p.SOURCE_RECEIPT))['comparisons']};probe=json.load(open(p.FIRST_PROBE))
 if probe.get('status')!='passed' or probe['case_id']!=rows[0]['brats2021_id']:raise ValueError('probe')
 if p.CACHE.exists() or p.CACHE_RECEIPT.exists():raise FileExistsError('cache')
 needed=33*128**3*5
 if shutil.disk_usage('/tmp').free<needed+5*1024**3:raise RuntimeError('disk')
 p.CACHE.mkdir();images=p.CACHE/'images.f32';labels=p.CACHE/'labels.u8'
 for path,size in ((images,needed*4//5),(labels,needed//5)):
  with path.open('xb') as f:f.truncate(size)
 cfg={'allowed':[x['brats2021_id'] for x in rows],'images':str(images),'labels':str(labels)};start=time.monotonic();values=[]
 with mp.get_context('spawn').Pool(workers,initializer=init,initargs=(cfg,)) as pool:
  for x in pool.imap_unordered(worker,[(i,r,source) for i,r in enumerate(rows)],chunksize=1):values.append(x);print(f"cached CPTAC {len(values)}/33",flush=True) if len(values)%5==0 or len(values)==33 else None
 values.sort();details={case:d for _,case,d in values};im=np.memmap(images,dtype=np.float32,mode='r',shape=(33,128,128,128));la=np.memmap(labels,dtype=np.uint8,mode='r',shape=(33,128,128,128));gates={'all_valid':all(np.isfinite(im[i]).all() and np.all((la[i]==0)|(la[i]==1)) and la[i].any() for i in range(33)),'probe_image_exact':details[rows[0]['brats2021_id']]['image_sha']==probe['image_sha'],'probe_label_exact':details[rows[0]['brats2021_id']]['label_sha']==probe['label_sha'],'all_wt':all(x['raw_wt']>0 and x['resized_wt']>0 for x in details.values())}
 if not all(gates.values()):raise RuntimeError('cache gates')
 index={'schema':'aflow.cptac33.cache','schema_version':1,'created_at_utc':datetime.now(timezone.utc).isoformat(),'split':'public_brats2021_training_external_to_project','cases':33,'subjects':33,'case_ids':[x['brats2021_id'] for x in rows],'subject_ids':[x['cptac_patient_id'] for x in rows],'images_sha256':p.sha(images),'labels_sha256':p.sha(labels),'model_shape':[128]*3,'details':details,'raw_union':sorted({v for x in details.values() for v in x['raw_values']}),'target':'preoperative WT=all positive','preprocessing':{'canonical_ras':True,'n4_shrink':2,'simpleitk_threads':1,'image_only_crop':True,'source_nonzero_zscore':True,'resolution':128},'gates':gates,'source_receipt_sha256':p.sha(p.SOURCE_RECEIPT),'probe_sha256':p.sha(p.FIRST_PROBE),'builder_sha256':p.sha(Path(__file__)),'elapsed_seconds':time.monotonic()-start,'model_inference':False};p.write_new(p.CACHE/'index.json',index);receipt={'schema':'aflow.cptac33.cache_receipt','status':'passed','all_gates_passed':True,'index_sha256':p.sha(p.CACHE/'index.json'),'images_sha256':index['images_sha256'],'labels_sha256':index['labels_sha256'],'raw_union':index['raw_union'],'elapsed_seconds':index['elapsed_seconds'],'model_inference':False};p.write_new(p.CACHE_RECEIPT,receipt);print(json.dumps(receipt,indent=2))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=('probe','build'));ap.add_argument('--workers',type=int,default=8);a=ap.parse_args();p.docs();r=json.load(open(p.SOURCE_RECEIPT));
 if r.get('status')!='passed' or r.get('exact_files')!=66 or r.get('arrays') is not False:raise ValueError('source receipt')
 probe() if a.action=='probe' else build(a.workers)
if __name__=='__main__':main()
