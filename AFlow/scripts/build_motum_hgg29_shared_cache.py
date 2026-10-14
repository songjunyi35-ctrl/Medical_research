#!/usr/bin/env python3
"""Build MOTUM-HGG29 cache with the verified shared discrete orientation."""
from __future__ import annotations
import argparse,hashlib,json,multiprocessing as mp,os,shutil,time
from datetime import datetime,timezone
from pathlib import Path
import nibabel as nib,numpy as np,torch
from sammed3d import t1ce_binary
from scripts.build_no_n4_clean_cache import arrays
from scripts import motum_hgg29_shared_panel_protocol as p
CFG={};ORIG=t1ce_binary._case_id
def init(cfg):
 CFG.update(cfg);CFG['allowed']=set(cfg['allowed']);torch.set_num_threads(1);import SimpleITK as sitk;sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1);t1ce_binary._case_id=lambda v:v if v in CFG['allowed'] else ORIG(v);CFG['ifd']=os.open(cfg['images'],os.O_WRONLY);CFG['lfd']=os.open(cfg['labels'],os.O_WRONLY)
def arrsha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def prepare(row):
 paths=[Path(row[k]) for k in ('image_path','flair_mask_path','t1ce_mask_path')];expected=[row['image_sha256'],row['flair_mask_sha256'],row['t1ce_mask_sha256']]
 for path,digest in zip(paths,expected):p.checked(path,digest)
 ims=[nib.load(path) for path in paths]
 if not ims[0].shape==ims[1].shape==ims[2].shape:raise ValueError('shape')
 native=max(float(np.max(np.abs(ims[0].affine-x.affine))) for x in ims[1:])
 if native>5e-7:raise ValueError('native affine')
 ori=[nib.orientations.io_orientation(x.affine) for x in ims]
 if not all(np.array_equal(ori[0],x) for x in ori[1:]):raise ValueError('orientation')
 transform=nib.orientations.ornt_transform(ori[0],nib.orientations.axcodes2ornt(('R','A','S')));raw=[np.asanyarray(x.dataobj) for x in ims];shared=[np.ascontiguousarray(nib.orientations.apply_orientation(x,transform)) for x in raw];ind=[np.ascontiguousarray(np.asanyarray(nib.as_closest_canonical(x).dataobj)) for x in ims]
 if not all(np.array_equal(a,b) for a,b in zip(shared,ind)):raise ValueError('array parity')
 image,fa,ta=shared;fr=np.rint(fa);tr=np.rint(ta);fv=sorted(int(x) for x in np.unique(fr));tv=sorted(int(x) for x in np.unique(tr));label=np.ascontiguousarray((fr>0)|(tr>0),dtype=np.uint8)
 if not np.isfinite(image).all() or not (image>0).any() or not set(fv)<={0,1} or not set(tv)<={0,1,2} or not label.any():raise ValueError('domain')
 affine=ims[0].affine@nib.orientations.inv_ornt_aff(transform,ims[0].shape);oi,ol,geo=arrays(row['case_id'],image,label,nib.affines.voxel_sizes(affine),use_n4=True);crop=tuple(slice(o,o+e) for o,e in zip(geo['origin'],geo['extent']));cropped=int(label[crop].sum())
 if cropped!=int(label.sum()) or not ol.any():raise ValueError('WT crop loss')
 return oi,ol,{'native_shape':list(ims[0].shape),'shared_shape':list(image.shape),'spacing_mm':[float(x) for x in nib.affines.voxel_sizes(affine)],'native_affine_error':native,'transform':transform.tolist(),'shared_affine_sha256':arrsha(affine.astype(np.float64)),'flair_values':fv,'t1ce_values':tv,'raw_wt':int(label.sum()),'cropped_wt':cropped,'resized_wt':int(ol.sum()),'geometry':geo,'image_sha256':arrsha(oi),'label_sha256':arrsha(ol)}
def worker(task):
 slot,row=task;i,l,d=prepare(row);vox=128**3
 if os.pwrite(CFG['ifd'],i.tobytes(),slot*vox*4)!=vox*4 or os.pwrite(CFG['lfd'],l.tobytes(),slot*vox)!=vox:raise IOError('short')
 return slot,row['case_id'],row['subject_id'],d
def probe():
 if p.FIRST_PROBE.exists():raise FileExistsError(p.FIRST_PROBE)
 rows=p.cases();torch.set_num_threads(1);import SimpleITK as sitk;sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1);first=rows[0]['case_id'];t1ce_binary._case_id=lambda v:v if v==first else ORIG(v);i,l,d=prepare(rows[0]);p.write_new(p.FIRST_PROBE,{'schema':'aflow.motum_hgg29.shared.first_probe','created_at_utc':datetime.now(timezone.utc).isoformat(),'status':'passed','case_id':first,**d,'output_image_shape':list(i.shape),'output_label_values':sorted(int(x) for x in np.unique(l)),'model_inference':False,'other_cases_preprocessed':False,'runner_sha256':p.sha(Path(__file__))});print(json.dumps(json.load(open(p.FIRST_PROBE)),indent=2))
def build(workers):
 rows=p.cases();probe=json.load(open(p.FIRST_PROBE))
 if probe['status']!='passed' or probe['case_id']!=rows[0]['case_id']:raise ValueError('probe')
 if p.CACHE.exists() or p.CACHE_RECEIPT.exists():raise FileExistsError('cache')
 needed=p.CASES*128**3*5
 if shutil.disk_usage(p.CACHE.parent).free<needed+1024**3:raise RuntimeError('disk')
 p.CACHE.mkdir();images=p.CACHE/'images.f32';labels=p.CACHE/'labels.u8'
 for path,size in ((images,needed*4//5),(labels,needed//5)):
  with path.open('xb') as f:f.truncate(size)
 cfg={'allowed':[r['case_id'] for r in rows],'images':str(images),'labels':str(labels)};start=time.monotonic();out=[]
 with mp.get_context('spawn').Pool(workers,initializer=init,initargs=(cfg,)) as pool:
  for x in pool.imap_unordered(worker,enumerate(rows),chunksize=1):out.append(x);print(f'cached shared MOTUM {len(out)}/{p.CASES}',flush=True) if len(out)%5==0 or len(out)==p.CASES else None
 out.sort();details={case:d for _,case,_,d in out};im=np.memmap(images,dtype=np.float32,mode='r',shape=(p.CASES,128,128,128));la=np.memmap(labels,dtype=np.uint8,mode='r',shape=(p.CASES,128,128,128));fd=details[rows[0]['case_id']];gates={'all_finite_binary_nonempty':all(np.isfinite(im[i]).all() and np.all((la[i]==0)|(la[i]==1)) and la[i].any() for i in range(p.CASES)),'first_probe_image_exact':fd['image_sha256']==probe['image_sha256'],'first_probe_label_exact':fd['label_sha256']==probe['label_sha256'],'no_raw_wt_crop_loss':all(x['raw_wt']==x['cropped_wt'] for x in details.values()),'all_resized_wt_nonempty':all(x['resized_wt']>0 for x in details.values()),'native_affine_strict':all(x['native_affine_error']<=5e-7 for x in details.values())}
 if not all(gates.values()):raise RuntimeError(str(gates))
 index={'schema':'aflow.motum_hgg29.shared.cache','schema_version':1,'created_at_utc':datetime.now(timezone.utc).isoformat(),'split':'public_external_motum_hgg29_shared_orientation','cases':p.CASES,'subjects':p.CASES,'case_ids':[r['case_id'] for r in rows],'subject_ids':[r['subject_id'] for r in rows],'images_sha256':p.sha(images),'labels_sha256':p.sha(labels),'model_shape':[128]*3,'details':details,'target':'WT=positive union of two semantic masks','preprocessing':{'shared_orientation_bridge_sha256':p.BRIDGE_SHA,'canonical_ras_shared_image_affine':True,'interpolation_during_orientation':False,'n4_shrink':2,'iterations':[50]*4,'simpleitk_threads':1,'image_only_crop':True,'source_positive_zscore':True,'resolution':128},'gates':gates,'cohort_sha256':p.COHORT_SHA,'first_probe_sha256':p.sha(p.FIRST_PROBE),'builder_sha256':p.sha(Path(__file__)),'elapsed_seconds':time.monotonic()-start,'model_inference':False};p.write_new(p.CACHE/'index.json',index);receipt={'schema':'aflow.motum_hgg29.shared.cache_receipt','status':'passed','all_gates_passed':True,'cases':p.CASES,'index_sha256':p.sha(p.CACHE/'index.json'),'images_sha256':index['images_sha256'],'labels_sha256':index['labels_sha256'],'gates':gates,'elapsed_seconds':index['elapsed_seconds'],'model_inference':False};p.write_new(p.CACHE_RECEIPT,receipt);print(json.dumps(receipt,indent=2))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=('probe','build'));ap.add_argument('--workers',type=int,default=16);a=ap.parse_args();p.docs();probe() if a.action=='probe' else build(a.workers)
if __name__=='__main__':main()
