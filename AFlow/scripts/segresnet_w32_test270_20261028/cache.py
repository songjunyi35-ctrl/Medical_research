"""One N4 call per case using historical native_arrays, emitting both historical grids."""
import multiprocessing as mp,time,traceback
import numpy as np
from scripts.segresnet_w32_test270_20261028.common import *
from scripts.build_segresnet_native_res_test_cache import native_arrays
CFG={}
def init(config):
 import torch,SimpleITK as sitk
 torch.set_num_threads(1);sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1)
 for key,path in config.items():CFG[key]=os.open(path,os.O_WRONLY)
def worker(task):
 slot,row,reference=task
 try:
  x,y,cx,cy,g=native_arrays(row)
  for key in ['origin','extent','native_shape']:assert g[key]==reference[key],(row['case_id'],key)
  px,off=pad_to_canvas(x);py,off2=pad_to_canvas(y);assert off==off2
  for key,a in [('native_image',px),('native_label',py),('canon_image',cx),('canon_label',cy)]:
   raw=a.tobytes();assert os.pwrite(CFG[key],raw,slot*len(raw))==len(raw)
  g.update(canvas_offset=off,canonical_geometry_exact=True)
  return row['case_id'],g,None
 except BaseException as e:return row['case_id'],None,repr(e)+'\n'+traceback.format_exc()
def main():
 verify();validate_manifests();TMP.mkdir(parents=True,exist_ok=True)
 if CACHE.exists() or CANON.exists():raise RuntimeError('existing payload attempt; refuse overwrite')
 if shutil.disk_usage('/tmp').free<20*2**30:raise RuntimeError('scratch reserve below20GiB')
 write_new(R/'CACHE_PIXEL_PREACCESS.json',{'at':now(),'test_pixels_access':True,'candidate_model_forward':False,'scope':'single-N4 reproduce historical canonical and native payloads'})
 for d in [CACHE,CANON]:d.mkdir()
 config={}
 for directory,shape,prefix in [(CACHE,CANVAS,'native'),(CANON,(128,128,128),'canon')]:
  for name,width,kind in [('images.f32',4,'image'),('labels.u8',1,'label')]:
   p=directory/name
   with p.open('xb') as f:f.truncate(270*int(np.prod(shape))*width)
   config[prefix+'_'+kind]=str(p)
 canon=read(CANON_INDEX);tasks=[(i,d,canon['crop_geometry'][d['case_id']]) for i,d in enumerate(rows('test'))]
 started=time.monotonic();geometry={}
 with mp.get_context('spawn').Pool(23,initializer=init,initargs=(config,)) as pool:
  for done,(cid,g,error) in enumerate(pool.imap_unordered(worker,tasks,chunksize=1),1):
   deadline()
   if error:raise RuntimeError(error)
   geometry[cid]=g
   if done%15==0 or done==270:status('building_cache_single_N4',completed=done,total=270,seconds=time.monotonic()-started,candidate_model_forward=False)
 actual={key:sha(Path(path)) for key,path in config.items()};ref=read(NATIVE_REF)
 assert actual['canon_image']==canon['images_sha256'] and actual['canon_label']==canon['labels_sha256'],'canonical historical SHA mismatch'
 assert actual['native_image']==ref['images_sha256'] and actual['native_label']==ref['labels_sha256'],'native historical SHA mismatch'
 assert sum(g['positives_outside_crop'] for g in geometry.values())==ref['positives_outside_crop_total']
 index={'schema':SCHEMA+'.test','schema_version':1,'split':'test','test_access':True,'canvas':list(CANVAS),'count':270,'case_ids':[d['case_id'] for d in rows('test')],'geometry':geometry,'images_sha256':actual['native_image'],'labels_sha256':actual['native_label'],'preprocessing':{'uses_gt_for_localization':False,'n4_shrink_factor':2,'normalization':'historical native zscore','padding':'centered zero'},'historical_payload_exact':True}
 write(CACHE/'index.json',index)
 write(R/'CACHE_READY.json',{'at':now(),'status':'passed','historical_payload_exact':True,'all_four_payloads_exact':True,'payload_sha256':actual,'index_sha256':sha(CACHE/'index.json'),'count':270,'subjects':119,'seconds':time.monotonic()-started,'processes':23,'N4_evaluations_per_case':1,'pixels_outside_crop':ref['positives_outside_crop_total']})
if __name__=='__main__':
 try:main()
 except BaseException as e:write(R/'CACHE_FAILURE.json',{'at':now(),'error':repr(e),'traceback':traceback.format_exc()});raise
