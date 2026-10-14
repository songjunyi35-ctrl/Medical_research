"""Label-free T1ce NIfTI -> WT mask using the frozen campaign deployment manifest."""
import sys,argparse,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.t1ce_10h_common import *

@torch.no_grad()
def infer_views(manifest,image128,native_canvas,extent,offset,device):
 numeric();outputs=[]
 for c in manifest['components']:
  path=ROOT/c['checkpoint_path'];assert sha(path)==c['checkpoint_sha256'];m=model_for(device,path);m.eval()
  with torch.autocast('cuda',dtype=torch.bfloat16):
   if c['mode']=='128':z=predict(m,image128.to(device),'128');crop=F.interpolate(z,size=extent,mode='trilinear',align_corners=False)[0,0]
   else:
    z=predict(m,native_canvas.to(device),c['mode']);crop=z[0,0][tuple(slice(o,o+n) for o,n in zip(offset,extent))]
  outputs.append(crop.cpu());del m;torch.cuda.empty_cache()
 return sum(outputs)/len(outputs)

def nifti(image_path,output_path,manifest_path):
 import nibabel as nib
 from sammed3d.t1ce_binary import prepare_t1ce_inference_array,_normalize_nonzero_t1ce
 from sammed3d.t1ce_wt_training import _load_canonical_nifti
 from scripts.build_t1ce_wt_cache import n4_correct
 manifest=read(manifest_path);img=_load_canonical_nifti(Path(image_path),'T1ce only')
 spacing=nib.affines.voxel_sizes(img.affine)
 if not np.allclose(spacing,[1,1,1],atol=1e-3):raise ValueError('this frozen deployment expects1mm preprocessed T1ce; no automatic resampling')
 x=np.ascontiguousarray(np.asarray(img.dataobj,dtype=np.float32));corrected=n4_correct(x,2);coords=np.nonzero(corrected>0)
 if not all(len(c) for c in coords):raise ValueError('empty brain support')
 origin=[int(c.min()) for c in coords];end=[int(c.max())+1 for c in coords];sl=tuple(slice(a,b) for a,b in zip(origin,end));crop=corrected[sl];extent=list(crop.shape)
 view=prepare_t1ce_inference_array('image_only',crop,spacing,output_shape=(128,128,128));normalized,_,_=_normalize_nonzero_t1ce(crop)
 shape=(160,208,168)
 if any(e>s for e,s in zip(extent,shape)):raise ValueError('image-derived crop exceeds frozen canvas; cannot silently truncate')
 offset=[(s-e)//2 for s,e in zip(shape,extent)];native=torch.zeros((1,1,*shape));native[(0,0,*[slice(o,o+e) for o,e in zip(offset,extent)])]=normalized
 z=infer_views(manifest,view.image.unsqueeze(0),native,extent,offset,torch.device('cuda:0'));threshold=math.log(manifest['threshold']/(1-manifest['threshold']))
 mask=np.zeros(x.shape,dtype='uint8');mask[sl]=(z.numpy()>threshold).astype('uint8');nib.save(nib.Nifti1Image(mask,img.affine),str(output_path))

def check():
 manifest=read(OUT/'deployment/manifest.json');idx=read(SCRATCH/'test_native/index.json');ds128=GridDataset('test');dsn=GridDataset('test',native=True)
 cases=[];tick=time.monotonic();device=torch.device('cuda:0');torch.set_num_threads(2)
 # Predeclared first/last two scans, identity/reload check only; no new selection.
 write(R/'RELOAD_PREACCESS.json',{'utc':now(),'purpose':'fixed deployment consistency only','indices':[0,1,268,269],'selection_or_tuning':False,'test_tuned_campaign':True})
 for i in [0,1,268,269]:
  a=ds128[i];n=dsn[i];cid=a['case_id'];assert cid==n['case_id'];geo=idx['geometry'][cid]
  z=infer_views(manifest,a['image'][None],n['image'][None],geo['extent'],geo['canvas_offset'],device)
  r=read(ROOT/manifest['source_report']);component_ids=r['components'];cached=[]
  for ident in component_ids:
   path=SCRATCH/'logits'/ident/(cid+'.npy')
   if not path.exists():path=SCRATCH/'imported_logits'/ident/(cid+'.npy')
   cached.append(torch.from_numpy(np.load(path,allow_pickle=False)))
  expected=sum(cached)/len(cached);maxdiff=float((z-expected).abs().max());threshold=math.log(manifest['threshold']/(1-manifest['threshold']))
  disagreement=float(((z>threshold)!=(expected>threshold)).float().mean())
  # Batch1 versus frozen batch2 can differ slightly in bf16 kernels; measure and disclose.
  cases.append({'case_id':cid,'max_logit_error':maxdiff,'mask_disagreement_fraction':disagreement,'source_checkpoint_reload_strict':True})
  assert disagreement<.001,('deployment mismatch',cid,disagreement)
 write(R/'DEPLOYMENT_CHECK.json',{'utc':now(),'status':'passed','cases':cases,'seconds_for_four_scans':time.monotonic()-tick,'precision':'bf16, batch1 deployment versus batch2 cached scoring','input':'T1ce only; no label accepted by infer_views','check_not_used_for_selection':True,'manifest_sha256':sha(OUT/'deployment/manifest.json')})

if __name__=='__main__':
 if len(sys.argv)>1 and sys.argv[1]=='check':check()
 else:
  p=argparse.ArgumentParser();p.add_argument('--image',required=True);p.add_argument('--output',required=True);p.add_argument('--manifest',default=str(OUT/'deployment/manifest.json'));a=p.parse_args();nifti(a.image,a.output,a.manifest)
