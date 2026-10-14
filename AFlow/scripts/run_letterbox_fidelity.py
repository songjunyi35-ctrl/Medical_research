"""Prospective, fixed fit-only comparison of two label rasterizations."""
from collections import defaultdict
from datetime import datetime,timezone
import json
import multiprocessing as mp
from pathlib import Path
import statistics
import time
import traceback
import nibabel as nib
import numpy as np
from scipy import ndimage
import torch
from sammed3d.letterbox_fidelity import layout,encode,decode
from sammed3d.spatial_roundtrip_audit import resize,measure
from scripts import structured_background_protocol as p

REC=p.ROOT/'archive/research_records/2026-09-10_letterbox_fidelity'
CFG={}


def init(config):
    torch.set_num_threads(1);CFG.update(config)
    CFG['labels']=np.memmap(p.CACHE/'labels.u8',mode='r',dtype=np.uint8,shape=(972,128,128,128))


def worker(task):
    row,geo,slot,digest=task
    p.checked(Path(row['label_path']),digest)
    source=nib.as_closest_canonical(nib.load(row['label_path']))
    array=np.asarray(source.dataobj)
    assert list(array.shape)==geo['native_shape'] and nib.aff2axcodes(source.affine)==('R','A','S')
    assert np.isfinite(array).all() and np.array_equal(array,np.rint(array)) and set(np.unique(array))<={0,1,2,3,4}
    truth=np.isin(array,[1,2,3]);box=tuple(slice(o,o+e) for o,e in zip(geo['origin'],geo['extent']))
    crop=truth[box];old=resize(crop,(128,128,128),'nearest-exact')
    assert np.array_equal(old,CFG['labels'][slot]),'old hard cache parity'
    new=encode(crop);small,slices=layout(crop.shape)
    assert np.array_equal(new[slices],resize(crop,small,'nearest-exact'))
    ids,count=ndimage.label(truth,structure=np.ones((3,3,3),dtype=bool))
    sizes=np.bincount(ids.ravel());inside=np.bincount(ids[box].ravel(),minlength=count+1)
    old_ids=resize(ids[box],(128,128,128),'nearest-exact').astype(np.int32)
    new_ids=encode(ids[box]).astype(np.int32)
    assert np.array_equal(new_ids>0,new>0) and np.array_equal(old_ids>0,old>0)
    old_counts=np.bincount(old_ids.ravel(),minlength=count+1);new_counts=np.bincount(new_ids.ravel(),minlength=count+1)
    details=[{'id':k,'native_voxels':int(sizes[k]),'crop_voxels':int(inside[k]),'old_sampled_voxels':int(old_counts[k]),'letterbox_sampled_voxels':int(new_counts[k])} for k in range(1,count+1)]
    measures={}
    for name,restored in [('direct128',resize(old,crop.shape,'nearest-exact')),('letterbox128',decode(new,crop.shape))]:
        full=np.zeros(truth.shape,dtype=bool);full[box]=restored>.5
        measures[name]=measure(truth,full,(1,1,1),with_surfaces=False)
    return {'case_id':row['case_id'],'subject_id':row['subject_id'],'source_label_sha256':digest,'crop_origin':geo['origin'],'crop_extent':geo['extent'],'letterbox_content_shape':list(small),'padding_fraction':1-np.prod(small)/128**3,'crop_lost_voxels':int(truth.sum()-crop.sum()),'old_cache_byte_exact':True,'native_components':count,'components':details,'metrics':measures,'letterbox_empty_coarse':not bool(new.any())}


def main():
    if (REC/'PREACCESS.json').exists():raise FileExistsError('already started')
    fit,_=p.check_roles();p.checked(p.CACHE/'index.json',p.INDEX_SHA)
    index=json.loads((p.CACHE/'index.json').read_text());mapping={c:i for i,c in enumerate(index['case_ids'])}
    receipt=json.loads((p.RECORD/'AUX_CACHE_RECEIPT.json').read_text());p.checked(p.CACHE/'labels.u8',receipt['source_labels_sha256'])
    hashes={r['case_id']:r['source_label_sha256'] for r in receipt['details']}
    code=['scripts/run_letterbox_fidelity.py','sammed3d/letterbox_fidelity.py','sammed3d/spatial_roundtrip_audit.py','tests/test_letterbox_fidelity.py','scripts/finalize_letterbox_fidelity.py']
    p.write_new(REC/'PREACCESS.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'fit_sha256':p.FIT_SHA,'cache_index_sha256':p.INDEX_SHA,'labels_sha256':receipt['source_labels_sha256'],'source_label_hashes':hashes,'files':{n:p.sha(p.ROOT/n) for n in code},'idea_sha256':p.sha(REC/'IDEA.md'),'new_fit_label_reads_at_write':0,'new_forwards':0,'development_access':False,'test_access':False,'workers':4})
    tasks=[(r,index['crop_geometry'][r['case_id']],mapping[r['case_id']],hashes[r['case_id']]) for r in fit]
    started=time.monotonic();rows=[]
    try:
        with (REC/'RAW_CASES.jsonl').open('x') as output,mp.get_context('spawn').Pool(4,initializer=init,initargs=({},)) as pool:
            for item in pool.imap(worker,tasks):
                rows.append(item);output.write(json.dumps(item)+'\n');output.flush()
                if len(rows)%100==0:print(f'fit label comparison {len(rows)}/684',flush=True)
        assert len(rows)==684
        p.write_new(REC/'RUN_DONE.json',{'status':'completed','created_utc':datetime.now(timezone.utc).isoformat(),'seconds':time.monotonic()-started,'scans':684,'raw_sha256':p.sha(REC/'RAW_CASES.jsonl'),'preaccess_sha256':p.sha(REC/'PREACCESS.json'),'model_forwards':0,'development_access':False,'test_access':False})
    except BaseException as error:
        p.write_new(REC/'FAILURE.json',{'error':repr(error),'traceback':traceback.format_exc(),'saved_rows':len(rows),'unquantified_fit_worker_prefetch':True,'development_access':False,'test_access':False});raise


if __name__=='__main__':main()
