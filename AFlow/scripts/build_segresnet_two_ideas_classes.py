"""Original972 training labels only, with independent nearest-index verification."""
import sys, os, json, time, multiprocessing as mp
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import nibabel as nib
import torch
from sammed3d.subregion_labels import torch_sample, numpy_sample
from scripts.segresnet_two_ideas_common import *
CFG={}


def init():
    torch.set_num_threads(1)
    CFG['labels']=np.memmap(TRAIN_CACHE/'labels.u8',dtype=np.uint8,mode='r',shape=(972,128,128,128))
    CFG['fd']=os.open(CLASS_CACHE/'classes.u8',os.O_WRONLY)


def case(task):
    slot,row,geo,oldslot=task; path=Path(row['seg_path']); digest=sha(path)
    nii=nib.as_closest_canonical(nib.load(path)); raw=np.asarray(nii.dataobj)
    assert list(raw.shape)==geo['native_shape'] and nib.aff2axcodes(nii.affine)==('R','A','S')
    assert np.isfinite(raw).all() and np.array_equal(raw,np.rint(raw)) and set(np.unique(raw))<={0,1,2,3,4}
    assert all(o>=0 and e>0 and o+e<=n for o,e,n in zip(geo['origin'],geo['extent'],raw.shape))
    classes=torch_sample(raw,geo['origin'],geo['extent'])
    independent=numpy_sample(raw,geo['origin'],geo['extent'])
    assert np.array_equal(classes,independent)
    assert np.array_equal(np.isin(classes,[1,2,3]),CFG['labels'][oldslot]>0)
    payload=classes.tobytes(); assert os.pwrite(CFG['fd'],payload,slot*128**3)==len(payload)
    return {'case_id':row['case_id'],'subject_id':row['subject_id'],'source_label_sha256':digest,
            'classes_sha256':hashlib.sha256(payload).hexdigest(),'counts':np.bincount(classes.ravel(),minlength=5).tolist(),
            'independent_classes_exact':True,'canonical_wt_exact':True}


def main():
    verify(); rows=read_rows(MANIFEST['train']); assert len(rows)==972
    if (R/'CLASS_CACHE_PREACCESS.json').exists() or CLASS_CACHE.exists(): raise FileExistsError('class cache already attempted')
    import shutil
    assert shutil.disk_usage('/tmp').free>4*(1<<30)
    index=json.loads((TRAIN_CACHE/'index.json').read_text()); lookup={c:i for i,c in enumerate(index['case_ids'])}
    assert set(lookup)=={r['case_id'] for r in rows}
    write(R/'CLASS_CACHE_PREACCESS.json',{'utc':now(),'role':'original972 training only','freeze_sha256':sha(R/'FREEZE.json'),
                                         'manifest_sha256':sha(MANIFEST['train']),'test_access':False})
    CLASS_CACHE.mkdir(parents=True,exist_ok=False)
    with (CLASS_CACHE/'classes.u8').open('xb') as f:f.truncate(972*128**3)
    tasks=[(i,row,index['crop_geometry'][row['case_id']],lookup[row['case_id']]) for i,row in enumerate(rows)]
    started=time.monotonic(); results=[]
    with (R/'CLASS_CACHE_RAW.jsonl').open('x') as log,mp.get_context('spawn').Pool(8,initializer=init) as pool:
        for row in pool.imap(case,tasks,chunksize=1):
            results.append(row);log.write(json.dumps(row)+'\n');log.flush()
            if len(results)%100==0:print(json.dumps({'classes_done':len(results),'total':972}),flush=True)
    with (CLASS_CACHE/'classes.u8').open('rb') as f:os.fsync(f.fileno())
    digest=sha(CLASS_CACHE/'classes.u8')
    meta={'count':972,'case_ids':[r['case_id'] for r in rows],'test_access':False,'split':'train','dtype':'uint8',
          'shape':[972,128,128,128],'original_labels':[0,1,2,3,4],'classes_sha256':digest,
          'canonical_index_sha256':sha(TRAIN_CACHE/'index.json')}
    write(CLASS_CACHE/'index.json',meta)
    write(R/'CLASS_CACHE_RECEIPT.json',{'utc':now(),'status':'passed','count':972,'subjects':444,'all_independent_classes_exact':True,
          'all_canonical_wt_exact':True,'cache':str(CLASS_CACHE),'index_sha256':sha(CLASS_CACHE/'index.json'),
          'classes_sha256':digest,'raw_sha256':sha(R/'CLASS_CACHE_RAW.jsonl'),'seconds':time.monotonic()-started,'test_access':False})


if __name__=='__main__':main()
