"""Reproduce historical native test payload; no overwrite or automatic rebuild."""
import sys,time,shutil,multiprocessing as mp
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.followup_test270_common import *
from scripts import build_segresnet_native_res_test_cache as original


def main():
    verify();assert json.loads((S/'CANONICAL_RECEIPT.json').read_text())['status']=='passed'
    write_new(S/'NATIVE_CACHE_PREACCESS.json',{'utc':now(),'test_pixels_access':True,'purpose':'frozen native cache reproduction',
              'freeze_sha256':sha(S/'FREEZE.json'),'candidate_model_forward':False})
    if TEST_CACHE.exists():raise RuntimeError('refuse existing native cache; inspect before recovery')
    assert shutil.disk_usage('/tmp').free>270*int(np.prod(TEST_CANVAS))*5+4*(1<<30)
    TEST_CACHE.mkdir(parents=True);rows=read_rows(MANIFEST['test']);canon=json.loads((CANONICAL_TEST/'index.json').read_text())
    assert canon['case_ids']==[x['case_id'] for x in rows]
    n=int(np.prod(TEST_CANVAS))
    for name,width in [('images.f32',4),('labels.u8',1)]:
        with (TEST_CACHE/name).open('xb') as f:f.truncate(270*n*width)
    config={'images':str(TEST_CACHE/'images.f32'),'labels':str(TEST_CACHE/'labels.u8'),'canon_dir':str(CANONICAL_TEST)}
    tasks=[(i,row,canon['crop_geometry'][row['case_id']]) for i,row in enumerate(rows)]
    geometry={};start=time.monotonic()
    with mp.get_context('spawn').Pool(24,initializer=original.init,initargs=(config,)) as pool:
        for k,(slot,cid,g,error) in enumerate(pool.imap_unordered(original.worker,tasks,chunksize=1),1):
            if error:raise RuntimeError(cid+': '+error)
            if not (g['canonical_image_within_tolerance'] and g['canonical_label_exact'] and g['canonical_geometry_exact'] and g['positive_voxels']>0):
                raise RuntimeError(cid+': canonical reproduction failed')
            geometry[cid]=g
            if k%30==0 or k==270:print(json.dumps({'utc':now(),'done':k,'of':270,'errors':0,'seconds':time.monotonic()-start}),flush=True)
    payload={'images_sha256':sha(TEST_CACHE/'images.f32'),'labels_sha256':sha(TEST_CACHE/'labels.u8')}
    old=json.loads((OLD_NATIVE/'CACHE_RECEIPT.json').read_text())
    assert all(payload[k]==old[k] for k in payload),'native payload differs from historical reference cache'
    index={'schema':SCHEMA+'.test','schema_version':1,'split':'test','test_access':True,'canvas':list(TEST_CANVAS),'count':270,
           'case_ids':[r['case_id'] for r in rows],'manifest_sha256':sha(MANIFEST['test']),'geometry':geometry,'image_dtype':'float32','label_dtype':'uint8',
           'preprocessing':{'uses_gt_for_localization':False,'spatial':'braincrop_native_canvas','n4_shrink_factor':2,
                            'normalization':'canonical nonzero population zscore','padding':'centered zero'},**payload,
           'historical_payload_exact':True,'builder_sha256':sha(Path(__file__))}
    write(TEST_CACHE/'index.json',index)
    receipt={'utc':now(),'status':'passed','worker':1104883,'cache':str(TEST_CACHE),'index_sha256':sha(TEST_CACHE/'index.json'),
             **payload,'historical_payload_exact':True,'historical_receipt_sha256':sha(OLD_NATIVE/'CACHE_RECEIPT.json'),
             'count':270,'subjects':119,'canvas':list(TEST_CANVAS),'build_seconds':time.monotonic()-start,
             'bytes':270*n*5,'all_canonical_labels_geometry_exact':True,
             'image_max_abs_diff':max(g['canonical_image_max_abs_diff'] for g in geometry.values()),
             'positives_outside_crop_total':sum(g['positives_outside_crop'] for g in geometry.values())}
    write(S/'NATIVE_CACHE_RECEIPT.json',receipt);write(RECORDS['native']/'CACHE_RECEIPT.json',receipt)


if __name__=='__main__':main()
