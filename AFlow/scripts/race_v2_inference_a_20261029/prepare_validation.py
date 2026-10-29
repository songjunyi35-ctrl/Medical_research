"""Build validation once on inference worker; no training/test pixels loaded."""
import concurrent.futures,multiprocessing as mp,time
import numpy as np
from scripts.race_v2_inference_a_20261029.common import *
from scripts.race_seg_20261028.prepare import process_case

def main():
    verify();CACHE.mkdir(parents=True,exist_ok=True)
    with (R/'VALIDATION_DATA_PREACCESS.json').open('x') as f:
        json.dump(dict(at=now(),split='validation',scans=108,subjects=50,test_pixels=False),f)
    reference=read(ROOT/'archive/research_records/2026-10-28_race_seg/DATA_validation.json')
    cohort=rows();assert reference['case_ids']==[r['case_id'] for r in cohort]
    count=len(cohort);vol=int(np.prod(CANVAS));sizes={'images.f32':vol*4,'labels.u8':vol,'canonical_images.f32':128**3*4,'canonical_labels.u8':128**3}
    for name,size in sizes.items():
        with (CACHE/name).open('xb') as f:f.truncate(count*size)
    start=time.monotonic();geometry={}
    with concurrent.futures.ProcessPoolExecutor(12,mp_context=mp.get_context('spawn')) as pool:
        for cid,g in pool.map(process_case,[('validation',i,row,str(CACHE)) for i,row in enumerate(cohort)]):
            guard();assert g==reference['geometry'][cid],cid
            geometry[cid]=g
            write(R/'CACHE_HEARTBEAT.json',dict(at=now(),completed=len(geometry),total=count,seconds=time.monotonic()-start))
    digests={name:sha(CACHE/name) for name in sizes}
    assert digests==reference['digests'],dict(actual=digests,expected=reference['digests'])
    index=dict(reference);index['geometry']=geometry
    write(CACHE/'index.json',index)
    ready=dict(at=now(),status='passed',worker=1154121,split='validation',count=108,subjects=50,canvas=list(CANVAS),cache_path=str(CACHE),index_sha256=sha(CACHE/'index.json'),payload_sha256=digests,all_four_payloads_exact=True,historical_geometry_exact=True,seconds=time.monotonic()-start,test_access=False)
    write(R/'CACHE_READY.json',ready);write(CACHE.parent/'CACHE_READY.json',ready)

if __name__=='__main__':
    try:main()
    except BaseException as e:
        import traceback
        write(R/'CACHE_FAILURE.json',dict(at=now(),error=repr(e),traceback=traceback.format_exc()));raise

