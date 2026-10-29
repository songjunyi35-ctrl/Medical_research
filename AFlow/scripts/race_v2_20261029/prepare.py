"""Bounded worker-local reconstruction, original numerical builder unchanged."""
import concurrent.futures
import multiprocessing as mp
import numpy as np
from scripts.race_v2_20261029.common import *
from scripts.race_seg_20261028.prepare import process_case

def main():
    disk(); deadline()
    assert sha(SOURCE)==SOURCE_SHA
    from scripts.nnunet_resenc_20261013.common import verify_manifests
    verify_manifests()
    TMP.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(TMP).free < 100*2**30: raise RuntimeError('scratch below100GiB')
    with (R/'DATA_PREACCESS.json').open('x') as handle:
        json.dump(dict(at=now(),train=972,validation=108,test_pixels=False,
                       worker='1154259',single_n4_per_case=True,
                       original_builder='scripts.race_seg_20261028.prepare.process_case'),handle)
    historical={x['role']:x for x in read(ROOT/'archive/research_records/2026-10-26_segresnet_teacher_distill/NATIVE_DATA_AUDIT.json')['caches']}
    for split in ('validation','train'):
        deadline(); records=rows(split); folder=CACHE/split; folder.mkdir(parents=True,exist_ok=True)
        count=len(records); vol=int(np.prod(CANVAS))
        sizes={'images.f32':vol*4,'labels.u8':vol,'canonical_images.f32':128**3*4,'canonical_labels.u8':128**3}
        if split=='train': sizes['classes.u8']=vol
        for name,size in sizes.items():
            with (folder/name).open('xb') as handle: handle.truncate(count*size)
        geometries={}; started=time.monotonic()
        tasks=[(split,i,row,str(folder)) for i,row in enumerate(records)]
        with concurrent.futures.ProcessPoolExecutor(16,mp_context=mp.get_context('spawn')) as pool:
            for cid,geometry in pool.map(process_case,tasks):
                deadline(); geometries[cid]=geometry
                if len(geometries)%25==0 or len(geometries)==count:
                    write(R/'PREPARE_HEARTBEAT.json',dict(at=now(),split=split,done=len(geometries),count=count,seconds=time.monotonic()-started))
        digests={name:sha(folder/name) for name in sizes}
        for name,key in [('images.f32','images_sha256'),('labels.u8','labels_sha256')]:
            if digests[name]!=historical[split][key]: raise RuntimeError('native SHA mismatch '+split+' '+name)
        canonical_path=ROOT/('workspace/BraTS/cache/t1ce_wt_n4_braincrop_128_20260817/train/index.json' if split=='train' else 'workspace/BraTS/cache/t1ce_wt_n4_braincrop_128_ema_20260819/validation108/index.json')
        old=read(canonical_path)
        assert digests['canonical_labels.u8']==old['labels_sha256'] and digests['canonical_images.f32']==old['images_sha256']
        disabled=sum(not g['reference_enabled'] for g in geometries.values())
        assert disabled/count<=.05
        index=dict(schema='aflow.race.native',split=split,count=count,canvas=list(CANVAS),
                   case_ids=[r['case_id'] for r in records],geometry=geometries,digests=digests,
                   test_access=False,reference_disabled=disabled,historical_native_sha_exact=True,
                   historical128_sha_exact=True)
        write(folder/'index.json',index)
        write(R/f'DATA_{split}.json',dict(at=now(),index_path=str(folder/'index.json'),index_sha256=sha(folder/'index.json'),
              payload_bytes={name:(folder/name).stat().st_size for name in sizes},digests=digests,
              historical_native_sha_exact=True,historical128_sha_exact=True,reference_disabled=disabled,
              canvas=list(CANVAS),count=count,seconds=time.monotonic()-started))
    write(R/'DATA_READY.json',dict(at=now(),status='passed',worker='1154259',historical_payloads_exact=True,
          train=972,validation=108,test_pixels=False))

if __name__=='__main__':
    try: main()
    except BaseException as error:
        write(R/f'PREPARE_FAILURE_{time.time_ns()}.json',dict(at=now(),error=repr(error))); raise
