"""Read-only SHA/shape audit of existing270; no new forward or decoding."""
import concurrent.futures,time
import numpy as np
from scripts.race_v2_inference_a_20261029.common import *
def atom_check(a):
    p=Path(a['path']);value=np.load(p,mmap_mode='r',allow_pickle=False)
    assert list(value.shape)==a['shape'] and value.dtype==np.float32
    assert sha(p)==a['sha256'],str(p)
    assert np.isfinite(value).all(),str(p)
    return dict(slot=a['slot'],view=a['view'],bytes=p.stat().st_size,sha256=a['sha256'])
def main():
    start=time.monotonic();oldtmp=Path('/tmp/aflow_race_e164_test270_20261029');ready=read(OLD/'CACHE_READY.json')
    paths={'native_image':oldtmp/'native/images.f32','native_label':oldtmp/'native/labels.u8','canon_image':oldtmp/'canonical128/images.f32','canon_label':oldtmp/'canonical128/labels.u8'}
    actual={key:dict(path=str(path),exists=path.exists(),bytes=path.stat().st_size,sha256=sha(path)) for key,path in paths.items()}
    assert all(a['sha256']==ready['payload_sha256'][k] for k,a in actual.items())
    assert sha(oldtmp/'native/index.json')==ready['index_sha256']
    assert sha(CHECKPOINT)==CHECKPOINT_SHA
    atoms=[a for rank in (0,1) for a in read(OLD/f'FORWARD_COMPLETE_{rank}.json')['atoms']]
    assert len(atoms)==2160 and len({(a['slot'],a['view']) for a in atoms})==2160
    with concurrent.futures.ThreadPoolExecutor(4) as pool:checked=list(pool.map(atom_check,atoms))
    assert len(list((oldtmp/'atoms').glob('*/*.npy')))==2160
    write(R/'EXISTING270_CACHE_AUDIT.json',dict(at=now(),status='passed',payloads=actual,index_sha256=sha(oldtmp/'native/index.json'),checkpoint_sha256=CHECKPOINT_SHA,atoms_checked=len(checked),atoms_bytes=sum(a['bytes'] for a in checked),every_atom_sha_shape_dtype_finite=True,all2160_exact=True,model_forwards_added=0,new270_scoring=False,seconds=time.monotonic()-start))
if __name__=='__main__':
    try:main()
    except BaseException as e:
        import traceback
        write(R/'EXISTING270_CACHE_AUDIT_FAILURE.json',dict(at=now(),error=repr(e),traceback=traceback.format_exc()));raise
