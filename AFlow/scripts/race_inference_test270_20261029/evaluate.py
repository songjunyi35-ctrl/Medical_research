"""Two disjoint ranks; one forward per case/view, raw atoms before decoding."""
import argparse
import sys
import traceback
import numpy as np
import torch
from scripts.race_inference_test270_20261029.common import *
from scripts.race_inference_test270_20261029.predictor import load_model,forward_view,decode_outputs,clean_geometry


def data():
    index=read(CACHE/'index.json');assert sha(CACHE/'index.json')==read(R/'CACHE_READY.json')['index_sha256']
    assert index['case_ids']==[x['case_id'] for x in rows()]
    images=np.memmap(CACHE/'images.f32',np.float32,'r',shape=(270,*CANVAS))
    labels=np.memmap(CACHE/'labels.u8',np.uint8,'r',shape=(270,*CANVAS))
    return index,images,labels


def atom_path(slot,flip):return TMP/'atoms'/f'{slot:03}'/f'flip{flip}.npy'


def forward(rank,index,images):
    batch=read(R/'BATCH_DECISION.json')['batch'];slots=list(range(rank,270,2));case_rows=rows()
    assert read(R/'MODEL_PREACCESS.json')['checkpoint_sha256']==CHECKPOINT_SHA
    exclusive(R/f'MODEL_RANK_ACCESS_{rank}.json',dict(at=now(),rank=rank,slots=slots,
              case_ids=[case_rows[i]['case_id'] for i in slots],views=list(range(8)),checkpoint_sha256=CHECKPOINT_SHA,
              maximum_case_forwards=1080,no_automatic_repeated_inference=True))
    model=load_model();torch.cuda.reset_peak_memory_stats();started=time.monotonic();atoms=[];calls=0
    for first in range(0,len(slots),batch):
        guard();current=slots[first:first+batch]
        image=torch.from_numpy(np.array(images[current],dtype=np.float32)).cuda()
        geometry=[clean_geometry(index['geometry'][case_rows[i]['case_id']]) for i in current]
        for flip in range(8):
            guard();values=forward_view(model,image,geometry,flip);calls+=1
            for slot,value in zip(current,values):
                path=atom_path(slot,flip);path.parent.mkdir(parents=True,exist_ok=True)
                with path.open('xb') as f:np.save(f,value,allow_pickle=False)
                atoms.append(dict(slot=slot,view=flip,path=str(path),shape=list(value.shape),sha256=sha(path)))
            write(R/f'FORWARD_JOURNAL_{rank}.json',dict(at=now(),rank=rank,case_forwards_completed=len(atoms),
                  neural_batch_calls=calls,atoms=atoms,patient_model_access=True))
            write(R/f'HEARTBEAT_{rank}.json',dict(at=now(),stage='neural_inference',rank=rank,
                  completed_case_forwards=len(atoms),total_case_forwards=1080))
        del image,values
    assert len(atoms)==1080 and len({(x['slot'],x['view']) for x in atoms})==1080
    write(R/f'FORWARD_COMPLETE_{rank}.json',dict(at=now(),rank=rank,atoms=atoms,case_forwards=1080,
          neural_batch_calls=calls,seconds=time.monotonic()-started,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
          checkpoint_sha256_after=sha(CHECKPOINT)))
    assert sha(CHECKPOINT)==CHECKPOINT_SHA
    del model


def decode(rank,index,images,labels):
    original=read(R/f'FORWARD_COMPLETE_{rank}.json');assert original['case_forwards']==1080
    atom_index={(a['slot'],a['view']):a for a in original['atoms']}
    config=read(R/'SELECTED_RECIPE.json');case_rows=rows();records={'raw':[],'enhanced':[]};started=time.monotonic()
    for completed,slot in enumerate(range(rank,270,2),1):
        guard();cid=case_rows[slot]['case_id'];g=index['geometry'][cid];sl=slices(g);logits=[]
        for flip in range(8):
            entry=atom_index[slot,flip];path=atom_path(slot,flip);assert sha(path)==entry['sha256']
            value=np.load(path,allow_pickle=False);assert list(value.shape)==entry['shape'];logits.append(value)
        outputs=decode_outputs(logits,np.array(images[slot][sl],np.float32),config)
        target=np.array(labels[slot][sl]>0)
        for scheme,value in outputs.items():
            mask=value['mask'];point=value['automatic_point'];assert mask.shape==target.shape
            tp=int((mask&target).sum());pred=int(mask.sum());positive=int(target.sum());den=pred+positive
            score=float(np.float32(2*tp)/np.float32(den)) if den else 1.
            path=R/'predictions'/scheme/(cid+'.npz');path.parent.mkdir(parents=True,exist_ok=True)
            if path.exists():
                previous=np.load(path);old=np.unpackbits(previous['mask_bits'])[:mask.size].reshape(mask.shape).astype(bool)
                assert np.array_equal(old,mask) and previous['point'].tolist()==point
            else:np.savez_compressed(path,mask_bits=np.packbits(mask.ravel()),shape=np.array(mask.shape),point=np.array(point))
            records[scheme].append(dict(case_id=cid,subject_id=case_rows[slot]['subject_id'],source_slot=slot,
                 locator_dsc=score,automatic_point=point,automatic_point_hits_gt=bool(target[tuple(point)]),
                 tp=tp,fp=pred-tp,fn=positive-tp,prediction_path=str(path.relative_to(ROOT)),prediction_sha256=sha(path),
                 **({'quantized_cut':value['quantized_cut']} if scheme=='enhanced' else {})))
        write(R/f'DECODE_JOURNAL_{rank}.json',dict(at=now(),rank=rank,completed_scans=completed,per_scheme=records))
        write(R/f'HEARTBEAT_{rank}.json',dict(at=now(),stage='decoding_scoring',rank=rank,completed=completed,total=135))
    write(R/f'RAW_RANK_{rank}.json',dict(at=now(),rank=rank,per_scheme=records,
          neural=dict(case_forwards=original['case_forwards'],batch_calls=original['neural_batch_calls'],
          seconds=original['seconds'],peak_reserved_gib=original['peak_reserved_gib']),decode_seconds=time.monotonic()-started))


def main(rank,statistics_only):
    verify();manifests();numeric_mode();torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.95)
    assert read(R/'CACHE_READY.json')['all_four_payloads_exact']
    index,images,labels=data()
    if not statistics_only:forward(rank,index,images)
    decode(rank,index,images,labels)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--rank',type=int,choices=(0,1),required=True);p.add_argument('--statistics-only',action='store_true');a=p.parse_args()
    try:main(a.rank,a.statistics_only)
    except BaseException as error:
        traceback.print_exc();write(R/f'INFERENCE_FAILURE_{a.rank}_{time.time_ns()}.json',dict(at=now(),error=repr(error),
              patient_model_access=(R/f'MODEL_RANK_ACCESS_{a.rank}.json').exists(),no_automatic_neural_retry=True))
        sys.stdout.flush();sys.stderr.flush();os._exit(1)
    sys.stdout.flush();sys.stderr.flush();os._exit(0)
