"""Bounded native3D DenseCRF sweep; label access is isolated to scoring."""
import concurrent.futures,multiprocessing as mp,csv
import numpy as np
from scripts.segresnet_auto_inference12h_20261028.common import *
from scripts.segresnet_auto_inference12h_20261028 import common as shared
from scripts.segresnet_auto_inference12h_20261028.guard_cached_v1 import guard
from scripts.segresnet_auto_inference12h_20261028.dense_crf_v1 import infer
shared.guard=guard
_DATA=None
THRESHOLDS=np.round(np.arange(.025,.9751,.025),6).tolist()


def definitions():
    rows=[dict(id='unary_only',graph='unary',spatial_sigma=5.,color_sigma=.5,weight=0.,gaussian_weight=0.,iterations=1)]
    for iteration in [5,10]:rows.append(dict(id=f'gaussian_only_i{iteration}',graph='gaussian',spatial_sigma=5.,color_sigma=.5,weight=0.,gaussian_weight=1.,iterations=iteration))
    for space in [2.,5.,10.]:
        for color in [.25,1.]:
            for weight in [1.,3.]:
                for iteration in [5,10]:rows.append(dict(id=f'dense_s{space}_c{color}_w{weight}_i{iteration}',graph='bilateral',spatial_sigma=space,color_sigma=color,weight=weight,gaussian_weight=1.,iterations=iteration))
    return rows


def case_task(args):
    global _DATA
    slot,identifier=args
    if _DATA is None:_DATA=data()
    rows,geometry,images,labels=_DATA;cid=rows[slot]['case_id'];sl=slices(geometry[cid])
    image=np.array(images[slot][sl],np.float32);unary=np.load(TMP/'dense_crf_inputs'/(cid+'.npy'))
    recipes=definitions();lookup={r['id']:i for i,r in enumerate(recipes)};values=np.zeros((len(recipes),len(THRESHOLDS)),np.float32)
    target=np.array(labels[slot][sl]>0);start=time.monotonic()
    def record(name,p):
        ri=lookup[name];codes=(p.clip(0,1)*65535+.5).astype(np.int64)
        hist=np.bincount((codes+target.astype(np.int64)*65536).ravel(),minlength=131072).reshape(2,65536)
        tail=np.cumsum(hist[:,::-1],axis=1,dtype=np.int64)[:,::-1];cuts=np.floor(np.array(THRESHOLDS)*65535).astype(int)+1
        tp=tail[1,cuts];fp=tail[0,cuts];values[ri]=(2*tp/np.maximum(tp+fp+hist[1].sum(),1)).astype(np.float32)
        save_array(TMP/'analyses'/identifier/'histograms'/str(ri)/(cid+'.npy'),hist.astype(np.uint32))
    vendor=TMP/'densecrf_vendor'
    record('unary_only',infer(unary,image,5.,.5,0.,iterations=(1,),vendor=vendor,gaussian_weight=0.)[1])
    for iteration,p in infer(unary,image,5.,.5,0.,iterations=(5,10),vendor=vendor,gaussian_weight=1.).items():record(f'gaussian_only_i{iteration}',p)
    for space in [2.,5.,10.]:
        for color in [.25,1.]:
            for weight in [1.,3.]:
                for iteration,p in infer(unary,image,space,color,weight,iterations=(5,10),vendor=vendor,gaussian_weight=1.).items():
                    assert np.isfinite(p).all();record(f'dense_s{space}_c{color}_w{weight}_i{iteration}',p)
    dest=TMP/'analyses'/identifier/'cases'/(cid+'.npy');save_array(dest,values)
    return slot,time.monotonic()-start


def execute(job):
    verify_job(job);assert read(R/'DENSE_CRF_INPUT_READY.json')['state']=='complete'
    folder=R/'analyses'/job['id'];folder.mkdir(parents=True,exist_ok=True);scratch=TMP/'analyses'/job['id'];scratch.mkdir(parents=True,exist_ok=True)
    rows,_,_,_=data();recipes=definitions();n=len(recipes);ready=read(R/'DENSE_CRF_INPUT_READY.json');lookup={r['case_id']:r for r in ready['records']}
    write(folder/'PLAN.json',dict(at=now(),recipes=recipes,thresholds=THRESHOLDS,input_ready_sha256=sha(R/'DENSE_CRF_INPUT_READY.json'),reference='current8-view calibrated unary; zero-pairwise control included, not assume numerical identity with original decoder',GT_input_to_infer=False,test_tuned=True,model_forwards_added=0))
    from scripts.segresnet_auto_inference12h_20261028.guard_v2 import stable_size
    required=len(recipes)*270*(2*65536*4+128)+4*2**20
    while stable_size(TMP)+stable_size(CACHE)+read(R/'MAIN_SCRATCH_BUDGET.json')['bytes']+required+256*2**20>300*2**30:
        guard();write(R/'DENSE_CRF_STATUS.json',dict(at=now(),state='waiting_for_scratch'));time.sleep(10)
    tasks=[(i,job['id']) for i,r in enumerate(rows) if not (scratch/'cases'/(r['case_id']+'.npy')).exists()]
    start=time.monotonic();complete=270-len(tasks);timings=[]
    with concurrent.futures.ProcessPoolExecutor(6,mp_context=mp.get_context('spawn')) as pool:
        for slot,seconds in pool.map(case_task,tasks,chunksize=1):
            guard();complete+=1;timings.append(seconds)
            write(R/'DENSE_CRF_STATUS.json',dict(at=now(),completed=complete,total=270,configurations=n*len(THRESHOLDS),last_case_seconds=seconds,seconds=time.monotonic()-start))
    dice=np.stack([np.load(scratch/'cases'/(r['case_id']+'.npy')) for r in rows],axis=-1);save_array(scratch/'dice.npy',dice)
    manifest=read(CACHE/'index.json');labels=np.memmap(CACHE/'labels.u8',np.uint8,'r',shape=(270,*CANVAS));hits=0
    for i,r in enumerate(rows):hits+=bool(labels[(i,)+tuple(np.array(lookup[r['case_id']]['point'])+np.array(manifest['geometry'][r['case_id']]['canvas_offset']))])
    groups=[[i for i,r in enumerate(rows) if r['subject_id']==subject] for subject in sorted({r['subject_id'] for r in rows})]
    mean=dice.mean(-1,dtype=np.float64);sub=np.stack([dice[...,g].mean(-1,dtype=np.float64) for g in groups],-1).mean(-1);ranking=[]
    for ri,recipe in enumerate(recipes):
        ti=int(np.argmax(mean[ri]));r=dict(recipe_index=ri,recipe=dict(recipe,kind='dense_crf',views=list(range(8)),neural_case_forwards=8),threshold=THRESHOLDS[ti],threshold_index=ti,scan_dice=float(mean[ri,ti]),subject_dice=float(sub[ri,ti]),point_hits=hits);ranking.append(r)
    ranking.sort(key=lambda x:(-x['scan_dice'],-x['subject_dice'],x['recipe_index']));chosen=ranking[:8]
    # Include unary-only in post-calibration even if its constant threshold is lower.
    control=next(r for r in ranking if r['recipe_index']==0)
    if control['recipe_index'] not in {x['recipe_index'] for x in chosen}:chosen.append(control)
    with (folder/'ALL_CONFIGURATIONS.tsv').open('w') as f:
        writer=csv.writer(f,delimiter='\t',lineterminator='\n');writer.writerow(['recipe','threshold','scan_dice','subject_dice','point_hits'])
        for ri,recipe in enumerate(recipes):
            for ti,t in enumerate(THRESHOLDS):writer.writerow([recipe['id'],t,mean[ri,ti],sub[ri,ti],hits])
    write(folder/'RANKING.json',dict(at=now(),best=ranking[0],best_by_recipe=ranking,calibration_shortlist=chosen,total_configurations=n*len(THRESHOLDS),test_tuned=True))
    write(folder/'DONE.json',dict(at=now(),state='complete',best=ranking[0],configurations=n*len(THRESHOLDS),recipes=n,histogram_shortlist=len(chosen),seconds=time.monotonic()-start,median_case_seconds=float(np.median(timings)) if timings else None,dice_sha256=sha(scratch/'dice.npy'),model_forwards=0,test_tuned=True))
    print(ranking[0],flush=True)


if __name__=='__main__':execute(read(R/'cpu_aux_queue_v1/c1_0000_dense_crf.json'))
