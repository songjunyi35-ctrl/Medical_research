"""Two disjoint GPU case lanes; all prediction mathematics stays in frozenv1."""
import argparse,csv,subprocess,sys
import numpy as np
import torch
from scripts.segresnet_auto_inference12h_20261028.common import *
from scripts.segresnet_auto_inference12h_20261028.guard_cached_v1 import guard as cached_guard
from scripts.segresnet_auto_inference12h_20261028.guard_v2 import stable_size
from scripts.segresnet_auto_inference12h_20261028 import intensity_ensemble_v1 as math
prepare=math.prepare
probability=math.probability
definitions=math.definitions
ID='g2_0000_intensity_ensemble'


def storage_worker(path,lane):
    values=np.lib.format.open_memmap(path,mode='r+')
    for slot in range(lane,270,2):values[:,:,slot]=np.arange(64*3,dtype=np.float32).reshape(64,3)+slot
    values.flush()


def storage_gate():
    folder=TMP/'dual_ensemble_storage_gate';folder.mkdir(parents=True,exist_ok=True);path=folder/'values.npy';values=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(64,3,270));values[:]=np.nan;values.flush()
    command=[sys.executable,'-m','scripts.segresnet_auto_inference12h_20261028.intensity_ensemble_dual_v2','--storage-worker',str(path)]
    children=[subprocess.Popen(command+['--lane',str(lane)],cwd=ROOT) for lane in [0,1]]
    assert all(child.wait()==0 for child in children)
    expected=np.arange(64*3,dtype=np.float32).reshape(64,3,1)+np.arange(270,dtype=np.float32)[None,None,:];assert np.array_equal(values,expected)
    del values;path.unlink()
    gate=read(R/'INTENSITY_ENSEMBLE_GATE.json');assert gate['state']=='passed' and gate['source_sha256']==sha(ROOT/'scripts/segresnet_auto_inference12h_20261028/intensity_ensemble_v1.py')
    write(R/'INTENSITY_ENSEMBLE_DUAL_GATE.json',dict(at=now(),state='passed',source_sha256=sha(Path(__file__)),math_source_sha256=gate['source_sha256'],disjoint_mmap_columns_exact=True,all270_slots_once=True,patient_forward=False,network_forward=False,GT_access=False,numerical_grid_changed=False))
    print(dict(state='passed',storage_columns=270,unchanged_math=True))


def run_lane(job,lane,phase):
    numeric_mode();torch.cuda.set_per_process_memory_fraction(.07)
    folder=R/'analyses'/job['id'];scratch=TMP/'analyses'/job['id'];recipes=definitions(job);rows,geometry,_,labels=data();thresholds=np.round(np.arange(.025,.9751,.025),6).tolist();start=time.monotonic()
    state=R/'G2_LANES';state.mkdir(exist_ok=True);write(state/f'{phase}_lane{lane}_PID.json',dict(at=now(),pid=os.getpid(),physical_gpu=lane,phase=phase))
    dice=np.lib.format.open_memmap(scratch/'dice.npy',mode='r+') if phase=='score' else None;hits=np.lib.format.open_memmap(scratch/'hits.npy',mode='r+') if phase=='score' else None
    chosen=read(folder/'RANKING.json')['calibration_shortlist'] if phase=='histograms' else None
    for done,slot in enumerate(range(lane,270,2),1):
        cached_guard();assert read(R/'RESOURCE_CURRENT.json')['token']==job['resource_token'];cid=rows[slot]['case_id'];target=np.array(labels[slot][slices(geometry[cid])]>0)
        if phase=='score' and (folder/f'case{slot:03}.json').exists():continue
        cache=prepare(cid,job);valid=cache['valid'];gt=torch.from_numpy(target).cuda()
        if phase=='score':
            for ri,recipe in enumerate(recipes):
                p=probability(cache,recipe);h=math.histogram(p[valid],gt[valid]);dice[ri,:,slot]=math.threshold_scores(h,thresholds,int(target.sum()));hits[ri,slot]=bool(gt.flatten()[int(p.argmax())])
            dice.flush();hits.flush();write(folder/f'case{slot:03}.json',dict(at=now(),case_id=cid,lane=lane))
        else:
            for item in chosen:
                path=scratch/'histograms'/str(item['recipe_index'])/(cid+'.npy')
                if not path.exists():save_array(path,math.histogram(probability(cache,item['recipe'])[valid],gt[valid]).astype(np.uint32))
        if done%5==0:write(state/f'{phase}_lane{lane}_STATUS.json',dict(at=now(),pid=os.getpid(),phase=phase,physical_gpu=lane,completed=done,total=135,seconds=time.monotonic()-start))
    write(state/f'{phase}_lane{lane}_DONE.json',dict(at=now(),state='complete',slots=135,physical_gpu=lane,seconds=time.monotonic()-start))


def dual_phase(job,phase):
    env=dict(os.environ);env['CUDA_VISIBLE_DEVICES']='1';log=TMP/('g2_dual_'+phase+'_lane1.log')
    with log.open('a') as output:
        child=subprocess.Popen([sys.executable,'-m','scripts.segresnet_auto_inference12h_20261028.intensity_ensemble_dual_v2','--lane','1','--phase',phase],cwd=ROOT,env=env,stdout=output,stderr=subprocess.STDOUT)
        run_lane(job,0,phase)
        assert child.wait()==0,f'dual child failed; inspect{log}'


@torch.inference_mode()
def execute(job):
    verify_job(job);gate=read(R/'INTENSITY_ENSEMBLE_DUAL_GATE.json');assert gate['state']=='passed' and gate['source_sha256']==sha(Path(__file__)) and gate['math_source_sha256']==sha(ROOT/'scripts/segresnet_auto_inference12h_20261028/intensity_ensemble_v1.py')
    folder=R/'analyses'/job['id'];folder.mkdir(parents=True,exist_ok=True);scratch=TMP/'analyses'/job['id'];scratch.mkdir(parents=True,exist_ok=True);recipes=definitions(job);thresholds=np.round(np.arange(.025,.9751,.025),6).tolist();required=len(recipes)*39*270*4+len(recipes)*270+8*270*(65536*2*4+128)+8*270*11475*6+32*2**20
    assert required<1536*2**20 and stable_size(TMP)+stable_size(CACHE)+read(R/'MAIN_SCRATCH_BUDGET.json')['bytes']+required<300*2**30
    for group in job['group_sources']:assert (R/'jobs'/group/'DONE.json').exists() and len(list((TMP/'groups'/group/'mean_probability').glob('*.npy')))==270
    write(folder/'PLAN.json',dict(at=now(),recipes=recipes,thresholds=thresholds,source_sha256=sha(Path(__file__)),math_source_sha256=gate['math_source_sha256'],configurations=len(recipes)*39,full_original_GT_domain=True,model_forwards_added=0,test_tuned=True,lanes='GPU0 even/GPU1 odd,135 each, fixed original order'))
    dp=scratch/'dice.npy';hp=scratch/'hits.npy'
    for path,dtype,shape in [(dp,np.float32,(len(recipes),39,270)),(hp,bool,(len(recipes),270))]:
        if not path.exists():values=np.lib.format.open_memmap(path,mode='w+',dtype=dtype,shape=shape);values[:]=0;values.flush();del values
    start=time.monotonic();status('intensity_ensemble_dual_scoring',job=job['id'],lanes=2);dual_phase(job,'score')
    assert len(list(folder.glob('case*.json')))==270
    dice=np.load(dp,mmap_mode='r');hits=np.load(hp,mmap_mode='r');assert np.isfinite(dice).all() and (dice>=0).all() and (dice<=1).all();rows,geometry,_,labels=data();means,subjects=math.summary_arrays(np.asarray(dice),rows);ranking=[]
    for ri,recipe in enumerate(recipes):
        ti=int(means[ri].argmax());ranking.append(dict(recipe_index=ri,recipe=recipe,threshold=thresholds[ti],threshold_index=ti,scan_dice=float(means[ri,ti]),subject_dice=float(subjects[ri,ti]),point_hits=int(hits[ri].sum())))
    ranking.sort(key=lambda r:(-r['scan_dice'],-r['subject_dice'],r['recipe_index']));chosen=[r for r in ranking if r['recipe_index']!=0][:7]+[next(r for r in ranking if r['recipe_index']==0)]
    write(folder/'RANKING.json',dict(at=now(),best=ranking[0],best_by_recipe=ranking,calibration_shortlist=chosen,test_tuned=True));write(folder/'GT_TOTALS.json',{r['case_id']:int((labels[i][slices(geometry[r['case_id']])]>0).sum()) for i,r in enumerate(rows)})
    with (folder/'ALL_CONFIGURATIONS.tsv').open('w') as handle:
        writer=csv.writer(handle,delimiter='\t',lineterminator='\n');writer.writerow(['recipe','threshold','scan_dice','subject_dice','point_hits'])
        for ri,recipe in enumerate(recipes):
            for ti,threshold in enumerate(thresholds):writer.writerow([recipe['id'],threshold,means[ri,ti],subjects[ri,ti],int(hits[ri].sum())])
    status('intensity_ensemble_dual_histograms',job=job['id'],lanes=2);dual_phase(job,'histograms')
    write(folder/'DONE.json',dict(at=now(),state='complete',configurations=len(recipes)*39,best=ranking[0],histogram_shortlist=8,seconds=time.monotonic()-start,model_forwards_added=0,test_tuned=True,dual_lanes=2,dice_sha256=sha(dp),hits_sha256=sha(hp)))
    print(ranking[0])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--storage-gate',action='store_true');parser.add_argument('--storage-worker');parser.add_argument('--lane',type=int);parser.add_argument('--phase',choices=['score','histograms']);args=parser.parse_args()
    if args.storage_worker:storage_worker(Path(args.storage_worker),args.lane)
    elif args.storage_gate:storage_gate()
    else:run_lane(read(R/'analysis_queue_aux_v1/g2_0000_intensity_ensemble.json'),args.lane,args.phase)
