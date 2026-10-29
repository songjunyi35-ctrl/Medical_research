"""32 frozen image-only decoders; labels used only for metrics and selection."""
import concurrent.futures,multiprocessing as mp,time
import numpy as np
import torch
from scripts.race_v2_inference_a_20261029.common import *
from scripts.race_v2_inference_a_20261029.predictor import decode_all

def case_task(slot):
    torch.set_num_threads(1);index=read(CACHE/'index.json');cid=index['case_ids'][slot];g=index['geometry'][cid];sl=slices(g)
    images=np.memmap(CACHE/'images.f32',np.float32,'r',shape=(108,*CANVAS))
    labels=np.memmap(CACHE/'labels.u8',np.uint8,'r',shape=(108,*CANVAS))
    atoms=read(R/'ATOMS_READY.json')['atoms'];lookup={(a['slot'],a['view']):a for a in atoms}
    logits=[]
    for view in range(8):
        a=lookup[slot,view];assert sha(a['path'])==a['sha256'];v=np.load(a['path'],allow_pickle=False);assert list(v.shape)==a['shape'];logits.append(v)
    recipes=read(R/'PLAN.json')['recipes']
    # Label cannot enter the prediction API.
    outputs=decode_all(logits,np.array(images[slot][sl],np.float32),recipes)
    target=np.array(labels[slot][sl]>0);row=rows()[slot];records=[]
    for recipe,out in zip(recipes,outputs):
        mask=out.pop('mask');tp=int((mask&target).sum());pred=int(mask.sum());positive=int(target.sum());den=pred+positive
        d=float(np.float32(2*tp)/np.float32(den)) if den else 1.
        records.append(dict(case_id=cid,subject_id=row['subject_id'],source_slot=slot,configuration_id=recipe['id'],locator_dsc=d,tp=tp,fp=pred-tp,fn=positive-tp,automatic_point_hits_gt=bool(target[tuple(out['automatic_point'])]),**out))
    return slot,records

def metrics(records):
    subjects=sorted({r['subject_id'] for r in records})
    return dict(scan_dice=float(np.mean([r['locator_dsc'] for r in records],dtype=np.float64)),subject_dice=float(np.mean([np.mean([r['locator_dsc'] for r in records if r['subject_id']==s],dtype=np.float64) for s in subjects])),point_hits=sum(r['automatic_point_hits_gt'] for r in records))

def main():
    verify();assert read(R/'ATOMS_READY.json')['case_forwards']==864
    recipes=read(R/'PLAN.json')['recipes'];start=time.monotonic();matrix={}
    with concurrent.futures.ProcessPoolExecutor(8,mp_context=mp.get_context('spawn')) as pool:
        for slot,records in pool.map(case_task,range(108)):
            guard();matrix[slot]=records;write(TMP/'cases'/f'{slot:03}.json',records)
            write(R/'ANALYSIS_HEARTBEAT.json',dict(at=now(),completed=len(matrix),total=108,configurations=len(recipes),seconds=time.monotonic()-start))
    per_config=[[matrix[slot][i] for slot in range(108)] for i in range(len(recipes))]
    ranking=[dict(configuration_index=i,id=recipes[i]['id'],neural_case_forwards=recipes[i]['neural_case_forwards'],**metrics(record)) for i,record in enumerate(per_config)]
    ranking.sort(key=lambda r:(-r['scan_dice'],-r['subject_dice'],-r['point_hits'],r['configuration_index']))
    plan=read(R/'PLAN.json');cv=[];heldout=[];heldout_reference=[]
    for fold in range(5):
        select=[i for i,r in enumerate(rows()) if plan['folds'][r['subject_id']]!=fold]
        held=[i for i,r in enumerate(rows()) if plan['folds'][r['subject_id']]==fold]
        chosen=min(range(len(recipes)),key=lambda ri:(-metrics([per_config[ri][i] for i in select])['scan_dice'],-metrics([per_config[ri][i] for i in select])['subject_dice'],-metrics([per_config[ri][i] for i in select])['point_hits'],ri))
        actual=[per_config[chosen][i] for i in held];reference=[per_config[2][i] for i in held]
        heldout.extend(actual);heldout_reference.extend(reference)
        cv.append(dict(fold=fold,selected_index=chosen,selected_id=recipes[chosen]['id'],development_scans=len(select),heldout_scans=len(held),heldout_metrics=metrics(actual),legacy_heldout_metrics=metrics(reference),all_candidates_heldout=[dict(id=recipes[i]['id'],**metrics([per_config[i][j] for j in held])) for i in range(len(recipes))]))
    from scripts.race_v2_inference_a_20261029.statistics import summary_ci,paired
    best=ranking[0];bi=best['configuration_index'];reference=per_config[2]
    comparisons={recipes[i]['id']:paired(per_config[bi],per_config[i]) for i in (0,1,2)}
    worsened=sorted([dict(case_id=c['case_id'],subject_id=c['subject_id'],dice=c['locator_dsc'],legacy_dice=b['locator_dsc'],delta=c['locator_dsc']-b['locator_dsc']) for c,b in zip(per_config[bi],reference) if c['locator_dsc']<b['locator_dsc']],key=lambda r:r['delta'])
    write(R/'RAW_VALIDATION108.json',dict(at=now(),per_configuration={recipes[i]['id']:per_config[i] for i in range(len(recipes))},model_forwards=864,new_test_access=False))
    write(R/'RANKING.json',dict(at=now(),ranking=ranking,best=best,references=ranking[:0]+[next(r for r in ranking if r['configuration_index']==i) for i in (0,1,2)],best_metrics_ci=summary_ci(per_config[bi]),best_paired_vs_references=comparisons,patient_fivefold=cv,heldout_selected_metrics=metrics(heldout),heldout_legacy_metrics=metrics(heldout_reference),heldout_paired_vs_legacy=paired(sorted(heldout,key=lambda r:r['source_slot']),sorted(heldout_reference,key=lambda r:r['source_slot'])),worsened_vs_legacy=worsened,configuration_count=len(recipes),selection_adjusted=False,descriptive_intervals=True,independent_confirmation=False,test_tuned_history=True,model_forwards_added_by_decoding=0,seconds=time.monotonic()-start))
    write(R/'SELECTED_RECIPE.json',dict(at=now(),selection_on='validation108',candidate=recipes[bi],checkpoint_sha256=CHECKPOINT_SHA,plan_sha256=sha(R/'PLAN.json'),ranking_sha256=sha(R/'RANKING.json'),new270_scoring_authorized_by_this_file=False))
    write(R/'DONE.json',dict(at=now(),state='validation108_complete',best=best,recipes=len(recipes),patient_case_forwards=864,new270_scoring=False,test_tuned_history=True))

if __name__=='__main__':
    try:main()
    except BaseException as e:
        import traceback
        write(R/'ANALYSIS_FAILURE.json',dict(at=now(),error=repr(e),traceback=traceback.format_exc()));raise
