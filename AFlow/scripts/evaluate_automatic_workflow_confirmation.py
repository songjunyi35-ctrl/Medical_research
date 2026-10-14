"""One frozen six-output internal test panel; no test-driven selection."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import statistics
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from sammed3d.workflow_medoid import medoid
from sammed3d.automatic_fixed_workflow import restored_atom,fuse_atoms,online_logits
from sammed3d.t1ce_wt_capg3d import AutomaticPromptLocator
from sammed3d.packed_prediction_masks import PackedPredictionWriter
from scripts.train_t1ce_wt_capg3d import BrainCropDataset
from scripts.evaluate_t1ce_wt_plain_locator_test270 import TestBrainCropDataset
from scripts.evaluate_structured_background_aux import bootstrap,compare
from scripts import weak_elastic_test270_helpers as t
from scripts import structured_background_protocol as p

SOURCE=p.ROOT/'archive/research_records/2026-09-11_automatic_workflow_search'
RECORD=p.ROOT/'archive/research_records/2026-09-11_automatic_workflow_confirmation'
TMP=Path('/tmp/aflow-automatic-workflow-confirmation-20260911')
FILES=['sammed3d/workflow_medoid.py','tests/test_workflow_medoid.py',
       'sammed3d/automatic_fixed_workflow.py','sammed3d/automatic_workflow_search.py',
       'sammed3d/packed_prediction_masks.py','sammed3d/t1ce_wt_capg3d.py',
       'scripts/evaluate_automatic_workflow_confirmation.py','scripts/verify_automatic_workflow_confirmation.py',
       'scripts/finalize_automatic_workflow_confirmation.py','scripts/weak_elastic_test270_helpers.py',
       'scripts/evaluate_structured_background_aux.py','scripts/train_t1ce_wt_capg3d.py',
       'scripts/evaluate_t1ce_wt_plain_locator_test270.py',str((RECORD/'IDEA.md').relative_to(p.ROOT))]


def summary(rows,seed):
    groups={s:[r['dice'] for r in rows if r['subject_id']==s] for s in sorted({r['subject_id'] for r in rows})}
    assert len(rows)==270 and len(groups)==119
    return {'scans':270,'subjects':119,'scan_dice':statistics.fmean(r['dice'] for r in rows),
            'subject_dice':statistics.fmean(statistics.fmean(v) for v in groups.values()),
            'point_hits':sum(r['point_hit'] for r in rows),**bootstrap(rows,seed),'per_case':rows}


def fixed_models():
    result=json.loads((SOURCE/'RESULT.json').read_text())
    receipt=json.loads((SOURCE/'VERIFICATION.json').read_text())
    assert receipt['status']=='passed' and p.sha(SOURCE/'RESULT.json')==receipt['result_sha256']
    assert result['development_engineering_loop_passed']
    models=[]
    for report in result['checkpoints']:
        p.checked(report['checkpoint'],report['sha256'])
        folds=[r['selection']['selected'] for r in result['selection_receipts']
               if r['seed']==report['seed'] and r['controller']=='exhaustive']
        m=medoid(folds);assert len(m['views'])<=4
        models.append({**report,'medoid':m})
    freeze={'source_result_sha256':p.sha(SOURCE/'RESULT.json'),'source_verification_sha256':p.sha(SOURCE/'VERIFICATION.json'),
            'models':models,'method':'medoid of three primary fold view sets; ties cost then ID',
            'rule_proposed_after_development':True,'test_used_for_choice':False}
    path=RECORD/'FINAL_WORKFLOWS.json'
    if path.exists():assert json.loads(path.read_text())==json.loads(json.dumps(freeze))
    else:p.write_new(path,freeze)
    return models


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['smoke','test'],required=True);args=ap.parse_args()
    formal=args.phase=='test'
    output=RECORD/('RAW_TEST270.json' if formal else 'SMOKE.json')
    prepath=RECORD/('TEST_PREACCESS.json' if formal else 'SMOKE_PREACCESS.json')
    if output.exists() or prepath.exists():raise FileExistsError('one-run phase already started')
    fit,dev=p.check_roles();models=fixed_models();hashes={f:p.sha(p.ROOT/f) for f in FILES}
    testinfo=None
    if formal:
        smoke=json.loads((RECORD/'SMOKE.json').read_text());assert smoke['status']=='passed' and smoke['files']==hashes
        p.checked(t.TEST,t.TEST_SHA);p.checked(t.CACHE/'index.json',t.INDEX_SHA)
        p.checked(t.CACHE/'images.f32',t.IMAGES_SHA);p.checked(t.CACHE/'labels.u8',t.LABELS_SHA)
        cases=p.rows(t.TEST);assert len(cases)==270 and len({r['subject_id'] for r in cases})==119
        assert not {r['subject_id'] for r in cases}&{r['subject_id'] for r in fit+dev}
        if TMP.exists():raise FileExistsError(TMP)
        if shutil.disk_usage('/tmp').free<20*1024**3:raise RuntimeError('20GiB output reserve required')
        testinfo={'manifest':str(t.TEST),'manifest_sha256':t.TEST_SHA,'cache':str(t.CACHE),
                  'index_sha256':t.INDEX_SHA,'images_sha256':t.IMAGES_SHA,'labels_sha256':t.LABELS_SHA,
                  'scans':270,'subjects':119,'case_ids':[r['case_id'] for r in cases]}
    else:
        p.checked(p.CACHE/'index.json',p.INDEX_SHA)
        frozen=json.loads((SOURCE/'DEV_PREACCESS.json').read_text())
        p.checked(p.CACHE/'images.f32',frozen['images_sha256']);p.checked(p.CACHE/'labels.u8',frozen['labels_sha256'])
        cases=fit[:8]
    expected=len(cases)*sum(8+len(m['medoid']['views']) for m in models)
    p.write_new(prepath,{'created_utc':datetime.now(timezone.utc).isoformat(),'files':hashes,'models':models,
        'final_workflows_sha256':p.sha(RECORD/'FINAL_WORKFLOWS.json'),'test':testinfo,
        'expected_case_forwards':expected,'bootstrap_seeds':[2026091161,2026091162],'bootstrap_samples':10000,
        'output':str(output),'tmp_output':str(TMP),'forwards_at_write':0,'threshold':0,
        'point':'global raw-logit argmax','views_fusion':'canonical ordered float32 mean',
        'authorization_scope':'user requested complete workflow-search/pruning research and autonomous continuation; fixed internal report only, no test-driven selection',
        'evidence_status':'reused internal nonblind, development-derived final workflow, no new-test tuning',
        'one_run_only':True,'performance_access_at_write':False})
    if formal:
        dataset=TestBrainCropDataset(t.CACHE,[r['case_id'] for r in cases])
        TMP.mkdir(exist_ok=False)
    else:dataset=BrainCropDataset(p.CACHE,expected_split='train',case_ids=[r['case_id'] for r in cases])
    loader=DataLoader(dataset,batch_size=8,num_workers=0,shuffle=False)
    device=torch.device('cuda:0');torch.cuda.set_device(device);torch.set_num_threads(1);torch.cuda.reset_peak_memory_stats(device)
    names=[f'seed{m["seed"]}_{kind}' for m in models for kind in ('identity','pruned','full8')]
    writer=PackedPredictionWriter(TMP/'masks',names,[r['case_id'] for r in cases]) if formal else None
    results={};outputs={};counts=[0];parity_cases=0;started=time.monotonic()
    def count(model,args,out):counts[0]+=len(args[0])
    try:
        for ordinal,report in enumerate(models):
            seed=report['seed'];model=AutomaticPromptLocator(center_aware=False).to(device).eval()
            ck=torch.load(report['checkpoint'],map_location='cpu',weights_only=False)
            model.load_state_dict(ck['model_state_dict'],strict=True);hook=model.register_forward_hook(count)
            rows={kind:[] for kind in ('identity','pruned','full8')};stores={}
            if formal:
                for kind in rows:stores[kind]=np.memmap(TMP/f'seed{seed}_{kind}.f32',mode='w+',dtype=np.float32,shape=(270,128,128,128))
            slot=0
            for batch in loader:
                if time.monotonic()-started>1800:raise TimeoutError('30-minute test phase limit')
                image=batch['image'].to(device);labels=batch['label'].to(device)
                atoms=[restored_atom(model,image,v) for v in range(8)]
                pruned=online_logits(model,image,report['medoid']['views'])
                assert torch.equal(pruned,fuse_atoms(atoms,report['medoid']['views'])),'online subset not exact'
                parity_cases+=len(image)
                out={'identity':atoms[0],'pruned':pruned,'full8':fuse_atoms(atoms,range(8))}
                for kind,logits in out.items():
                    assert torch.isfinite(logits).all()
                    if formal:
                        rows[kind].extend(t.rows_from_logits(logits,labels,batch))
                        writer.append(f'seed{seed}_{kind}',(logits[:,0]>0).cpu().numpy(),batch['case_id'])
                        stores[kind][slot:slot+len(image)]=logits[:,0].cpu().numpy()
                slot+=len(image)
                print(json.dumps({'phase':args.phase,'seed':seed,'cases':slot,'forwards':counts[0]}),flush=True)
                del out,atoms,pruned,logits,image,labels
            if formal:
                for kind,store in stores.items():
                    store.flush();path=TMP/f'seed{seed}_{kind}.f32'
                    outputs[f'seed{seed}_{kind}']={'path':str(path),'sha256':p.sha(path),'bytes':path.stat().st_size,'shape':[270,128,128,128],'dtype':'float32'}
                    results[f'seed{seed}_{kind}']=summary(rows[kind],2026091161+ordinal)
                del stores,store
            hook.remove();del model,ck;torch.cuda.empty_cache()
        assert counts[0]==expected and parity_cases==2*len(cases)
        if not formal:
            p.write_new(output,{'status':'passed','files':hashes,'case_forwards':counts[0],
                'online_exact_cases':parity_cases,'seconds':time.monotonic()-started,
                'preaccess_sha256':p.sha(prepath),'test_access':False})
        else:
            mask_index=writer.finalize();p.write_new(RECORD/'MASK_INDEX.json',mask_index)
            comparisons={};gates={}
            for ordinal,m in enumerate(models):
                seed=m['seed'];a,b,c=(results[f'seed{seed}_{k}'] for k in ('identity','pruned','full8'))
                contrast=compare(c,b,2026091161+ordinal)
                comparisons[str(seed)]={'pruned_vs_full8':contrast,'pruned_vs_identity':compare(a,b,2026091161+ordinal)}
                gates[str(seed)]={'fixed_forwards_at_most4':len(m['medoid']['views'])<=4,
                    'scan_loss_at_most003':contrast['scan_delta']>=-.003,
                    'subject_loss_at_most003':contrast['subject_delta']>=-.003,
                    'scan_ci_floor':contrast['scan_cluster_95ci'][0]>=-.005,
                    'subject_ci_floor':contrast['subject_95ci'][0]>=-.005,
                    'point_floor':b['point_hits']>=max(a['point_hits'],c['point_hits'])-1}
            p.write_new(output,{'status':'completed','created_utc':datetime.now(timezone.utc).isoformat(),
                'results':results,'comparisons':comparisons,'gates':gates,'both_models_passed':all(all(g.values()) for g in gates.values()),
                'logits':outputs,'mask_index_sha256':p.sha(RECORD/'MASK_INDEX.json'),
                'test_preaccess_sha256':p.sha(prepath),'case_forwards':counts[0],
                'online_exact_cases':parity_cases,'seconds':time.monotonic()-started,
                'peak_bytes':torch.cuda.max_memory_allocated(device),'test_access':True,
                'test_tuned':False,'reused_nonblind':True,'external':False,
                'closed_no_retuning':True})
        print('Completed '+str(output),flush=True)
    except BaseException as error:
        if writer:writer.close()
        p.write_new(RECORD/(args.phase.upper()+'_FAILURE.json'),{'error':repr(error),'case_forwards':counts[0],
            'completed_results':results,'test_access':formal,'preaccess_sha256':p.sha(prepath)})
        raise


if __name__=='__main__':main()
