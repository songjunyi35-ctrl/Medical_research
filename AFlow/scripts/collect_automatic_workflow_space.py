"""Freeze and collect all 128 workflows; controller outcomes are not shown here."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from sammed3d.automatic_fixed_workflow import restored_atom, fuse_atoms, online_logits, WORKFLOWS
from sammed3d.automatic_workflow_search import views_for,patient_groups
from sammed3d.t1ce_wt_capg3d import AutomaticPromptLocator
from scripts.train_t1ce_wt_capg3d import BrainCropDataset
from scripts.evaluate_t1ce_wt_flip8_tta_replication_inner import flip8_logits
from scripts import structured_background_protocol as p

RECORD=p.ROOT/'archive/research_records/2026-09-11_automatic_workflow_search'
ENGINEERING=p.ROOT/'archive/research_records/2026-09-11_automatic_workflow_engineering'
TMP=Path('/tmp/aflow-automatic-workflow-search-20260911')
FILES=['sammed3d/automatic_fixed_workflow.py','sammed3d/automatic_workflow_search.py',
       'tests/test_automatic_workflow_search.py','tests/test_automatic_fixed_workflow.py',
       'scripts/collect_automatic_workflow_space.py','scripts/analyze_automatic_workflow_search.py',
       'scripts/structured_background_protocol.py','scripts/evaluate_structured_background_aux.py',
       'scripts/train_t1ce_wt_capg3d.py','scripts/evaluate_t1ce_wt_flip8_tta_replication_inner.py',
       'sammed3d/t1ce_wt_capg3d.py',str((RECORD/'IDEA.md').relative_to(p.ROOT)),
       str((RECORD/'METADATA_CORRECTION.md').relative_to(p.ROOT))]


def score(logits,truth,batch):
    b=len(logits)
    mask=(logits[:,0]>0).reshape(b,-1)
    target=truth[:,0].reshape(b,-1)
    point=logits[:,0].reshape(b,-1).argmax(1)
    inter=(mask & target).sum(1).tolist()
    predicted=mask.sum(1).tolist()
    targets=target.sum(1).tolist()
    hits=target.gather(1,point[:,None])[:,0].tolist()
    points=point.tolist()
    rows=[]
    for i,case in enumerate(batch['case_id']):
        assert targets[i]>0
        rows.append({'case_id':case,'subject_id':batch['subject_id'][i],
                     'dice':2*inter[i]/(predicted[i]+targets[i]),'point_hit':bool(hits[i]),
                     'point_flat_index':points[i],'intersection':inter[i],
                     'predicted_voxels':predicted[i],'target_voxels':targets[i]})
    return rows


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',choices=['smoke','development'],required=True)
    args=parser.parse_args()
    output=RECORD/('SMOKE.json' if args.phase=='smoke' else 'RAW_DEVELOPMENT.json')
    pre=RECORD/('SMOKE_PREACCESS.json' if args.phase=='smoke' else 'DEV_PREACCESS.json')
    if output.exists() or pre.exists(): raise FileExistsError('single phase already started')
    fit,dev=p.check_roles()
    engineering=json.loads((ENGINEERING/'PREACCESS.json').read_text())
    p.checked(p.CACHE/'index.json',p.INDEX_SHA)
    p.checked(p.CACHE/'images.f32',engineering['images_sha256'])
    aux=json.loads((p.RECORD/'AUX_CACHE_RECEIPT.json').read_text())
    p.checked(p.CACHE/'labels.u8',aux['source_labels_sha256'])
    p.checked(p.RECORD/'RAW_DEVELOPMENT.json','80e9bd449d7c72a61de0ecdbdfbc5bb3a61814e7770bbcc1c983ac097b4cd2c6')
    native=json.loads((p.RECORD/'RAW_DEVELOPMENT.json').read_text())
    for model in engineering['models']:
        p.checked(model['checkpoint'],model['sha256'])
    hashes={name:p.sha(p.ROOT/name) for name in FILES}
    if args.phase=='development':
        smoke=json.loads((RECORD/'SMOKE.json').read_text())
        assert smoke['status']=='passed' and smoke['case_forwards']==496
        assert smoke['files']==hashes
        if TMP.exists(): raise FileExistsError(TMP)
        if shutil.disk_usage('/tmp').free < 30*1024**3: raise RuntimeError('30GiB temporary reserve required')
    groups=patient_groups(dev)
    selected=fit[:8] if args.phase=='smoke' else dev
    p.write_new(pre,{'created_utc':datetime.now(timezone.utc).isoformat(),'files':hashes,
        'models':engineering['models'],'fit_sha256':p.FIT_SHA,'dev_sha256':p.DEV_SHA,
        'index_sha256':p.INDEX_SHA,'images_sha256':engineering['images_sha256'],
        'labels_sha256':aux['source_labels_sha256'],'patient_groups':groups,
        'case_ids':[r['case_id'] for r in selected],
        'space':{str(k):views_for(k) for k in range(128)},'forwards_at_write':0,
        'access':{'fit_smoke':args.phase=='smoke','reused_development':args.phase=='development',
                  'test':False,'outer':False,'confirmation':False,'external':False}})
    torch.set_num_threads(1)
    device=torch.device('cuda:0');torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    loader=DataLoader(BrainCropDataset(p.CACHE,expected_split='train',case_ids=[r['case_id'] for r in selected]),
                      batch_size=8,num_workers=0,shuffle=False)
    if args.phase=='development': TMP.mkdir(exist_ok=False)
    counts=[0];results={};cache={};started=time.monotonic()
    def counter(module,args,output): counts[0]+=len(args[0])
    try:
        for report in engineering['models']:
            seed=report['seed']
            model=AutomaticPromptLocator(center_aware=False).to(device).eval()
            ck=torch.load(report['checkpoint'],map_location='cpu',weights_only=False)
            model.load_state_dict(ck['model_state_dict'],strict=True)
            handle=model.register_forward_hook(counter)
            rows={str(k):[] for k in range(128)}
            atom_path=TMP/f'seed{seed}.f32'
            store=np.memmap(atom_path,dtype=np.float32,mode='w+',shape=(180,8,128,128,128)) if args.phase=='development' else None
            slot=0
            for bi,batch in enumerate(loader):
                if time.monotonic()-started>7200: raise TimeoutError('two-hour collection limit')
                image=batch['image'].to(device);truth=batch['label'].to(device)>.5
                atoms=[restored_atom(model,image,k) for k in range(8)]
                assert all(torch.isfinite(a).all() for a in atoms)
                if store is not None:
                    for view,atom in enumerate(atoms): store[slot:slot+len(image),view]=atom[:,0].cpu().numpy()
                for k in range(128):
                    logits=fuse_atoms(atoms,views_for(k))
                    assert torch.isfinite(logits).all()
                    if args.phase=='development': rows[str(k)].extend(score(logits,truth,batch))
                if args.phase=='smoke':
                    identity,full=flip8_logits(model,image)
                    assert torch.equal(identity,atoms[0]) and torch.equal(full,fuse_atoms(atoms,range(8)))
                    for views in WORKFLOWS.values():
                        assert torch.equal(online_logits(model,image,views),fuse_atoms(atoms,views))
                slot+=len(image)
                print(json.dumps({'phase':args.phase,'seed':seed,'cases':slot,'model_case_forwards':counts[0],
                                  'seconds':time.monotonic()-started}),flush=True)
                del atoms,logits,image,truth
            if args.phase=='development':
                assert slot==180
                assert rows['0']==native['results'][f'seed{seed}_control']['per_case'],'historical identity parity'
                store.flush();del store
                cache[str(seed)]={'path':str(atom_path),'sha256':p.sha(atom_path),
                                  'shape':[180,8,128,128,128],'dtype':'float32','bytes':atom_path.stat().st_size}
                p.write_new(RECORD/f'RAW_SEED{seed}.json',{'seed':seed,'rows':rows,'atom_cache':cache[str(seed)]})
                results[str(seed)]={'path':str(RECORD/f'RAW_SEED{seed}.json'),
                                    'sha256':p.sha(RECORD/f'RAW_SEED{seed}.json')}
            handle.remove();del model,ck
            torch.cuda.empty_cache()
        expected=496 if args.phase=='smoke' else 2880
        assert counts[0]==expected
        p.write_new(output,{'status':'passed' if args.phase=='smoke' else 'completed',
            'created_utc':datetime.now(timezone.utc).isoformat(),'case_forwards':counts[0],
            'files':hashes,'results':results,'cache':cache,'preaccess_sha256':p.sha(pre),
            'seconds':time.monotonic()-started,'peak_bytes':torch.cuda.max_memory_allocated(device),
            'historical_identity_exact':args.phase=='development','development_access':args.phase=='development',
            'test_access':False,'patient_groups':groups})
        print('Completed '+str(output),flush=True)
    except BaseException as error:
        p.write_new(RECORD/(args.phase.upper()+'_FAILURE.json'),{'error':repr(error),
            'completed_case_forwards':counts[0],'completed_results':results,
            'development_access':args.phase=='development','test_access':False,
            'preaccess_sha256':p.sha(pre)})
        raise


if __name__=='__main__': main()
