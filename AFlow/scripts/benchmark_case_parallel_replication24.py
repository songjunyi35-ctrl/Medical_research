"""Matched 1/4-process, label-free, bounded online batch pipeline."""
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime,timezone
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import resource
import time
import numpy as np
import SimpleITK as sitk
import torch
from sammed3d.bounded_ordered_pipeline import bounded_ordered_map
from sammed3d.automatic_fixed_workflow import online_logits
from sammed3d.t1ce_wt_capg3d import AutomaticPromptLocator
from scripts.benchmark_workflow_file_to_output import prepare
from scripts import structured_background_protocol as p

RECORD=p.ROOT/'archive/research_records/2026-09-11_case_parallel_replication24'


def initialize_worker():
    torch.set_num_threads(1)
    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1)


def worker(row):
    start=time.monotonic();cpu=time.process_time()
    with torch.inference_mode():image=prepare(row).numpy()
    return {'case_id':row['case_id'],'image':image,'worker_pid':os.getpid(),
            'prepare_seconds':time.monotonic()-start,'prepare_cpu_seconds':time.process_time()-cpu,
            'worker_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}


def rss(pid):
    try:
        for line in Path(f'/proc/{pid}/status').read_text().splitlines():
            if line.startswith('VmRSS:'):return int(line.split()[1])*1024
    except FileNotFoundError:return 0
    return 0


@torch.inference_mode()
def main():
    if (RECORD/'PREACCESS.json').exists():raise FileExistsError('matched run already started')
    p.checked(p.FIT,p.FIT_SHA);all_fit=p.rows(p.FIT); excluded={r['subject_id'] for r in all_fit[:8]}; fit=[r for r in all_fit if r['subject_id'] not in excluded][:24]
    assert len(fit)==24 and len({r['subject_id'] for r in fit})==12 and not excluded & {r['subject_id'] for r in fit}
    source=json.loads((p.ROOT/'archive/research_records/2026-09-11_automatic_workflow_search/DEV_PREACCESS.json').read_text())
    model_info=next(r for r in source['models'] if r['seed']==2026090711)
    p.checked(model_info['checkpoint'],model_info['sha256'])
    p.checked(p.CACHE/'index.json',p.INDEX_SHA);p.checked(p.CACHE/'images.f32',source['images_sha256'])
    files=['scripts/benchmark_case_parallel_replication24.py','scripts/create_case_parallel_replication24.py','scripts/benchmark_workflow_file_to_output.py',
           'sammed3d/bounded_ordered_pipeline.py','tests/test_bounded_ordered_pipeline.py',
           'sammed3d/automatic_fixed_workflow.py','sammed3d/t1ce_wt_capg3d.py',
           'scripts/build_t1ce_wt_cache.py','sammed3d/t1ce_binary.py','sammed3d/t1ce_wt_training.py',
           str((RECORD/'IDEA.md').relative_to(p.ROOT))]
    p.write_new(RECORD/'PREACCESS.json',{'created_utc':datetime.now(timezone.utc).isoformat(),
        'files':{n:p.sha(p.ROOT/n) for n in files},'model':model_info,'fit_sha256':p.FIT_SHA,
        'index_sha256':p.INDEX_SHA,'images_sha256':source['images_sha256'],
        'sources':[{'case_id':r['case_id'],'subject_id':r['subject_id'],'image_path':r['image_path'],
                    'sha256':p.sha(r['image_path'])} for r in fit],
        'order':[[1,4],[4,1]],'max_pending':8,'thread_count_per_worker':1,
        'views':list(range(8)),'expected_forwards':792,'excluded_prior_subjects':sorted(excluded),'selected_subjects':12,'labels_opened':False,'test_access':False})
    idx=json.loads((p.CACHE/'index.json').read_text());lookup={c:i for i,c in enumerate(idx['case_ids'])}
    cache=np.memmap(p.CACHE/'images.f32',mode='r',dtype=np.float32,shape=(len(lookup),128,128,128))
    cached={r['case_id']:np.array(cache[lookup[r['case_id']]],copy=True)[None,None] for r in fit}
    torch.set_num_threads(1);sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1)
    device=torch.device('cuda:0');torch.cuda.set_device(device)
    model=AutomaticPromptLocator(center_aware=False).to(device).eval()
    ck=torch.load(model_info['checkpoint'],map_location='cpu',weights_only=False)
    model.load_state_dict(ck['model_state_dict'],strict=True)
    calls=[0]
    def count(m,args,out):calls[0]+=len(args[0])
    hook=model.register_forward_hook(count)
    for row in fit[:3]:online_logits(model,torch.from_numpy(cached[row['case_id']]).to(device),range(8))
    torch.cuda.synchronize();runs=[];baseline={};started=time.monotonic();deadline=started+2400
    try:
        for repeat,order in enumerate(((1,4),(4,1))):
            for workers in order:
                rows=[];samples=[];begin=time.monotonic()
                ex=ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=initialize_worker)
                success=False
                try:
                    for slot,result in enumerate(bounded_ordered_map(ex,worker,fit,8,deadline)):
                        received=time.monotonic();case=result.pop('case_id');array=result.pop('image')
                        assert case==fit[slot]['case_id'] and np.array_equal(array,cached[case]),'input/order differs'
                        logits=online_logits(model,torch.from_numpy(array).to(device),range(8))
                        output=logits.cpu().numpy();mask=output>0;point=int(np.argmax(output));torch.cuda.synchronize()
                        completed=time.monotonic()
                        fingerprint={'logit_sha256':hashlib.sha256(output.tobytes()).hexdigest(),
                                     'mask_sha256':hashlib.sha256(mask.tobytes()).hexdigest(),'point':point}
                        if case not in baseline:baseline[case]=fingerprint
                        assert baseline[case]==fingerprint,'full output differs across schedules'
                        pids=[os.getpid()]+[v.pid for v in ex._processes.values()]
                        samples.append(sum(rss(pid) for pid in pids))
                        row={'case_id':case,'input_exact':True,'output_exact':True,**fingerprint,**result,
                             'received_since_start':received-begin,'completed_since_start':completed-begin}
                        rows.append(row)
                        print(json.dumps({'repeat':repeat,'workers':workers,'cases':slot+1,
                                          'completed_seconds':completed-begin}),flush=True)
                    success=True;outputs_done=time.monotonic()-begin
                finally:
                    if not success:
                        # Only processes owned by this executor are terminated on failure.
                        processes=list(ex._processes.values())
                        for process in processes:
                            if process.is_alive():process.terminate()
                        for process in processes:
                            process.join(timeout=3)
                            if process.is_alive():process.kill()
                    ex.shutdown(wait=True,cancel_futures=True)
                run={'repeat':repeat,'workers':workers,'rows':rows,'batch_wall_seconds':outputs_done,
                     'including_shutdown_seconds':time.monotonic()-begin,'cases_per_second':24/outputs_done,
                     'worker_prepare_cpu_seconds':sum(r['prepare_cpu_seconds'] for r in rows),
                     'max_sampled_parent_plus_children_rss_bytes':max(samples),
                     'rss_note':'sum of sampled RSS includes shared pages; not peak unique physical RAM'}
                assert len(rows)==24
                runs.append(run);p.write_new(RECORD/f'RUN_{repeat}_{workers}.json',run)
        assert calls[0]==792 and len(runs)==4 and len(baseline)==24
        speedups={str(r):next(v['batch_wall_seconds'] for v in runs if v['repeat']==r and v['workers']==1)/
                  next(v['batch_wall_seconds'] for v in runs if v['repeat']==r and v['workers']==4) for r in range(2)}
        result={'status':'completed','created_utc':datetime.now(timezone.utc).isoformat(),'runs':runs,'speedups':speedups,
                'all_gates_passed':all(v>=1.5 for v in speedups.values()),'case_forwards':calls[0],
                'input_exact':96,'output_exact':96,'seconds':time.monotonic()-started,
                'hardware':torch.cuda.get_device_name(device),'preaccess_sha256':p.sha(RECORD/'PREACCESS.json'),
                'access':{'fit_images':True,'labels':False,'development':False,'test':False,'outer':False,'confirmation':False,'external':False}}
        p.write_new(RECORD/'RAW_BENCHMARK.json',result)
        print(json.dumps({k:v for k,v in result.items() if k!='runs'},indent=2),flush=True)
    except BaseException as error:
        p.write_new(RECORD/'FAILURE.json',{'error':repr(error),'completed_runs':runs,'case_forwards':calls[0],
                                        'labels_opened':False,'test_access':False})
        raise
    finally:hook.remove()


if __name__=='__main__':main()
