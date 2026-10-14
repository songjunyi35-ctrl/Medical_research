"""Frozen fit-only file-to-model-grid timing; labels are never opened."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import statistics
import time
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from scripts import structured_background_protocol as p
from scripts.build_t1ce_wt_cache import n4_correct
from sammed3d.t1ce_wt_training import _load_canonical_nifti
from sammed3d.t1ce_binary import prepare_t1ce_inference_array
from sammed3d.t1ce_wt_capg3d import AutomaticPromptLocator
from sammed3d.automatic_fixed_workflow import online_logits
from scripts.publish_research_card import publish

RECORD=p.ROOT/'archive/research_records/2026-09-11_workflow_file_to_output_timing'
FINAL=p.ROOT/'archive/research_records/2026-09-11_automatic_workflow_confirmation/FINAL_WORKFLOWS.json'


def prepare(row):
    loaded=_load_canonical_nifti(Path(row['image_path']),row['case_id']+' T1ce')
    image=np.ascontiguousarray(np.asanyarray(loaded.dataobj))
    spacing=nib.affines.voxel_sizes(loaded.affine)
    corrected=n4_correct(image,2)
    nonzero=np.nonzero(corrected>0)
    if any(a.size==0 for a in nonzero):raise ValueError('empty image support')
    origin=[int(a.min()) for a in nonzero];stop=[int(a.max())+1 for a in nonzero]
    crop=corrected[tuple(slice(a,b) for a,b in zip(origin,stop))]
    view=prepare_t1ce_inference_array(row['case_id'],crop,spacing,output_shape=(128,128,128))
    return view.image[None].contiguous()


@torch.inference_mode()
def main():
    if (RECORD/'PREACCESS.json').exists():raise FileExistsError('file timing already started')
    p.checked(p.FIT,p.FIT_SHA);fit=p.rows(p.FIT)[:4]
    models=json.loads(FINAL.read_text())['models']
    source=json.loads((p.ROOT/'archive/research_records/2026-09-11_automatic_workflow_search/DEV_PREACCESS.json').read_text())
    p.checked(p.CACHE/'index.json',p.INDEX_SHA);p.checked(p.CACHE/'images.f32',source['images_sha256'])
    for model in models:p.checked(model['checkpoint'],model['sha256'])
    files=['scripts/benchmark_workflow_file_to_output.py','scripts/build_t1ce_wt_cache.py',
           'sammed3d/t1ce_wt_training.py','sammed3d/t1ce_binary.py','sammed3d/t1ce_wt_capg3d.py',
           'sammed3d/automatic_fixed_workflow.py',str((RECORD/'IDEA.md').relative_to(p.ROOT))]
    p.write_new(RECORD/'PREACCESS.json',{'created_utc':datetime.now(timezone.utc).isoformat(),
        'files':{name:p.sha(p.ROOT/name) for name in files},'models':models,
        'final_workflows_sha256':p.sha(FINAL),'fit_sha256':p.FIT_SHA,
        'cache_index_sha256':p.INDEX_SHA,'cache_images_sha256':source['images_sha256'],
        'sources':[{'case_id':r['case_id'],'path':r['image_path'],'sha256':p.sha(r['image_path'])} for r in fit],
        'expected_model_forwards':147,'expected_preprocessing_runs':16,'labels_opened':False,
        'source_hashing_warms_file_pages':True})
    idx=json.loads((p.CACHE/'index.json').read_text());lookup={case:i for i,case in enumerate(idx['case_ids'])}
    cache=np.memmap(p.CACHE/'images.f32',mode='r',dtype=np.float32,shape=(len(lookup),128,128,128))
    cached=[torch.from_numpy(np.array(cache[lookup[r['case_id']]],copy=True))[None,None] for r in fit]
    torch.set_num_threads(1);sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1)
    device=torch.device('cuda:0');torch.cuda.set_device(device)
    rows=[];calls=[0];started=time.monotonic()
    def count(model,args,out):calls[0]+=len(args[0])
    try:
        for ordinal,m in enumerate(models):
            model=AutomaticPromptLocator(center_aware=False).to(device).eval()
            ck=torch.load(m['checkpoint'],map_location='cpu',weights_only=False)
            model.load_state_dict(ck['model_state_dict'],strict=True);hook=model.register_forward_hook(count)
            views={'full8':list(range(8)),'pruned':m['medoid']['views']}
            for strategy in views:
                for cpu in cached[:3]:
                    warm=online_logits(model,cpu.to(device),views[strategy]);del warm
            for ci,row in enumerate(fit):
                order=['full8','pruned'] if (ci+ordinal)%2==0 else ['pruned','full8']
                for strategy in order:
                    if time.monotonic()-started>1800:raise TimeoutError('30-minute engineering limit')
                    torch.cuda.synchronize();begin=time.perf_counter()
                    image=prepare(row);prep_done=time.perf_counter()
                    logits=online_logits(model,image.to(device),views[strategy])
                    mask=(logits>0).cpu();point=int(logits.flatten().argmax().cpu());torch.cuda.synchronize()
                    end=time.perf_counter()
                    exact=torch.equal(image,cached[ci])
                    diagnostic={'seed':m['seed'],'case_id':row['case_id'],'strategy':strategy,
                        'preprocessing_seconds':prep_done-begin,'model_stage_seconds':end-prep_done,
                        'total_seconds':end-begin,'input_cache_byte_exact':exact,
                        'input_max_abs_error':float((image-cached[ci]).abs().max()),
                        'image_sha256':hashlib.sha256(image.numpy().tobytes()).hexdigest(),
                        'case_forwards':len(views[strategy])}
                    rows.append(diagnostic)
                    assert exact,'reconstructed image differs from frozen cache'
                    assert torch.isfinite(image).all() and mask.shape==(1,1,128,128,128) and 0<=point<128**3
                    print(json.dumps(diagnostic),flush=True)
                    del image,logits,mask
            hook.remove();del model,ck;torch.cuda.empty_cache()
        assert calls[0]==147 and len(rows)==16
        summary={}
        for m in models:
            seed=m['seed'];subset=[r for r in rows if r['seed']==seed]
            lookup_row={(r['case_id'],r['strategy']):r for r in subset}
            ratios=[lookup_row[(r['case_id'],'full8')]['total_seconds']/lookup_row[(r['case_id'],'pruned')]['total_seconds'] for r in fit]
            summary[str(seed)]={'paired_total_speedup_median':statistics.median(ratios),
                'paired_total_speedup_per_case':ratios,
                'full8_total_mean':statistics.fmean(r['total_seconds'] for r in subset if r['strategy']=='full8'),
                'pruned_total_mean':statistics.fmean(r['total_seconds'] for r in subset if r['strategy']=='pruned'),
                'preprocessing_mean':statistics.fmean(r['preprocessing_seconds'] for r in subset),
                'full8_preprocessing_share':statistics.fmean(r['preprocessing_seconds']/r['total_seconds'] for r in subset if r['strategy']=='full8')}
        body={'status':'completed','created_utc':datetime.now(timezone.utc).isoformat(),'rows':rows,'summary':summary,
              'case_forwards':calls[0],'wall_seconds':time.monotonic()-started,'input_cache_exact_count':16,
              'hardware':torch.cuda.get_device_name(device),'itk_threads':1,'preaccess_sha256':p.sha(RECORD/'PREACCESS.json'),
              'labels_opened':False,'test_access':False,'scope':'warm-page file read through CPU model-grid mask/point; excludes model load/native export'}
        p.write_new(RECORD/'RAW_TIMING.json',body)
        result={k:v for k,v in body.items() if k!='rows'};result['raw_sha256']=p.sha(RECORD/'RAW_TIMING.json')
        p.write_new(RECORD/'RESULT.json',result)
        lines=['# 文件读取到模型网格输出：耗时边界','',
               '仅fit前4张图像，两模型、两策略，共16次完整重新预处理。没有标签或精度评价。','',
               '| 模型seed | Full8平均总秒 | Pruned平均总秒 | 配对总加速中位数 | Full8预处理占比 |',
               '|---|---:|---:|---:|---:|']
        for seed,s in summary.items():lines.append(f"| {seed} | {s['full8_total_mean']:.4f} | {s['pruned_total_mean']:.4f} | {s['paired_total_speedup_median']:.4f}× | {s['full8_preprocessing_share']:.2%} |")
        lines.extend(['','逐次重建128影像16/16与历史缓存字节完全一致，输入未改变。只改full8/pruned视图；模型、N4、crop、归一化和输出规则固定。','',
            '这是NIfTI读取至CPU模型网格输出的总时间，不包含权重加载、原生空间导出/写盘或临床流程。源SHA检查暖了文件页缓存，不称冷I/O；四例探针不能给群体显著性。模型阶段3.96/2.65倍不能移作本口径倍率。','',
            '若总加速接近1，说明N4等预处理占主导；不否定已经缓存图像场景的推理收益。这里不据耗时改N4参数或删除预处理，也不回流改变已结束测试。','',
            f"0训练，147模型前向，16次N4，墙钟{body['wall_seconds']:.1f}秒，无大cache或新权重；0标签/开发/test/旧outer/confirmation/外部访问。",'',
            '[七问](IDEA.md)、[冻结](PREACCESS.json)、[逐例计时](RAW_TIMING.json)、[结果JSON](RESULT.json)、[核验](VERIFICATION.json)。'])
        with (RECORD/'RESULT.md').open('x') as f:f.write('\n'.join(lines)+'\n')
        for r in rows:assert abs(r['preprocessing_seconds']+r['model_stage_seconds']-r['total_seconds'])<1e-9
        p.write_new(RECORD/'VERIFICATION.json',{'status':'passed','raw_sha256':p.sha(RECORD/'RAW_TIMING.json'),
            'result_sha256':p.sha(RECORD/'RESULT.json'),'markdown_sha256':p.sha(RECORD/'RESULT.md'),
            'input_exact_rows':16,'counted_forwards':147,'preaccess_sha256':p.sha(RECORD/'PREACCESS.json'),
            'scope':'accounting/hash/byte parity; wall-clock measurements are not independent reruns'})
        before=(RECORD/'CARD.md').read_bytes()
        with (RECORD/'CARD.md.BEFORE_RESULT').open('xb') as f:f.write(before)
        card='## 文件读取到输出总耗时检查完成：阶段加速不能冒充总加速\n\n'
        card+='动机：N4等预处理可能主导总时长。fit前4图像、两模型两策略16次完整流程；仅改镜像集合，16/16重建输入与旧cache字节exact，其他因素不变。\n\n'
        for seed,s in summary.items():card+=f"Seed {seed}：full8/pruned平均总秒{s['full8_total_mean']:.4f}/{s['pruned_total_mean']:.4f}，配对总加速中位{s['paired_total_speedup_median']:.4f}倍，full8预处理占{s['full8_preprocessing_share']:.2%}。\n\n"
        card+='计时为暖页文件读取到128³CPU输出，不含模型加载/原生导出/临床其他步骤；四例无群体CI。不能拿模型阶段3.96/2.65倍替代这里。0训练/标签/开发或测试性能访问，147前向、16次N4，无新大cache/权重；不据此改已结束方法。\n\n'
        card+='[七问](archive/research_records/2026-09-11_workflow_file_to_output_timing/IDEA.md)、[结果](archive/research_records/2026-09-11_workflow_file_to_output_timing/RESULT.md)、[原始计时](archive/research_records/2026-09-11_workflow_file_to_output_timing/RAW_TIMING.json)、[核验](archive/research_records/2026-09-11_workflow_file_to_output_timing/VERIFICATION.json)。\n'
        assert (RECORD/'CARD.md').read_bytes()==before
        (RECORD/'CARD.md').write_text(card);publish(str(RECORD.relative_to(p.ROOT)))
        print(json.dumps(summary,indent=2),flush=True)
    except BaseException as error:
        p.write_new(RECORD/'FAILURE.json',{'error':repr(error),'partial_rows':rows,'case_forwards':calls[0],
            'labels_opened':False,'test_access':False,'preaccess_sha256':p.sha(RECORD/'PREACCESS.json')})
        raise


if __name__=='__main__':main()
