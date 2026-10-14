"""Validation archival, one-shot test, and a preregistered conditional six-output fusion panel."""
import math
import multiprocessing as mp
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.nnunet_resenc_20261013.common import *
os.environ.update(env())
import numpy as np
import torch
from scripts.nnunet_resenc_20261013.trainer_v2 import AFlowTrainer,restore_margin,storage_guard
from scripts.nnunet_resenc_20261013.publish import progress,checks,leaderboard,replace_block
from scripts.segresnet_plain972_common import summary_ci
from scripts.segresnet_two_ideas_common import paired

def deadline():
    if time.time()>datetime.datetime(2026,10,14,12,41,45,tzinfo=ZoneInfo('Asia/Shanghai')).timestamp():
        raise RuntimeError('overall deadline reached; do not continue evaluation')
    disk()

def new_access(path,obj):
    with Path(path).open('x') as f:json.dump(obj,f,ensure_ascii=False,indent=2)

def metrics_row(cid,sid,margin,target,threshold=.5):
    pred=margin>math.log(threshold/(1-threshold));target=target>0
    tp=int((pred&target).sum());fp=int((pred&~target).sum());fn=int((~pred&target).sum());den=2*tp+fp+fn
    point=np.unravel_index(int(margin.argmax()),margin.shape)
    return {'case_id':cid,'subject_id':sid,'locator_dsc':float(np.float32(2*tp)/np.float32(den)) if den else 1.,
       'automatic_point':[int(x) for x in point],'automatic_point_hits_gt':bool(target[point]),'tp':tp,'fp':fp,'fn':fn}

def test_cache():
    from scripts import build_segresnet_native_res_test_cache as b
    shape=tuple(b.TEST_CANVAS);cache=TMP/'test_native';cache.mkdir(parents=True,exist_ok=True)
    expected=read(ROOT/'archive/research_records/2026-09-20_segresnet_native_res_e173_test270/CACHE_RECEIPT.json')
    cases=rows('test');canon=read(b.CANONICAL_TEST/'index.json')
    assert canon['case_ids']==[r['case_id'] for r in cases]
    new_access(R/'TEST_PIXEL_PREACCESS.json',{'at':now(),'cohort':{'scans':270,'subjects':119},'test_tuned':True,'manifest_sha256':sha(MANIFESTS['test'])})
    n=int(np.prod(shape))
    for name,size in [('images.f32',4),('labels.u8',1)]:
        with (cache/name).open('xb') as f:f.truncate(270*n*size)
    config={'images':str(cache/'images.f32'),'labels':str(cache/'labels.u8'),'canon_dir':str(b.CANONICAL_TEST)}
    geometry={};started=time.monotonic()
    tasks=[(i,r,canon['crop_geometry'][r['case_id']]) for i,r in enumerate(cases)]
    with mp.get_context('spawn').Pool(24,initializer=b.init,initargs=(config,)) as pool:
        for k,(slot,cid,geo,error) in enumerate(pool.imap_unordered(b.worker,tasks,chunksize=1),1):
            if error:raise RuntimeError(error)
            if not all(geo[x] for x in ['canonical_image_within_tolerance','canonical_label_exact','canonical_geometry_exact']):raise RuntimeError('test geometry mismatch')
            geometry[cid]=geo
            if k%30==0:write(R/'TEST_CACHE_PROGRESS.json',{'at':now(),'done':k,'of':270})
    hashes={k:sha(cache/f) for k,f in [('images_sha256','images.f32'),('labels_sha256','labels.u8')]}
    if any(hashes[k]!=expected[k] for k in hashes):raise RuntimeError('test payload differs from historical')
    write(cache/'index.json',{'canvas':list(shape),'geometry':geometry,'case_ids':[r['case_id'] for r in cases]})
    write(R/'TEST_DATA_AUDIT.json',{'at':now(),'status':'passed','historical_payload_exact':True,**hashes,'geometry':geometry,'seconds':time.monotonic()-started})
    return cache,shape,geometry,cases

def report(identifier,cases,cost,formula,threshold,single,old,new_weight=None):
    assert len(cases)==len({r['case_id'] for r in cases})==270
    m=summary_ci(cases);m['point_hits']=sum(r['automatic_point_hits_gt'] for r in cases)
    d={'at':now(),'method':identifier,'metrics':m,'per_case':cases,'threshold':threshold,'single_model':single,
       'forward_equivalents':cost,'formula':formula,'new_model_weight':new_weight,'test_tuned':True,'new_outer_confirmation_access':False,
       'comparisons':{'old_best_panel10':paired(cases,old)}}
    path=R/'reports'/f'{identifier}.json';write(path,d);return path

def main():
    deadline();verify_source_freeze('EXECUTION_FREEZE.json');verify_manifests()
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark=True
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    write(R/'PIPELINE_STATUS.json',{'at':now(),'state':'validation_archival_and_selected_reload','running':True,'new_test_access':False})
    done=read(R/'TRAINING_DONE.json');epoch=done['best_epoch']
    selected=read(OUT/f'validation_e{epoch:04d}.json')
    expected_val={(x['case_id'],x['subject_id']) for x in rows('validation')}
    assert len(selected['per_case'])==108 and {(x['case_id'],x['subject_id']) for x in selected['per_case']}==expected_val
    assert abs(np.mean([x['locator_dsc'] for x in selected['per_case']])-selected['metrics']['scan_dice'])<1e-12
    assert abs(selected['metrics']['scan_dice']-done['best_scan'])<1e-12
    old_validation=read(ROOT/'workspace/BraTS/segresnet_t1ce_wt/native_multiscale_resume50_20260929/validation_e166.json')['summary']['per_case']
    contrast=paired(selected['per_case'],old_validation);m=selected['metrics']
    gate=m['scan_dice']>=.7630921331986233 and contrast['scan_delta_ci95'][0]>0 and m['subject_dice']>=.7396687182784081 and m['point_hits']>=94
    checkpoint=OUT/'model/fold_0/checkpoint_best.pth'
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if state['current_epoch']!=epoch:raise RuntimeError('selected checkpoint epoch mismatch')
    trainer=AFlowTrainer(read(R/'TRAIN972_PLANS.json'),'3d_fullres',0,read(PRE/'dataset.json'),torch.device('cuda',0))
    trainer.initialize();trainer.load_checkpoint(str(checkpoint));trainer.set_deep_supervision_enabled(False)
    trainer.network.eval().requires_grad_(False)
    # Validate full state after strict restoration before any test model forward.
    from scripts.nnunet_resenc_20261013.run_train_v2 import model_hash
    digest=model_hash(trainer.network)
    for key,tensor in trainer.network.state_dict().items():
        if not torch.equal(tensor.detach().cpu(),state['network_weights'][key]):raise RuntimeError('selected state reload mismatch')
    write(R/'RELOAD_CHECK.json',{'at':now(),'status':'passed','checkpoint_sha256':sha(checkpoint),'network_state_sha256':digest,'strict_state_exact':True})
    result={'at':now(),'training':done,'best_validation':m,'validation_comparison':contrast,'validation_gate_passed':gate,
        'checkpoint_sha256':sha(checkpoint),'new_test_access':False,'test_tuned_development':True,'new_outer_confirmation_access':False}
    write(R/'VALIDATION_RESULT.json',result)
    progress('训练与108验证完成',f'选{epoch}轮Scan{m["scan_dice"]:.6f}，验证明确收益门{"通过" if gate else "未通过"}；接续一次授权探索性270。')
    deadline();write(R/'TEST_FREEZE.json',{'at':now(),'checkpoint_sha256':sha(checkpoint),'validation_report_sha256':sha(OUT/f'validation_e{epoch:04d}.json'),
       'threshold':.5,'mirroring':False,'tile_step':.5,'source_freeze_sha256':sha(R/'EXECUTION_FREEZE.json'),'fusion_plan_sha256':sha(R/'FUSION_PLAN.md')})
    write(R/'PIPELINE_STATUS.json',{'at':now(),'state':'building_selected_test_cache','running':True,'test_model_access':False})
    cache,shape,geometry,cases=test_cache();deadline()
    new_access(R/'MODEL_PREACCESS.json',{'at':now(),'checkpoint_sha256':sha(checkpoint),'model':'nnunet_single','cohort_scans':270})
    write(R/'PIPELINE_STATUS.json',{'at':now(),'state':'selected_model_test_inference','running':True,'new_test_access':True})
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    predictor=nnUNetPredictor(tile_step_size=.5,use_gaussian=True,use_mirroring=False,perform_everything_on_device=True,
        device=torch.device('cuda',0),verbose=False,verbose_preprocessing=False,allow_tqdm=False)
    predictor.manual_initialization(trainer.network,trainer.plans_manager,trainer.configuration_manager,None,trainer.dataset_json,'AFlowTrainer',None)
    preprocessor=trainer.configuration_manager.preprocessor_class(verbose=False)
    images=np.memmap(cache/'images.f32',dtype=np.float32,mode='r',shape=(270,*shape))
    labels=np.memmap(cache/'labels.u8',dtype=np.uint8,mode='r',shape=(270,*shape))
    raw_dir=OUT/'test_logits';raw_dir.mkdir(parents=True,exist_ok=True)
    counter=[0]
    def count_hook(module,args):counter[0]+=1
    hook=trainer.network.register_forward_pre_hook(count_hook)
    scored=[];predictions=[];started=time.monotonic()
    for i,row in enumerate(cases):
        deadline();cid=row['case_id'];geo=geometry[cid];sl=tuple(slice(o,o+e) for o,e in zip(geo['canvas_offset'],geo['extent']))
        image=np.asarray(images[(i,*sl)]).copy();target=np.asarray(labels[(i,*sl)])
        data,_,props=preprocessor.run_case_npy(image.transpose(2,1,0)[None],None,{'spacing':list(reversed(geo['spacing_mm']))},
           trainer.plans_manager,trainer.configuration_manager,trainer.dataset_json)
        before=counter[0]
        with torch.inference_mode():logits=predictor.predict_sliding_window_return_logits(torch.from_numpy(data)).cpu()
        margin=restore_margin(logits,trainer.plans_manager,trainer.configuration_manager,props)
        if margin.shape!=target.shape or not np.isfinite(margin).all():raise RuntimeError('invalid native prediction')
        storage_guard(margin.nbytes+4096)
        p=raw_dir/f'{cid}.npz';np.savez_compressed(p,margin=margin)
        scored.append(metrics_row(cid,row['subject_id'],margin,target))
        predictions.append({'case_id':cid,'path':str(p.relative_to(ROOT)),'sha256':sha(p),'windows':counter[0]-before})
        write(R/'TEST_INFERENCE_JOURNAL.json',{'at':now(),'completed_scans':i+1,'per_case':scored,'predictions':predictions})
    hook.remove()
    cost=float(np.mean([p['windows'] for p in predictions])*np.prod(trainer.configuration_manager.patch_size)/128**3)
    old=read(ROOT/'archive/research_records/2026-09-21_t1ce_test270_10h/reports/panel_10.json')['per_case']
    paths=[report('nnunet_resenc_20261013_single',scored,cost,'single residual encoder; sliding-window Gaussian logits',.5,True,old)]
    write(R/'TEST_RUNTIME.json',{'at':now(),'seconds':time.monotonic()-started,'mean_windows':float(np.mean([p['windows'] for p in predictions])),
        'inference_mode':'FP16 batch1 patches; no mirroring; cudnn benchmark and TF32 enabled',
        'timing_includes':'nnU-Net preprocessing, inference, scoring, compressed float32 output persistence; excludes prior N4/cache build',
        'forward_equivalents':cost,'cost_is_not_flops_equivalence':True})
    # Publish the single-model outcome before optional reference/fusion access.
    leaderboard(paths,checkpoint,done)
    old_by={r['case_id']:r for r in old}
    diverse=sum(r['locator_dsc']>old_by[r['case_id']]['locator_dsc']+.03 for r in scored)
    allow=diverse>=15 or np.mean([r['locator_dsc'] for r in scored])>.78
    write(R/'FUSION_DECISION.json',{'at':now(),'allowed':bool(allow),'improved_more_than_003_scans':diverse,
         'rule':'>=15 scans improved >.03 versus old panel10 OR single Scan>.78','test_tuned':True})
    if allow:
        deadline()
        manifest=read(ROOT/'workspace/BraTS/segresnet_t1ce_wt/test270_10h_20260921/deployment/manifest.json')
        from scripts.followup_test270_common import numeric_mode
        numeric_mode()
        from scripts.segresnet_plain972_common import model_for
        refs=[]
        for component in manifest['components']:
            p=ROOT/component['checkpoint_path']
            assert sha(p)==component['checkpoint_sha256']
            model=model_for(torch.device('cuda',1));c=torch.load(p,map_location='cpu',weights_only=False)
            model.load_state_dict(c['model_state_dict'],strict=True);model.eval().requires_grad_(False);refs.append(model)
        new_access(R/'REFERENCE_MODEL_PREACCESS.json',{'at':now(),'components':manifest['components'],'precision':'bf16 batch1, historical reports batch2; do not overwrite old scores'})
        panel={(a,t):[] for a in [.25,.5,.75] for t in [.4,.5]}
        for i,row in enumerate(cases):
            deadline();cid=row['case_id'];geo=geometry[cid];sl=tuple(slice(o,o+e) for o,e in zip(geo['canvas_offset'],geo['extent']))
            x=torch.from_numpy(np.asarray(images[i]).copy())[None,None].cuda(1)
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                outs=[model(x).segmentation_logits[0,0].float().cpu().numpy()[sl] for model in refs]
            old_margin=(outs[0]+outs[1])*.5
            new_margin=np.load(raw_dir/f'{cid}.npz')['margin'];target=np.asarray(labels[(i,*sl)])
            for (a,t),items in panel.items():items.append(metrics_row(cid,row['subject_id'],(1-a)*old_margin+a*new_margin,target,t))
        more=[]
        for (a,t),items in panel.items():
            more.append(report(f'nnunet_resenc_20261013_mix_a{a:.2f}_t{t:.2f}',items,cost+2*np.prod(shape)/128**3,
                 f'{1-a}*mean(native173,combination166) + {a}*nnunet foreground-background logits',t,False,old,new_weight=a))
        leaderboard(more,checkpoint,done);paths+=more
    reports=[read(p) for p in paths];best=max(reports,key=lambda r:(r['metrics']['scan_dice'],r['metrics']['subject_dice'],r['metrics']['point_hits'],r['method']))
    result.update(at=now(),status='complete',test_reports=[str(p.relative_to(ROOT)) for p in paths],new_test_access=True,
        best_new_method=best['method'],best_new_metrics=best['metrics'],target_080_reached=best['metrics']['scan_dice']>=.8,
        old_best_scan=.7758914931930428,improved_old_best=best['metrics']['scan_dice']>.7758914931930428)
    write(OUT/'DEPLOYMENT_CANDIDATE.json',{'at':now(),'research_candidate_only':True,'existing_deployment_replaced':False,
        'selected_best_new_method':best['method'],'checkpoint_path':str(checkpoint.relative_to(ROOT)),
        'checkpoint_sha256':sha(checkpoint),'plans_path':str((OUT/'model/plans.json').relative_to(ROOT)),
        'plans_sha256':sha(OUT/'model/plans.json'),'dataset_json_path':str((OUT/'model/dataset.json').relative_to(ROOT)),
        'inference_implementation':'scripts/nnunet_resenc_20261013/finalize_v2.py (manual predictor initialization)',
        'single_input':'T1ce','preprocessing':'canonical N4 shrink2, image-support native crop, nonzero z-score; frozen nnU-Net preprocessing',
        'threshold':best['threshold'],'new_model_weight':1. if best['single_model'] else best['new_model_weight'],
        'old_components':[] if best['single_model'] else manifest['components'],
        'formula':best['formula'],'mirroring':False,'tile_step_size':.5,'test_tuned':True})
    write(R/'RESULT.json',result)
    b=best['metrics'];text=(f'# 残差编码器单组结果\n\n完成{done["last_complete_epoch"]}/{done["target_epochs"]}轮，最终保留轨迹的实际优化器更新{done["actual_optimizer_steps"]}；108选第{epoch}轮Scan {m["scan_dice"]:.6f}，验证门{"通过" if gate else "未通过"}。工程恢复与丢弃工作以逐次启动/失败收据及训练日志为准，不冒充逐位同轨迹。\n\n'
       f'270/119原生1mm本批最高{best["method"]}：Scan {b["scan_dice"]:.9f}、Subject {b["subject_dice"]:.9f}、Point {b["point_hits"]}/270，Scan描述性95%区间{b["scan_ci95"]}。达到0.8={result["target_080_reached"]}。旧榜首0.775891保留；新组不一定超过旧组。\n\n'
       f'全部test-tuned，非独立泛化；没有新outer/confirmation。单模型滑窗与条件三模型融合成本分别列账；体素前向等价不代表FLOPs或延迟相同。\n\n[机器结果](RESULT.json) · [协议](IDEA.md) · [固定融合范围](FUSION_PLAN.md) · [重载](RELOAD_CHECK.json)\n')
    (R/'RESULT.md').write_text(text)
    detail=f'108选{epoch}轮Scan{m["scan_dice"]:.6f}；本批270最高Scan{b["scan_dice"]:.6f}、Subject{b["subject_dice"]:.6f}、Point{b["point_hits"]}/270，全部test-tuned，0.8目标{"达到" if result["target_080_reached"] else "未达到"}。'
    progress('完成',detail);replace_block(ROOT/'AFLOW_BRATS_WEEKLY_REPORT.md',detail+f' [完整结果]({str(R.relative_to(ROOT))}/RESULT.md)。')
    catalog=ROOT/'AFLOW_BRATS_METHOD_CATALOG.md';catalog_text=catalog.read_text()
    if reports[0]['metrics']['scan_dice']>.772829645:
        catalog_text=catalog_text.replace('已测单模型最高点估计；8块推理','旧SegResNet单模型最高点估计；8块推理')
    if result['improved_old_best']:
        catalog_text=catalog_text.replace('已测总体最高点估计；2模型/2次原生前向','本组前总体最高点估计；2模型/2次原生前向')
        catalog_text=catalog_text.replace('是同队列已测方案的总体最高点估计','是本组开展前的同队列总体最高点估计')
    anchor='多尺度166也受既往测试启发；'
    if anchor in catalog_text:
        row=f'| 原生1mm | 本批残差编码器最佳输出 | **{b["scan_dice"]:.6f}** | {b["point_hits"]}/270 | 单模型或固定三模型融合见方法8；test-tuned，滑窗成本单列 |\n\n'
        catalog_text=catalog_text.replace('\n\n'+anchor,'\n'+row+anchor,1)
    catalog.write_text(catalog_text)
    checks();write(R/'COMPLETION_AUDIT.json',{'at':now(),'status':'passed','result_sha256':sha(R/'RESULT.json'),'checkpoint_sha256':sha(checkpoint),'test_scans':270,'target_reached':result['target_080_reached']})
    write(R/'PIPELINE_STATUS.json',{'at':now(),'state':'complete','running':False,'new_test_access':True,'result':str((R/'RESULT.json').relative_to(ROOT))})

if __name__=='__main__':
    try:main()
    except BaseException as e:
        write(R/'FINALIZATION_FAILURE.json',{'at':now(),'error':repr(e),'test_pixels_access':(R/'TEST_PIXEL_PREACCESS.json').exists(),'test_model_access':(R/'MODEL_PREACCESS.json').exists()});raise
