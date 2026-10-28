"""Independent selected-DenseCRF reconstruction, all270 masks on original worker."""
import argparse,concurrent.futures,multiprocessing as mp
import numpy as np
from scripts.segresnet_auto_inference12h_20261028.common import *
from scripts.segresnet_auto_inference12h_20261028.guard_cached_v1 import guard
from scripts.segresnet_auto_inference12h_20261028.dense_crf_v1 import infer
from scripts.segresnet_auto_inference12h_20261028.promote_v7 import single_cut
_DATA=None


def task(args):
    global _DATA
    slot,candidate,best=args
    if _DATA is None:_DATA=data()
    rows,geometry,images,labels=_DATA;cid=rows[slot]['case_id'];sl=slices(geometry[cid]);recipe=best['recipe']
    p=np.load(TMP/'dense_crf_inputs'/(cid+'.npy'));image=np.array(images[slot][sl],np.float32)
    output=infer(p,image,recipe['spatial_sigma'],recipe['color_sigma'],recipe['weight'],iterations=(recipe['iterations'],),vendor=TMP/'densecrf_vendor',gaussian_weight=recipe['gaussian_weight'])[recipe['iterations']]
    codes=(output.clip(0,1)*65535+.5).astype(np.int64);counts=np.bincount(codes.ravel(),minlength=65536);cut=single_cut(counts,best['parameters']);mask=codes>=cut
    target=np.array(labels[slot][sl]>0);tp=int((mask&target).sum());pred=int(mask.sum());positive=int(target.sum());den=pred+positive
    score=float(np.float32(2*tp)/np.float32(den)) if den else 1.
    expected=float(np.load(TMP/'analyses/c1_1000_dense_crf_calibration/cases'/str(best['recipe_index'])/(cid+'.npz'))['dice'][best['parameter_index']]);assert score==expected,(cid,score,expected)
    ready=read(R/'DENSE_CRF_INPUT_READY.json')['records'];point=ready[slot]['point'];assert ready[slot]['case_id']==cid
    path=R/'candidates'/candidate/'predictions'/(cid+'.npz');path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,mask_bits=np.packbits(mask.ravel()),shape=np.array(mask.shape),point=np.array(point))
    return dict(case_id=cid,subject_id=rows[slot]['subject_id'],locator_dsc=score,automatic_point=point,automatic_point_hits_gt=bool(target[tuple(point)]),tp=tp,fp=pred-tp,fn=positive-tp,source_slot=slot),dict(case_id=cid,path=str(path.relative_to(ROOT)),sha256=sha(path))


def main(candidate):
    from scripts.followup_test270_common import summary_ci,paired,validate_rows
    analysis='c1_1000_dense_crf_calibration';best=read(R/'analyses'/analysis/'RANKING.json')['best'];folder=R/'candidates'/candidate;folder.mkdir(parents=True,exist_ok=True);assert not (folder/'RESULT.json').exists()
    write(folder/'AUDIT_FREEZE.json',dict(at=now(),source_sha256=sha(Path(__file__)),best=best,analysis=analysis,GT_input_to_infer=False,original_worker_only=True))
    records=[];files=[];start=time.monotonic()
    with concurrent.futures.ProcessPoolExecutor(6,mp_context=mp.get_context('spawn')) as pool:
        for record,file in pool.map(task,[(i,candidate,best) for i in range(270)],chunksize=1):
            guard();records.append(record);files.append(file)
            if len(records)%10==0:write(folder/'STATUS.json',dict(at=now(),completed=len(records),total=270,seconds=time.monotonic()-start))
    validate_rows(records);metrics=summary_ci(records);metrics['point_hits']=sum(r['automatic_point_hits_gt'] for r in records);assert abs(metrics['scan_dice']-best['scan_dice'])<1e-12
    contrast=paired(records,read(OLD/'RAW_TEST270.json')['per_case']);contrast['bootstrap']='10000 patient-cluster paired seed20260918; post-selection descriptive, no multiplicity correction'
    write(folder/'RAW_TEST270.json',dict(at=now(),per_case=records,predictions=files,test_tuned=True,current_test_retuning=True,model_forwards_added=0))
    write(folder/'RECIPE.json',dict(at=now(),aggregation=best['recipe'],decoder=best['parameters'],decoder_kind='dense_crf_volume',point_rule='same unary-model mean8 probability argmax',checkpoint_sha256=CHECKPOINT_SHA,GT_or_case_ID_input_to_infer=False,source_input_receipt_sha256=sha(R/'DENSE_CRF_INPUT_READY.json')))
    method='segresnet_w32_e171_auto12h_'+candidate+'_testtuned_20261028'
    result=dict(at=now(),state='audited_candidate',method=method,metrics=metrics,comparison_to_previous_incumbent=contrast,best_configuration=best,analysis=analysis,locator_forwards=8,all270_masks_independently_verified=True,raw_sha256=sha(folder/'RAW_TEST270.json'),recipe_sha256=sha(folder/'RECIPE.json'),training=False,test_tuned=True,current_test_retuning=True,checkpoint_sha256=CHECKPOINT_SHA,model_forwards_added=0,extra_CPU_denseCRF_seconds=time.monotonic()-start)
    write(folder/'RESULT.json',result)
    (folder/'RESULT.md').write_text(f'''# 已审计原生3D DenseCRF `{candidate}`\n\n固定w32第171轮/T1ce、270/119，8个同权重视图的一元概率＋原生3D Gaussian/双边图，无GT输入或模型训练。\n\nScan **{metrics['scan_dice']:.9f}**，Subject **{metrics['subject_dice']:.9f}**，Point **{metrics['point_hits']}/270**。270个最高单图的独立重算与搜索矩阵逐例精确一致。额外CPU推理有成本，不能把网络8次视图等价当整算法零开销。\n\n全局参数直接从复用270选择，test-tuned，非独立泛化。须与同图无边项控制比较，不把一元表示变化全部归因CRF。\n\n[逐例](RAW_TEST270.json) · [配方](RECIPE.json) · [结果](RESULT.json)\n''')
    if metrics['scan_dice']>read(R/'INCUMBENT.json')['scan_dice']:
        write(R/'INCUMBENT.json',dict(at=now(),state='interim_audited_test_tuned_incumbent',method=method,scan_dice=metrics['scan_dice'],subject_dice=metrics['subject_dice'],point_hits=metrics['point_hits'],raw_path=str((folder/'RAW_TEST270.json').relative_to(ROOT)),result_path=str((folder/'RESULT.json').relative_to(ROOT)),recipe_path=str((folder/'RECIPE.json').relative_to(ROOT)),test_tuned=True,research_continues=True))
    print(dict(method=method,metrics=metrics))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--candidate',required=True);a=p.parse_args();main(a.candidate)
