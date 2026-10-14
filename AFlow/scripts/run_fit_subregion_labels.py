"""Fit-only original-class cache with independent native-index reconstruction."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import time
import traceback
import nibabel as nib
import numpy as np
import torch
from sammed3d.subregion_labels import regions,torch_sample,numpy_sample
from scripts import structured_background_protocol as p
from scripts.publish_research_card import publish

REC=p.ROOT/'archive/research_records/2026-09-10_fit_subregion_labels'
CACHE=Path('/tmp/aflow-fit-subregions-20260910')
AUX_SHA='cb7a89d933e7fd3f21250ba002b9c082c80018274b800c5f820ea7eed8be9554'
CFG={}


def init(config):
    torch.set_num_threads(1);CFG.update(config)
    CFG['wt']=np.memmap(p.CACHE/'labels.u8',mode='r',dtype=np.uint8,shape=(972,128,128,128))
    CFG['rc']=np.memmap(config['rc_path'],mode='r',dtype=np.uint8,shape=(684,128,128,128))
    if config['stage']=='build':CFG['fd']=os.open(CACHE/'classes.u8',os.O_WRONLY)
    else:CFG['new']=np.memmap(CACHE/'classes.u8',mode='r',dtype=np.uint8,shape=(684,128,128,128))


def worker(task):
    slot,row,geometry,old_slot,detail=task
    p.checked(Path(row['label_path']),detail['source_label_sha256'])
    nii=nib.as_closest_canonical(nib.load(row['label_path']));raw=np.asarray(nii.dataobj)
    assert list(raw.shape)==geometry['native_shape'] and nib.aff2axcodes(nii.affine)==('R','A','S')
    assert np.isfinite(raw).all() and np.array_equal(raw,np.rint(raw)) and set(np.unique(raw))<={0,1,2,3,4}
    assert all(o>=0 and e>0 and o+e<=n for o,e,n in zip(geometry['origin'],geometry['extent'],raw.shape))
    if CFG['stage']=='build':
        classes=torch_sample(raw,geometry['origin'],geometry['extent'])
    else:
        classes=np.array(CFG['new'][slot])
        expected=numpy_sample(raw,geometry['origin'],geometry['extent'])
        assert np.array_equal(classes,expected),'independent original-class reconstruction'
    assert classes.shape==(128,128,128) and classes.dtype==np.uint8
    masks=regions(classes);old_wt=CFG['wt'][old_slot];old_rc=CFG['rc'][slot]
    assert hashlib.sha256(old_rc.tobytes()).hexdigest()==detail['rc_array_sha256'],'RC slot identity'
    assert np.array_equal(masks['whole_tumor'],old_wt>0)
    assert np.array_equal(masks['resection_cavity'],old_rc>0)
    assert np.all(masks['enhancing_tissue']<=masks['tumor_core']) and np.all(masks['tumor_core']<=masks['whole_tumor'])
    assert not (masks['whole_tumor']&masks['resection_cavity']).any() and masks['whole_tumor'].any()
    payload=classes.tobytes()
    if CFG['stage']=='build':assert os.pwrite(CFG['fd'],payload,slot*128**3)==len(payload)
    return {'case_id':row['case_id'],'subject_id':row['subject_id'],'source_label_sha256':detail['source_label_sha256'],'cached_class_sha256':hashlib.sha256(payload).hexdigest(),'class_counts':np.bincount(classes.ravel(),minlength=5).tolist(),'region_voxels':{k:int(v.sum()) for k,v in masks.items()},'old_wt_byte_exact':True,'old_rc_byte_exact':True}


def setup():
    fit,_=p.check_roles();p.checked(p.CACHE/'index.json',p.INDEX_SHA)
    path=p.RECORD/'AUX_CACHE_RECEIPT.json';p.checked(path,AUX_SHA);aux=json.loads(path.read_text())
    assert aux['fit_sha256']==p.FIT_SHA and aux['cases']==684 and aux['subjects']==313 and aux['all_wt_byte_exact']
    p.checked(p.CACHE/'labels.u8',aux['source_labels_sha256']);p.checked(aux['rc_file'],aux['rc_sha256'])
    details={r['case_id']:r for r in aux['details']};assert set(details)=={r['case_id'] for r in fit}
    index=json.loads((p.CACHE/'index.json').read_text());lookup={c:i for i,c in enumerate(index['case_ids'])}
    tasks=[(i,row,index['crop_geometry'][row['case_id']],lookup[row['case_id']],details[row['case_id']]) for i,row in enumerate(fit)]
    return fit,aux,tasks


def summarize(rows):
    counts=np.asarray([r['class_counts'] for r in rows],dtype=np.int64)
    assert len(rows)==684 and counts.shape==(684,5) and np.all(counts.sum(1)==128**3)
    return {'class_voxels_total':counts.sum(0).tolist(),'cases_containing_class':(counts>0).sum(0).tolist(),'region_voxels_total':{k:sum(r['region_voxels'][k] for r in rows) for k in rows[0]['region_voxels']},'cases_containing_region':{k:sum(r['region_voxels'][k]>0 for r in rows) for k in rows[0]['region_voxels']}}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['build','verify'],required=True);args=ap.parse_args()
    fit,aux,tasks=setup();stage=args.stage
    if stage=='build':
        if (REC/'BUILD_PREACCESS.json').exists() or CACHE.exists():raise FileExistsError('build already attempted')
        assert shutil.disk_usage('/tmp').free>3*1024**3
        code=['scripts/run_fit_subregion_labels.py','sammed3d/subregion_labels.py','tests/test_subregion_labels.py','sammed3d/spatial_roundtrip_audit.py','scripts/structured_background_protocol.py']
        p.write_new(REC/'BUILD_PREACCESS.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'files':{n:p.sha(p.ROOT/n) for n in code},'idea_sha256':p.sha(REC/'IDEA.md'),'prereg_sha256':p.sha(REC/'PREREG.md'),'fit_sha256':p.FIT_SHA,'source_index_sha256':p.INDEX_SHA,'aux_receipt_sha256':AUX_SHA,'source_wt_sha256':aux['source_labels_sha256'],'source_rc_sha256':aux['rc_sha256'],'source_label_sha256':{r['case_id']:r['source_label_sha256'] for r in aux['details']},'scans':684,'subjects':313,'workers':4,'current_blur_training_unchanged':True,'new_model_forwards':0,'development_label_arrays':False,'test_access':False})
        CACHE.mkdir(exist_ok=False);fd=os.open(CACHE/'classes.u8',os.O_CREAT|os.O_EXCL|os.O_RDWR,0o600);os.ftruncate(fd,684*128**3);os.close(fd)
    else:
        if (REC/'RESULT.json').exists():raise FileExistsError('already verified')
        receipt=json.loads((REC/'CACHE_RECEIPT.json').read_text());p.checked(CACHE/'classes.u8',receipt['class_cache_sha256']);p.checked(CACHE/'index.json',receipt['index_sha256']);p.checked(REC/'RAW_BUILD.jsonl',receipt['raw_build_sha256'])
        pre=json.loads((REC/'BUILD_PREACCESS.json').read_text())
        for n,d in pre['files'].items():p.checked(p.ROOT/n,d)
        p.checked(REC/'IDEA.md',pre['idea_sha256']);p.checked(REC/'PREREG.md',pre['prereg_sha256'])
    start=time.monotonic();rows=[];output=REC/('RAW_BUILD.jsonl' if stage=='build' else 'RAW_VERIFY.jsonl')
    try:
        with output.open('x') as stream,mp.get_context('spawn').Pool(4,initializer=init,initargs=({'stage':stage,'rc_path':aux['rc_file']},)) as pool:
            for row in pool.imap(worker,tasks):
                rows.append(row);stream.write(json.dumps(row)+'\n');stream.flush()
                if len(rows)%100==0:print(f'{stage}: {len(rows)}/684 fit labels',flush=True)
        summary=summarize(rows);seconds=time.monotonic()-start
        assert [(r['case_id'],r['subject_id']) for r in rows]==[(r['case_id'],r['subject_id']) for r in fit]
        if stage=='build':
            fd=os.open(CACHE/'classes.u8',os.O_RDWR);os.fsync(fd);os.close(fd)
            digest=p.sha(CACHE/'classes.u8')
            index={'schema':'aflow.fit_subregion_labels','case_ids':[r['case_id'] for r in fit],'shape':[684,128,128,128],'dtype':'uint8','label_values':[0,1,2,3,4],'regions':{'whole_tumor':[1,2,3],'tumor_core':[1,3],'enhancing_tissue':[3],'resection_cavity':[4]},'fit_sha256':p.FIT_SHA,'class_cache_sha256':digest,'source_cache_index_sha256':p.INDEX_SHA,'data_role':'fit_only_no_model_admission'}
            p.write_new(CACHE/'index.json',index)
            p.write_new(REC/'CACHE_RECEIPT.json',{'status':'built_requires_independent_verification','created_utc':datetime.now(timezone.utc).isoformat(),'cache_path':str(CACHE/'classes.u8'),'class_cache_sha256':digest,'index_path':str(CACHE/'index.json'),'index_sha256':p.sha(CACHE/'index.json'),'raw_build_sha256':p.sha(output),'build_preaccess_sha256':p.sha(REC/'BUILD_PREACCESS.json'),'bytes':684*128**3,'compute_seconds_excluding_preaccess_hashes':seconds,'summary':summary,'new_forwards':0,'new_training':False})
        else:
            old=[json.loads(line) for line in (REC/'RAW_BUILD.jsonl').read_text().splitlines()]
            assert rows==old,'all original-class bytes and counts must independently match'
            assert summary==receipt['summary'];p.checked(REC/'BUILD_PREACCESS.json',receipt['build_preaccess_sha256'])
            result={'status':'fit_only_label_engineering_verified','created_utc':datetime.now(timezone.utc).isoformat(),'cohort':{'scans':684,'subjects':313},'summary':summary,'old_wt_byte_exact_cases':684,'old_rc_byte_exact_cases':684,'independent_native_index_exact_cases':684,'cache':receipt,'raw_verify_sha256':p.sha(output),'cache_receipt_sha256':p.sha(REC/'CACHE_RECEIPT.json'),'verify_compute_seconds_excluding_prechecks':seconds,'new_training':False,'new_model_forwards':0,'current_blur_recipe_and_targets_unchanged':True,'development_or_test_label_arrays':False,'next_decision':'cache ready for future separately registered region-encoding controls; no performance claim'}
            p.write_new(REC/'RESULT.json',result);p.write_new(REC/'VERIFICATION.json',{'status':'passed_independent_native_class_reconstruction','result_sha256':p.sha(REC/'RESULT.json'),'source_and_cache_hashes_recomputed':True,'all_cases':684,'verifier_sha256':p.sha(__file__),'new_model_forwards':0})
            text=f'''## 保留原始子区标签的fit数据视图 ✅ 工程完成，未训练新模型

**一句话：** 已保留现有128栅格中的原始类别信息，供未来严格比较区域编码。

**出发点：** 旧开发漏画约92%来自标签2，而错误最高点约38%—65%在切除腔；仅WT/RC二值视图不能直接研究前景内部类别。

**具体做法：** fit684/313原生0—4标签按原image-only crop/nearest-exact128保存uint8；WT=[1,2,3]、TC=[1,3]、ET=3、RC=4。当前模糊的原标签/模型/损失/采样完全不改。Torch构建后用NumPy中心索引独立重建全部病例。

**结果：** 684/684源SHA、旧WT/RC字节和独立完整类数组匹配。包含各类0—4的病例数={summary['cases_containing_class']}；各区域病例数={summary['cases_containing_region']}。这是数据工程结果，没有新模型Dice/Point或CI。

**是否支持直觉：** 支持能无损保留当前栅格类别信息，不证明区域编码会提高指标、不恢复原生缩放细节。旧RC辅助0.1负结果保留，后续编码模型需另注册匹配对照；不能照抄nnU-Net重新编号示例的核心标签。

**代价/下一步：** /tmp缓存1.336GiB，4 CPU；构建主体{receipt['compute_seconds_excluding_preaccess_hashes']:.1f}秒、独立重建主体{seconds:.1f}秒（均不含前置哈希核查），0 GPU/模型前向。无dev/test标签新读取，当前模糊训练不接入新cache。[七问](archive/research_records/2026-09-10_fit_subregion_labels/IDEA.md)、[结果](archive/research_records/2026-09-10_fit_subregion_labels/RESULT.json)、[缓存收据](archive/research_records/2026-09-10_fit_subregion_labels/CACHE_RECEIPT.json)、[独立验证](archive/research_records/2026-09-10_fit_subregion_labels/VERIFICATION.json)。
'''
            with (REC/'RESULT.md').open('x') as f:f.write(text.replace('](archive/','](../../../archive/'))
            (REC/'CARD.md').write_text(text);publish(str(REC.relative_to(p.ROOT)))
        print(json.dumps({'stage':stage,'cases':684,'seconds':seconds,'summary':summary},indent=2),flush=True)
    except BaseException as error:
        p.write_new(REC/(stage.upper()+'_FAILURE.json'),{'error':repr(error),'traceback':traceback.format_exc(),'saved_fit_rows':len(rows),'additional_fit_worker_prefetch_may_have_occurred':True,'new_forwards':0,'development_or_test_label_arrays':False});raise


if __name__=='__main__':main()
