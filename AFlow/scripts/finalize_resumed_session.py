"""Audit now; write the final eight-hour report only after the authorized deadline."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import statistics
import subprocess
import sys
from scripts.structured_background_protocol import ROOT,sha,checked,write_new
from scripts.require_development_mask_verification import require_masks
from scripts.publish_research_card import publish

REC=ROOT/'archive/research_records/2026-09-10_resumed_eight_hour'
START=datetime(2026,9,10,13,39,55,tzinfo=timezone.utc)
END=datetime(2026,9,10,21,39,55,tzinfo=timezone.utc)
STUDIES=('small_rotation','image_blur')


def collect():
    methods={};checkpoints={};sources={};training_hours=0.;dev_forwards=0;mask_count=0
    for study in STUDIES:
        path=ROOT/f'archive/research_records/2026-09-10_{study}'
        proof=require_masks(path);r=json.loads((path/'RESULT.json').read_text())
        assert r['selected_arm'] is None and r['status']=='completed_reused_development_gate_failed'
        assert not (path/'TEST_PREACCESS.json').exists() and not (path/'RAW_TEST270.json').exists()
        raw=json.loads((path/'RAW_DEVELOPMENT.json').read_text());dev_forwards+=raw['new_forwards'];mask_count+=proof['actual_masks_checked']
        methods[study]={'status':r['status'],'metrics':r['metrics'],'comparisons':r['comparisons'],'gates':r['gates'],'arm_gates':r['arm_gates'],'selected_arm':None,'result_path':str((path/'RESULT.json').relative_to(ROOT)),'result_sha256':sha(path/'RESULT.json'),'mask_verification_sha256':sha(path/'DEV_MASK_VERIFICATION.json')}
        for key,c in r['checkpoints'].items():
            checked(ROOT/c['path'],c['sha256']);training_hours+=c['seconds']/3600;checkpoints[key]=c
        for name in ['IDEA.md','PREREG.md','TRAIN_PREACCESS.json','IMPLEMENTATION_FREEZE.json','DEVELOPMENT_IMPLEMENTATION_FREEZE.json','DEV_PREACCESS.json','RAW_DEVELOPMENT.json','RESULT.json','RESULT.md','VERIFICATION_RECEIPT.json','TRAINING_REPLAY_FINAL.json','DEV_MASK_PREACCESS.json','DEV_MASK_INDEX.json','DEV_MASK_VERIFICATION.json','INTERPRETATION.json','INTERPRETATION.md','ERROR_CHANGE_DESCRIPTION.json','ERROR_CHANGE_DESCRIPTION.md']:
            file=path/name;sources[str(file.relative_to(ROOT))]=sha(file)
    assert len(checkpoints)==6 and mask_count==1080 and dev_forwards==1080
    full=ROOT/'archive/research_records/2026-09-10_small_rotation_full864';unadmitted=json.loads((full/'RESULT.json').read_text())
    assert unadmitted['status']=='not_admitted_parent_development_gates_failed' and not unadmitted['former_development180_consumed_as_training']
    assert not (full/'TRAINING_STARTED.json').exists() and not (full/'TEST_PREACCESS.json').exists()
    sources[str((full/'RESULT.json').relative_to(ROOT))]=sha(full/'RESULT.json')
    diagnostics={}
    for name in ['letterbox_fidelity','development_error_anatomy','fit_subregion_labels']:
        path=ROOT/f'archive/research_records/2026-09-10_{name}';r=json.loads((path/'RESULT.json').read_text());v=json.loads((path/'VERIFICATION.json').read_text())
        checked(path/'RESULT.json',v['result_sha256'])
        diagnostics[name]={'result_path':str((path/'RESULT.json').relative_to(ROOT)),'result_sha256':sha(path/'RESULT.json'),'verification_path':str((path/'VERIFICATION.json').relative_to(ROOT)),'verification_sha256':sha(path/'VERIFICATION.json'),'status':r['status']}
        sources[str((path/'RESULT.md').relative_to(ROOT))]=sha(path/'RESULT.md')
        if name=='development_error_anatomy':
            assert r['new_forwards']==720 and not r['test_access'];dev_forwards+=720
            sources[str((path/'RAW_DEVELOPMENT.json').relative_to(ROOT))]=sha(path/'RAW_DEVELOPMENT.json')
        if name=='fit_subregion_labels':
            assert r['independent_native_index_exact_cases']==684 and not r['new_training']
            checked(r['cache']['cache_path'],r['cache']['class_cache_sha256']);checked(r['cache']['index_path'],r['cache']['index_sha256'])
            diagnostics[name]['cache']=r['cache']
    board=json.loads((ROOT/'AFLOW_BRATS_LEADERBOARD.json').read_text());rows=board['test270_direct_locator_rank_by_scan']
    best=rows[0];one=next(r for r in rows if r.get('locator_forwards')==1)
    assert best['scan_dice']==.7572696616942132 and one['scan_dice']==.7520031755617648
    for name in ['POINT_RECORD_CORRECTION.json','POINT_RECORD_CORRECTION.md','RESEARCH_SYNTHESIS.md','EXECUTION_ORDER.md','SESSION.md','ENGINEERING_NOTES.md']:
        sources[str((REC/name).relative_to(ROOT))]=sha(REC/name)
    return {'methods':methods,'new_checkpoints':checkpoints,'new_training_runs':6,'new_training_allocated_gpu_hours':training_hours,'new_formal_development_forwards':dev_forwards,'new_candidate_development_forwards':1080,'old_model_development_diagnostic_forwards':720,'actual_candidate_masks_verified':1080,'old_diagnostic_masks_verified':720,'new_test_forwards':0,'new_test_access':False,'former_development180_consumed_as_training':False,'new_confirmation_outer_fixed_validation_external_performance':False,'diagnostics':diagnostics,'unadmitted_full864':unadmitted,'old_overall_test270':best,'old_one_forward_test270':one,'leaderboard_sha256':sha(ROOT/'AFLOW_BRATS_LEADERBOARD.json'),'leaderboard_markdown_sha256':sha(ROOT/'AFLOW_BRATS_LEADERBOARD.md'),'source_sha256':sources,'prior_window_postdeadline_extra5760_not_counted_in_this_window':True,'leaderboard_surpassed':False}


def validators():
    results=[]
    for name in ['verify_aflow_brats_method_catalog.py','verify_aflow_brats_doc_layout.py','verify_aflow_brats_leaderboard.py']:
        run=subprocess.run([sys.executable,'scripts/'+name],cwd=ROOT,capture_output=True,text=True)
        results.append({'command':name,'returncode':run.returncode,'stdout':run.stdout,'stderr':run.stderr})
    assert all(r['returncode']==0 for r in results),results
    return results


def final_markdown(r):
    lines=['## 第二个八小时研究结束：未超越旧榜','',
           f"新窗口2026-09-10 13:39:55—21:39:55 UTC；报告时间{r['finished_utc']}。六个新模型全部固定168轮完成，投入约{r['new_training_allocated_gpu_hours']:.3f} GPU小时。旋转与模糊都未过联合开发门，未新增test/outer/confirmation/fixed validation/external性能，不把开发分数冒充超榜。",'',
           '**一句话/动机：** 分别检验整体小角度旋转和标签不变的轻度图像模糊，针对覆盖与错误最高点不同步；结果没有支持固定配方的稳定总体提升。',
           '**改变/固定：** 各比较只改变登记的增强因素；fit684/313、reused dev180/78、采样层、初始化、基础随机流、168轮、主损失、阈值0、raw-logit argmax和1×部署保持控制。历史对照复用，没有宣称完整旧训练轨迹从头重现。','',
           '| 本轮新候选（dev180/78） | Scan Dice | Subject Dice | Point /180 |','|---|---:|---:|---:|']
    for study in STUDIES:
        for key,m in r['methods'][study]['metrics'].items():lines.append(f"| {key} | {m['scan_dice']:.9f} | {m['subject_dice']:.9f} | {m['point_hits']} |")
    lines+=['','**结果与直觉：** 旋转在患者均衡层两seed自身对照Scan差−0.005932/−0.006323，四Dice区间均为负；原采样层符号不稳且点门失败。模糊自身对照Scan差−0.006284/−0.003085、Point+3/−3，未形成双seed收益。失败配方关闭，不调强度/阈值/seed救门；不证明所有数据处理到上限。','',
            '**同队列旧榜：** reused internal test270/119 overall Scan仍0.757269662、Subject0.741253585、Point265/270（Full972知识蒸馏模型固定八镜像，8×）；单次前向Scan仍0.752003176、Subject0.737266420、Point252/270（GCSAM184）。这里是项目内部复用队列，不是官方隐藏榜或全球SOTA。',
            '**区间范围：** 新候选只在同一78人的复用开发集评估，两seed不是独立队列；10,000次患者簇配对95%CI未作多重探索校正。完整区间见各RESULT.json，点估计与统计支持分开。','',
            f"**成本与访问：** 新训练6条/{r['new_training_allocated_gpu_hours']:.3f} GPU小时；正式开发共1800前向，其中候选1080、旧模型诊断720，工程smoke另计。候选mask270MiB与旧诊断mask180MiB均已核对；旧诊断类别cache360MiB、新fit类别cache1.336GiB保留在/tmp，六模型及恢复点保留。推理仍4.70M级/1×，当前没有获准新部署模型；全864分支0训练/0测试，原dev未转为训练。",'',
            '**下一优先级：** 先设计保留不同组织区域标签的匹配对照，原始fit0—4类cache已684/684独立核验，尚未训练区域模型；其次才是更高成本的原生体素块/盲滑窗参照。旧官方训练源字节parity限制保留，不重开保护队列调参。', '',
            '[本窗口总报告](archive/research_records/2026-09-10_resumed_eight_hour/RESULT.md)、[机器汇总](archive/research_records/2026-09-10_resumed_eight_hour/RESULT.json)、[证据核验](archive/research_records/2026-09-10_resumed_eight_hour/VERIFICATION_RECEIPT.json)、[后续研究判断](archive/research_records/2026-09-10_resumed_eight_hour/RESEARCH_SYNTHESIS.md)、[区域标签工程](archive/research_records/2026-09-10_fit_subregion_labels/RESULT.md)、[旧点记录更正](archive/research_records/2026-09-10_resumed_eight_hour/POINT_RECORD_CORRECTION.md)。',
            '以下带旧时间的运行记录保留历史；本轮所有训练/候选评测均已结束，不重启已关闭入口。']
    return '\n'.join(lines)+'\n'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['audit','final'],required=True);args=ap.parse_args()
    if args.stage=='final':
        assert datetime.now(timezone.utc)>=END,'do not finalize the eight-hour window early'
        assert not (REC/'RESULT.json').exists()
    r=collect();checks=validators();now=datetime.now(timezone.utc)
    if args.stage=='audit':
        out=REC/('END_AUDIT_'+now.strftime('%H%M%S')+'.json');write_new(out,{'status':'all_completed_evidence_verified_before_window_end','created_utc':now.isoformat(),'evidence':r,'validators':checks,'window_not_yet_finalized':True})
        print(json.dumps({'audit':str(out),'training_runs':6,'gpu_hours':r['new_training_allocated_gpu_hours'],'development_forwards':r['new_formal_development_forwards'],'test_forwards':0},indent=2));return
    quality_path=sorted(REC.glob('QUALITY_*.json'))[-1];quality=json.loads(quality_path.read_text());assert quality['status']=='passed'
    r.update(status='eight_hour_window_completed_leaderboard_target_not_achieved',started_utc=START.isoformat(),deadline_utc=END.isoformat(),finished_utc=now.isoformat(),elapsed_seconds=(now-START).total_seconds(),quality_report_path=str(quality_path.relative_to(ROOT)),quality_report_sha256=sha(quality_path))
    write_new(REC/'RESULT.json',r)
    write_new(REC/'VERIFICATION_RECEIPT.json',{'status':'passed_declared_evidence_hash_and_protocol_checks','created_utc':now.isoformat(),'result_sha256':sha(REC/'RESULT.json'),'source_sha256':r['source_sha256'],'validators_before_publication':checks,'new_forwards':0,'checkpoint_hashes_recomputed':True,'mask_files_rechecked':True,'verifier_sha256':sha(__file__)})
    card=final_markdown(r);(REC/'CARD.md').write_text(card)
    body=card.replace('](archive/','](../../../archive/')
    body+='\n## Checkpoints（未晋级）\n\n'
    for key,c in r['new_checkpoints'].items():body+=f"- [{key}](../../../{c['path']}) SHA `{c['sha256']}`。\n"
    with (REC/'RESULT.md').open('x') as f:f.write(body)
    publish(str(REC.relative_to(ROOT)))
    write_new(REC/'FINAL_VALIDATORS.json',{'status':'passed','validators':validators(),'handoff_sha256':sha(ROOT/'AFLOW_BRATS_HANDOFF_CURRENT.md'),'catalog_sha256':sha(ROOT/'AFLOW_BRATS_METHOD_CATALOG.md'),'leaderboard_sha256':sha(ROOT/'AFLOW_BRATS_LEADERBOARD.json'),'leaderboard_markdown_sha256':sha(ROOT/'AFLOW_BRATS_LEADERBOARD.md')})
    print(json.dumps({'status':r['status'],'elapsed_seconds':r['elapsed_seconds'],'gpu_hours':r['new_training_allocated_gpu_hours'],'new_models':6,'new_test_forwards':0,'leaderboard_surpassed':False},indent=2))


if __name__=='__main__':main()
