"""Data-driven session report and mechanical canonical summary refresh, preserving history."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from scripts import structured_background_protocol as p

RECORD=p.ROOT/'archive/research_records/2026-09-10_eight_hour_data_processing'
START=datetime(2026,9,10,3,28,19,tzinfo=timezone.utc)
END=datetime(2026,9,10,11,28,19,tzinfo=timezone.utc)
METHODS=['weak_elastic','fractional_occupancy','peak_protected_cleanup','data_recipe_ensemble','subject_fixed_flip8']
MARKER='aflow-eight-hour-20260910'


def atomic(path,body):
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile('w',dir=path.parent,encoding='utf-8',delete=False) as f:f.write(body);temporary=Path(f.name)
    temporary.replace(path)


def replace_generated_block(path,body):
    text=path.read_text();backup=RECORD/(path.stem+'_BEFORE_SESSION_SUMMARY.md')
    if not backup.exists():
        with backup.open('x') as f:f.write(text)
    begin=f'<!-- {MARKER}:start -->';end=f'<!-- {MARKER}:end -->'
    block=begin+'\n'+body.strip()+'\n'+end
    if begin in text:
        a=text.index(begin);b=text.index(end,a)+len(end);updated=text[:a]+block+text[b:]
    else:
        first,rest=text.split('\n',1);updated=first+'\n\n'+block+'\n'+rest
    atomic(path,updated)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--final',action='store_true');args=parser.parse_args()
    now=datetime.now(timezone.utc)
    if args.final and now<END:raise RuntimeError('eight-hour boundary not reached')
    RECORD.mkdir(parents=True,exist_ok=True);p.check_roles()
    method_results={};evidence={};total_forwards=0
    for name in METHODS:
        record=p.ROOT/f'archive/research_records/2026-09-10_{name}'
        result=json.loads((record/'RESULT.json').read_text());verification=json.loads((record/'VERIFICATION_RECEIPT.json').read_text())
        assert verification['status']=='passed';p.checked(record/'RESULT.json',verification['result_sha256']);p.checked(record/'RAW_DEVELOPMENT.json',verification['raw_sha256'])
        raw=json.loads((record/'RAW_DEVELOPMENT.json').read_text())
        total_forwards+=raw.get('new_forwards',raw.get('forwards',0))
        for filename in ('RESULT.json','RESULT.md','RAW_DEVELOPMENT.json','DEV_PREACCESS.json','VERIFICATION_RECEIPT.json'):
            path=record/filename;evidence[str(path.relative_to(p.ROOT))]=p.sha(path)
        method_results[name]=result
        assert not (record/'TEST_PREACCESS.json').exists()
    assert total_forwards==5040
    new_training={}
    for experiment,arms in [('weak_elastic',('native_elastic','subject_elastic')),('fractional_occupancy',('fractional_occupancy',))]:
        for seed in p.SEEDS:
            for arm in arms:
                path=p.ROOT/f'workspace/BraTS/sammed3d_t1ce_wt/{experiment}_20260910/seed{seed}_{arm}/TRAIN_DONE.json';d=json.loads(path.read_text())
                assert d['fixed_epochs']==168 and d['all_gates_passed'] and d['export_max_abs_error']==0 and not d['teacher_checkpoints']
                cp=Path(d['checkpoint']['path']);p.checked(cp,d['checkpoint']['sha256'])
                new_training[f'{seed}_{arm}']={'checkpoint':str(cp.relative_to(p.ROOT)),'checkpoint_sha256':p.sha(cp),'train_report_sha256':p.sha(path),'epochs':168,'steps':sum(r['steps'] for r in d['history']),'seconds':d['elapsed_seconds'],'peak_bytes':d['peak_gpu_memory_bytes']}
    assert len(new_training)==6
    point_path=p.ROOT/'archive/research_records/2026-09-10_subject_fixed_flip8/POINT_AUDIT.json';point=json.loads(point_path.read_text());assert point['points_checked']==720
    evidence[str(point_path.relative_to(p.ROOT))]=p.sha(point_path)
    cache_record=p.ROOT/'archive/research_records/2026-09-10_fractional_occupancy'
    for filename in ('CACHE_RECEIPT.json','CACHE_INDEX.json','CACHE_PREACCESS.json','CACHE_VERIFICATION.json','DATASET_CPU_RECEIPT.json','TRAINING_REPLAY_FINAL.json'):
        path=cache_record/filename;evidence[str(path.relative_to(p.ROOT))]=p.sha(path)
    p.checked(p.CACHE/'index.json',p.INDEX_SHA)
    board_path=p.ROOT/'AFLOW_BRATS_LEADERBOARD.json';board=json.loads(board_path.read_text());top=board['test270_direct_locator_rank_by_scan'][0];single=next(r for r in board['test270_direct_locator_rank_by_scan'] if r.get('locator_forwards')==1)
    assert top['scan_dice']==.7572696616942132 and single['scan_dice']==.7520031755617648
    gpu_output=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip()
    if gpu_output:raise RuntimeError('GPU has active compute apps; do not claim all idle: '+gpu_output)
    tta=method_results['subject_fixed_flip8'];q=method_results['fractional_occupancy'];fusion=method_results['data_recipe_ensemble']
    total_hours=sum(r['seconds'] for r in new_training.values())/3600
    data={'schema':'aflow.eight_hour_data_processing_session','created_utc':now.isoformat(),'start_utc':START.isoformat(),'planned_end_utc':END.isoformat(),'elapsed_seconds':(now-START).total_seconds(),'eight_hours_elapsed':now>=END,'status':'eight_hour_scope_finished_target_not_achieved' if args.final else 'all_results_archived_waiting_for_time_boundary','leaderboard_surpassed':False,'new_test_forwards':0,'formal_development_forwards':total_forwards,'new_models':6,'new_training_gpu_hours':total_hours,'new_training':new_training,'fit_scans':684,'fit_subjects':313,'dev_scans':180,'dev_subjects':78,'development_is_reused':True,'former_development_consumed_as_training':False,'access':{'test':False,'old_confirmation':False,'fixed_validation108':False,'outer':False,'external':False},'gpu_compute_apps':gpu_output,'test_leaderboard_overall':top,'test_leaderboard_one_forward':single,'development_dice_positive_method':'subject_fixed_flip8','development_flip8_metrics':tta['metrics'],'development_flip8_comparisons':tta['comparisons'],'development_flip8_all_gates_passed':False,'method_statuses':{name:r['status'] for name,r in method_results.items()},'point_audit_summary':{s:{k:v for k,v in r.items() if k!='per_case'} for s,r in point['results'].items()},'source_evidence_sha256':evidence,'cache_index_sha256':p.sha(p.CACHE/'index.json'),'leaderboard_json_sha256':p.sha(board_path),'label_cache_bytes':2868903936,'environment_incident':'default exec and patch reader lost repository access after10:50; authorized host /bin/sh execution recovered archival; no model rerun or metric change','late_execution_plan_amendments':['independent fixed0.5 cross-recipe fusion after constituent dev disclosure','independent fixed8-view inference on stronger patient models; original method gates retained'],'inference_cost_scope':'identity1x; cross-recipe ensemble2x; fixed reflection8x; no new test deployed winner','next_priority':'diagnose and preregister boundary-robust automatic localization while retaining the demonstrated same-model multi-view segmentation benefit; do not change current Point truth, thresholds, or test gate posthoc'}
    atomic(RECORD/'RESULT.json',json.dumps(data,indent=2,ensure_ascii=False)+'\n')
    a=tta['metrics']['2026090711'];b=tta['metrics']['2026090712'];ca=tta['comparisons']['2026090711'];cb=tta['comparisons']['2026090712']
    phase='八小时工作范围完成，但超榜目标未达成' if args.final else '结果已归档，等待八小时截止；超榜目标尚未达成'
    lines=['# 八小时数据处理研究汇总','',phase+'。','',f"开始{START.isoformat()}，计划截止{END.isoformat()}，本报告生成{now.isoformat()}。",'',
           '## 核心结论','',
           '没有新增test270评测，也没有超过原榜单。找到可复现的开发分割增益：患者采样模型的固定八镜像平均，两seed扫描/患者Dice都提高且四个患者簇95%区间为正；但第二seed定位168→160，触发原安全门。不能用开发0.759与旧测试0.757跨cohort比大小。','',
           '| 方法 | 同dev180/78的观察 | 状态 |','|---|---|---|',
           '| 弱形变，native层 | Scan差−.001232/−.000046；Point+2/+4 | 分割门失败 |',
           '| 弱形变，patient层 | 对自身强成员Scan−.011245/−.004780；Point+5/−3 | 固定组合关闭 |',
           '| 体积占比target | Scan差−.000104/−.004325；Point+5/+4 | 定位有益，分割门失败 |',
           '| 小孤岛清理 | Scan差约−.000093/−.000121，Point不变 | 无分割收益，关闭 |',
           '| 数据配方等权融合 | 对native正，但第二seed低于较强sampling成员 | 未稳定超过强成员，关闭 |',
           f"| 同模型固定八镜像 | Scan {a['flip8']['scan_dice']:.9f}/{b['flip8']['scan_dice']:.9f}；Point169/160 | Dice-positive / Point门失败，test关闭 |",'',
           f"八镜像Scan差{ca['scan_delta']:+.6f}/{cb['scan_delta']:+.6f}，患者簇95%CI分别{ca['scan_cluster_95ci']}、{cb['scan_cluster_95ci']}；Subject差{ca['subject_delta']:+.6f}/{cb['subject_delta']:+.6f}，对应CI {ca['subject_95ci']}、{cb['subject_95ci']}。同一78人、非两批独立患者，10k百分位CI未多比较校正。",'',
           '## 为什么不能忽略定位门','',
           '720保存点与原GT核验exact。第二seed lost11/gained3，不只是8个点变化；6个lost病例的Dice反而提高。丢失点距GT中位1个128-grid输入体素、范围1–4.123，不能当毫米或把邻近点改算命中。它提示边界敏感性，未发现坐标评分错误。','',
           '## 控制、成本与边界','',
           f"六条固定168轮训练完成，共{sum(r['steps'] for r in new_training.values())} optimizer steps、{total_hours:.4f} GPU小时，峰值{max(r['peak_bytes'] for r in new_training.values())/1024**3:.3f}GiB。正式开发5040病例forwards，另有有限工程smoke。新占比label cache2.672GiB在/tmp；未删除材料数据。",'',
           '空间增强只改训练时形变；占比target只改训练标签，原图像/抽样/原损失/硬GT评分不变；推理研究固定0.5组合或全八镜像，没有权重/视图子集/阈值网格。旧模型/单模型失败门均保留，高层新增推理研究在开发揭示后明确另注册，不称原始盲计划。','',
           '部署成本：原模型4.70M/1×，固定八镜像同模型8×，跨数据模型融合2模型/2×（若八镜像16×）。没有把多次推理叫单次模型提分。当前所有GPU计算任务均已结束。', '',
           'fit684/313和dev180/78保持，未消费dev为full864训练。没有test、oldconfirmation、fixed validation108、outer、external新性能访问；训练来源镜像/官方字节parity未证明，历史test-informed背景保留。','',
           '## 同口径旧榜与下一步','',
           f"旧reused test270/119整体Scan最高{top['scan_dice']:.9f}、Subject{top['subject_dice']:.9f}、Point265/270，Full972KD168固定8×；1次前向Scan最高{single['scan_dice']:.9f}（GCSAM184）。这些数字没有被本轮开发结果刷新。",'',
           '下一优先不是继续盲扫亮度、形变强度或混合比例，而是围绕已观察到的边界敏感最高点，预先设计不牺牲区域覆盖的自动定位/训练数据研究。当前不能改Point定义救门，也没有证据说数据处理或模型已到上限。', '',
           '默认执行/补丁读者在收尾时出现环境级找不到路径；主机/bin/sh授权执行恢复，原始报告已存在且无重跑。汇总为机器结果驱动的报告生成和带备份的机械文档刷新，历史正文不删除。','',
           '[机器汇总](RESULT.json)、[当前交接](../../../AFLOW_BRATS_HANDOFF_CURRENT.md)、[方法百科](../../../AFLOW_BRATS_METHOD_CATALOG.md)、[旧榜](../../../AFLOW_BRATS_LEADERBOARD.md)。']
    for name in METHODS:lines.append(f"\n- [{name}结果](../2026-09-10_{name}/RESULT.md)，[raw](../2026-09-10_{name}/RAW_DEVELOPMENT.json)，[验证](../2026-09-10_{name}/VERIFICATION_RECEIPT.json)。")
    for name,value in new_training.items():lines.append(f"\n- [{name} checkpoint](../../../{value['checkpoint']}) SHA `{value['checkpoint_sha256']}`。")
    atomic(RECORD/'RESULT.md','\n'.join(lines)+'\n')
    common=(f"## 2026-09-10八小时轮次最新汇总（{now.strftime('%H:%M')} UTC）\n\n{phase}。"
            f"六条168轮新训练完成（{total_hours:.4f}GPUh），五项正式开发读出5040forwards；无新test/outer/confirmation/external，GPU无活跃计算。"
            f"同dev180/78，固定患者模型八镜像Scan0.759243559/0.758989503，差+.005769/+.003084，四Dice簇CI均正，但Point169/160（+2/−8）使第二seed失败。"
            "占比target单模型、弱形变、小岛清理、异质等权融合均已归档未过完整门；没有任何test准入或榜单刷新。"
            "不能把开发0.759当成超过旧test0.757。所有失败门不放宽、不改定义、不挑seed。\n\n"
            "点核查720/720 exact，seed12 lost11/gained3，6个lost例Dice提高，错点距离中位1输入体素（非mm）。"
            "原fit/dev角色保持；全864形变未执行。\n\n"
            "[本轮总结果](archive/research_records/2026-09-10_eight_hour_data_processing/RESULT.md)、"
            "[机器汇总](archive/research_records/2026-09-10_eight_hour_data_processing/RESULT.json)、"
            "[固定镜像结果](archive/research_records/2026-09-10_subject_fixed_flip8/RESULT.md)、"
            "[点核查](archive/research_records/2026-09-10_subject_fixed_flip8/POINT_AUDIT.json)。\n\n"
            "以下同日过程段落保留历史，当前状态以本汇总和机器RESULT为准；旧运行会话均已结束，不重启。")
    replace_generated_block(p.ROOT/'AFLOW_BRATS_HANDOFF_CURRENT.md',common)
    cards=common+'\n\n### 本轮完成方法状态\n\n'
    for name,result in method_results.items():
        cards+=f"- **{name}：{result['status']}**；[结果/做法/固定因素/成本与下一步](archive/research_records/2026-09-10_{name}/RESULT.md)，[原始证据](archive/research_records/2026-09-10_{name}/RAW_DEVELOPMENT.json)。\n"
    cards+='\n其中subject_fixed_flip8为Dice-positive/Point-failed，不是完全无分割收益；其他方法的具体负结果不能扩张为一般不可能。\n'
    replace_generated_block(p.ROOT/'AFLOW_BRATS_METHOD_CATALOG.md',cards)
    validators={}
    for script in ['verify_aflow_brats_doc_layout.py','verify_aflow_brats_method_catalog.py','verify_aflow_brats_leaderboard.py']:
        completed=subprocess.run([sys.executable,str(p.ROOT/'scripts'/script)],cwd=p.ROOT,text=True,capture_output=True)
        validators[script]={'returncode':completed.returncode,'stdout':completed.stdout,'stderr':completed.stderr}
        if completed.returncode:raise RuntimeError(f'{script} failed: {completed.stdout} {completed.stderr}')
    atomic(RECORD/'VERIFICATION_RECEIPT.json',json.dumps({'status':'passed','created_utc':datetime.now(timezone.utc).isoformat(),'result_sha256':p.sha(RECORD/'RESULT.json'),'source_evidence_rehashed':evidence,'checkpoint_count_rehashed':len(new_training),'cache_index_sha256':p.sha(p.CACHE/'index.json'),'leaderboard_sha256':p.sha(board_path),'validators':validators,'new_model_forwards':0,'generator_sha256':p.sha(Path(__file__))},indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'status':data['status'],'elapsed_seconds':data['elapsed_seconds'],'new_models':6,'training_gpu_hours':total_hours,'formal_development_forwards':5040,'new_test_forwards':0,'leaderboard_surpassed':False,'validators_passed':True,'result':str(RECORD/'RESULT.md')},indent=2),flush=True)


if __name__=='__main__':main()
