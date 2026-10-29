"""Publishable final scientific records, only after time/jobs/equivalence gates."""
import csv,re,json
from datetime import datetime,timezone
from scripts.segresnet_auto_inference12h_20261028.common import *
from scripts.segresnet_auto_inference12h_20261028.refresh_interim_v2 import main as refresh
from scripts.segresnet_auto_inference12h_20261028.completion_audit_v2 import collect


def main():
    authorization=read(R/'AUTHORIZATION.json');assert time.time()>=authorization['deadline_unix'],'minimum12h not complete'
    refresh();pre=collect(False)
    missing=[r for r in pre['requirements'] if r['status']!='proved' and r['requirement']!='final scientific report and machine summary exist'];assert not missing,missing
    inc=read(R/'INCUMBENT.json');folder=ROOT/Path(inc['result_path']).parent;result=read(folder/'RESULT.json');metrics=result['metrics'];config=read(R/'PORTABLE_RECIPE_CURRENT.json');compute=read(R/'COMPUTE_SNAPSHOT.json');runtime=read(R/'FINAL_RUNTIME_AUDIT.json');comparisons=read(folder/'PAIRWISE_REFERENCES.json')['comparisons'];fresh=read(R/'live_verification'/folder.name/'DONE.json');alternative=read(R/'candidates/0018_adaptive_holes/RESULT.json');count=compute['completed_configuration_tuples'];elapsed=time.time()-authorization['start_unix']
    high_cost=read(R/'candidates/0041_intensity_ensemble_holes/RESULT.json')
    compact_recipe=json.dumps(dict(aggregation=config['aggregation'],decoder=config['decoder'],postprocessing=config['postprocessing']),ensure_ascii=False)
    original=read(ROOT/'archive/research_records/2026-10-28_segresnet_w32_e171_test270/RESULT.json');previous=read(OLD/'RESULT.json')
    candidates=[]
    for path in (R/'candidates').glob('*/RESULT.json'):
        item=read(path)
        if item.get('all270_masks_independently_verified'):candidates.append((path.parent.name,item))
    candidates.sort(key=lambda x:(-x[1]['metrics']['scan_dice'],-x[1]['metrics']['subject_dice'],x[1]['locator_forwards'],x[0]))
    with (R/'ALL_AUDITED_CANDIDATES.tsv').open('w') as handle:
        writer=csv.writer(handle,delimiter='\t',lineterminator='\n');writer.writerow(['candidate','scan_dice','subject_dice','point_hits','neural_forwards','result_sha256'])
        for name,item in candidates:writer.writerow([name,item['metrics']['scan_dice'],item['metrics']['subject_dice'],item['metrics']['point_hits'],item['locator_forwards'],sha(R/'candidates'/name/'RESULT.json')])
    with (R/'ALL_COMPLETED_ANALYSES.tsv').open('w') as handle:
        writer=csv.writer(handle,delimiter='\t',lineterminator='\n');writer.writerow(['analysis','configuration_tuples','best_scan','best_subject','best_point_hits'])
        for item in compute['analyses']:
            best=item.get('best') or {};writer.writerow([item['id'],item['configurations'],best.get('scan_dice'),best.get('subject_dice'),best.get('point_hits')])
    final=dict(at=now(),state='research_complete_pending_publication',method=inc['method'],metrics=metrics,checkpoint_sha256=CHECKPOINT_SHA,selected_candidate=folder.name,selected_recipe_path=inc['recipe_path'],selected_recipe_sha256=sha(ROOT/inc['recipe_path']),portable_recipe_path=str((R/'PORTABLE_RECIPE_CURRENT.json').relative_to(ROOT)),portable_recipe_sha256=sha(R/'PORTABLE_RECIPE_CURRENT.json'),neural_forwards_per_scan=result['locator_forwards'],comparisons=comparisons,completed_configuration_tuples=count,audited_candidates=len(candidates),elapsed_seconds=elapsed,minimum_research_seconds=43200,training_model_weights=False,GT_input_to_predictor=False,GT_per_case_routing=False,test_tuned=True,new_outer_confirmation_access=False,new_validation108_access=False,independent_generalization_confirmed=False,fresh_neural_equivalence=fresh,data_model_runtime_audit_path=str((R/'FINAL_RUNTIME_AUDIT.json').relative_to(ROOT)),compute_snapshot_path=str((R/'COMPUTE_SNAPSHOT.json').relative_to(ROOT)),engineering_recovery_count=read(R/'RECOVERY_COUNT.json')['used'],raw_sha256=sha(folder/'RAW_TEST270.json'),result_sha256=sha(folder/'RESULT.json'))
    write(R/'FINAL_RESULT.json',final)
    (R/'DEPLOYMENT.md').write_text((R/'DEPLOYMENT_INTERIM.md').read_text().replace('# 当前无训练核心视图部署接口','# 最终选定推理配方的部署接口',1))
    write(R/'COMPLETION_AUDIT.json',dict(at=now(),state='pending_final_audit',note='Final independent evidence audit must pass before publication/goal completion'))
    versus=comparisons['original_w32'];versus_previous=comparisons['study_start_mirror5'];u=compute['utilization'];fresh_scope=f"完整{fresh['scans']}例" if fresh['scans']==270 else f"预登记固定前{fresh['scans']}例（高成本复核，不是270例重推）"
    report=f'''# 固定w32第171轮无训练推理研究：最终报告

本研究持续{elapsed/3600:.3f}小时，达到至少12小时窗口。固定单T1ce、SegResNet32通道第171轮及原972/108/270角色；270测试扫描/119患者、原native1mm去填充栅格。新网络训练0，不使用其他模型权重，不新增108/outer/confirmation评测，不干预独立RACE。用户授权直接用270反馈全局选择，**全部结果test-tuned，未独立确认泛化**。

| 同权重方案 | Scan Dice | Subject Dice | Point | NN前向/例 |
|---|---:|---:|---:|---:|
| 原始推理 | {original['scan_dice']:.9f} | {original['subject_dice']:.9f} | {original['point_hits']}/270 | 1 |
| 研究前固定五镜像/体积解码 | {previous['scan_dice']:.9f} | {previous['subject_dice']:.9f} | {previous['point_hits']}/270 | 5 |
| 七视图成本替代 | {alternative['metrics']['scan_dice']:.9f} | {alternative['metrics']['subject_dice']:.9f} | {alternative['metrics']['point_hits']}/270 | 7 |
| 本轮最高 `{folder.name}` | **{metrics['scan_dice']:.9f}** | **{metrics['subject_dice']:.9f}** | **{metrics['point_hits']}/270** | **{result['locator_forwards']}** |
| 高成本增强融合 | {high_cost['metrics']['scan_dice']:.9f} | {high_cost['metrics']['subject_dice']:.9f} | {high_cost['metrics']['point_hits']}/270 | {high_cost['locator_forwards']} |

对原始推理差 **{versus['scan_delta']:+.9f}**，患者配对描述性95%区间{versus['scan_delta_ci95']}；对研究前五镜像差{versus_previous['scan_delta']:+.9f}，区间{versus_previous['scan_delta_ci95']}。这些区间在大规模全局选择之后计算，未作多重搜索校正，不能当独立确认或稳定泛化保证。Scan是逐扫描均值，Subject是119患者各自均值后再平均，Point按原定义另列。

## 固定配方与部署核验

固定全局配置：`{compact_recipe}`。概率按65536级量化，使用只读预测总直方图的体积解码；不存在逐例GT最优阈值选择。

聚合`{config['aggregation']['kind']}`，解码`{config['decoder_kind']}`；完整全局参数见[便携配方](PORTABLE_RECIPE_CURRENT.json)和[配方链](candidates/{folder.name}/RECIPE.json)。部署只接图像和image-derived去填充几何，不输入GT、病例ID或270个逐例切点查表。图像脑区约束仅限制预测与无标签计数，评分始终保留整个原native crop GT，支撑外阳性仍计FN；原历史裁剪范围没有改变。

全部270选中掩膜/TP-FP-FN/点由原缓存概率或logits独立重建并与搜索矩阵对齐。独立API重新神经前向范围为**{fresh_scope}**，mask差{fresh['mask_different_voxels']}体素、点相同、逐例Dice差0，训练0。此前所有输入增强已真实覆盖270，这里的fresh是重复工程核对，非新队列验证。[具体核验](live_verification/{folder.name}/RESULT.json) · [部署接口](DEPLOYMENT.md)。

## 完整搜索与成本

共完成**{count:,}个有限全局参数元组**，不是独立方法数、独立统计假设数或唯一掩膜数；{len(candidates)}个候选完整掩膜审计。包含镜像子集、温度/体积解码、预测一致性、STAPLE控制、非线性权重、MC-dropout、轴置换、近尺度/ROI、噪声平均、两方向512整数相位、官方RankSEG、局部MRF、原生3D DenseCRF、signed-distance形状融合、脑区约束及组件/孔洞组合，和28强度组及其有限融合。所有正负结果保留：[全分析表](ALL_COMPLETED_ANALYSES.tsv)、[已审计表](ALL_AUDITED_CANDIDATES.tsv)、各analyses目录及原矩阵SHA。

DenseCRF该有限网格最高是无边项控制，不能归功于边项；部分phase融合最高为alpha0控制，不能归功于相位。两512方向各138240次患者case-forward；1/64/128/256/512共享前缀，不能相加作独立NN预算。已完成生成器记录下界{compute['completed_renderer_case_forwards_lower_bound']:,}次，另有fresh验证；失联旧节点/未提交失败计算不伪造精确总数。CPU后处理、I/O和统计成本另计，前向次数不是FLOPs。相位画布176×224×184，体素成本按实际填充形状单列，不能按native画布低估。

新worker1153287两A10080的20秒采样覆盖{u['coverage_seconds']/3600:.3f}小时，区间时间加权利用率GPU0 {u['time_weighted_mean_percent']['0']:.2f}%、GPU1 {u['time_weighted_mean_percent']['1']:.2f}%；不跨失联或采样外空档外推，不把某次100%称全程100%。[计算/采样口径](COMPUTE_SNAPSHOT.json)。合成NN-only硬件测量另见[批量吞吐与显存](HARDWARE_PROFILE.json)，不冒称真实病例端到端延迟，已核验生产批量仍B8/尾6。

## 边界、完整性和复现

唯一checkpoint SHA `{CHECKPOINT_SHA}`；原三载荷与index逐SHA一致。最终持久新增{runtime['persistent_bytes']/2**30:.3f}GiB，实际计入主机保守余量的scratch {runtime['scratch_bytes_including_main_allowance']/2**30:.3f}GiB，满足2/300GiB上限及20/70GiB空闲门；原数据、模型、core8逐视图logits、全部分数/切点/审计掩膜保留。滚动清理只处理已归档无读者的本轮可重建派生文件，保留目录骨架；为组合28组输出，清理监督后来正常结束，没有模型重启。工程恢复{final['engineering_recovery_count']}/6，旧失败与修复收据保留。[最终SHA与预算](FINAL_RUNTIME_AUDIT.json) · [权重/缓冲区不变及API输入契约](INFERENCE_CONTRACT_PROBE.json)。

联网原始论文/官方资料及借鉴范围见[WEB_REVIEW](WEB_REVIEW.md)，每轮网格和资源门见ROUND001–030。没有把其他架构、多权重或额外生成模型的收益归到本权重，也没有把当前有限网格的负结果扩称所有相关方法无效。全局参数直接复用270调优，病灶碎片过滤可能损害部分扫描，平均提升不表示每例都提高。

[机器结果](FINAL_RESULT.json) · [完成证据审计](COMPLETION_AUDIT.json)。科研仓库只同步源码和文本证据，不上传MRI/标签、权重、预测或数值数组；最终发布收据单独记录，历史缺失私有载荷的旧链接不冒称修复。
'''
    (R/'RESULT.md').write_text(report)
    marker='AUTO_INFERENCE12H_20261028';pattern=rf'<!-- {marker} -->.*?<!-- END_{marker} -->';link=f'archive/research_records/{R.name}/RESULT.md'
    body=f"本轮至少12小时自动推理研究已完成。固定w32第171轮/T1ce/原270/119，网络权重训练0。最高Scan **{metrics['scan_dice']:.9f}**、Subject **{metrics['subject_dice']:.9f}**、Point **{metrics['point_hits']}/270**，{result['locator_forwards']}次前向；对原始差{versus['scan_delta']:+.9f}。完成{count:,}有限全局元组，全部test-tuned，选择后区间未校正搜索，无独立泛化确认。270缓存掩膜审计及API神经复核范围见[最终报告]({link})。不自动扩搜或重启旧实验，独立RACE不受影响。"
    for name in ['AGENTS.md','AFLOW_BRATS_HANDOFF_CURRENT.md','AFLOW_BRATS_METHOD_CATALOG.md','AFLOW_BRATS_RESULTS.md','AFLOW_BRATS_WEEKLY_REPORT.md']:
        path=ROOT/name;old=path.read_text();assert len(re.findall(pattern,old,re.S))==1
        if name=='AFLOW_BRATS_METHOD_CATALOG.md':text='## 2026-10-29：固定权重推理自动研究完成\n\n**一句话：** 固定模型检验多视图、概率解码与结构组合。\n**出发点：** 提高复用270的Dice。\n**具体做法：** 原始文献调研、有限全局网格、完整270及独立重建。\n**结果：** '+body+'\n**是否支持直觉：** 当前内部队列的探索增益，不是独立泛化确认。\n**代价/下一步：** 见报告完整成本；本轮终态，不自动续搜。'
        else:text='## 2026-10-29：12小时无训练推理研究完成\n\n'+body
        new=re.sub(pattern,lambda _:f'<!-- {marker} -->\n{text}\n<!-- END_{marker} -->',old,flags=re.S);assert re.sub(pattern,'',old,flags=re.S)==re.sub(pattern,'',new,flags=re.S);path.write_text(new)
    board=read(ROOT/'AFLOW_BRATS_LEADERBOARD.json')
    for row in board['test270_tuned_native_rank_by_scan']:
        if row['method'].startswith('segresnet_w32_e171_auto12h_'):
            row.update(research_continues=False,evidence_status='complete: directly selected on reused270; post-selection test-tuned descriptive evidence',new_validation108_access=False,new_network_training=False,validation_gate_note='Historical training gate only; no new108 evaluation in this study')
    board['latest_segresnet_auto12h_interim_20261028'].update(research_continues=False,superseded_by='latest_segresnet_auto12h_final_20261029')
    board['latest_segresnet_auto12h_final_20261029']=dict(method=inc['method'],scan_dice=inc['scan_dice'],subject_dice=inc['subject_dice'],point_hits=inc['point_hits'],research_continues=False,test_tuned=True,report_path=str((R/'RESULT.md').relative_to(ROOT)))
    board['updated_utc']=datetime.now(timezone.utc).isoformat()
    write(ROOT/'AFLOW_BRATS_LEADERBOARD.json',board)
    inc.update(state='complete_audited_test_tuned_result',research_continues=False);write(R/'INCUMBENT.json',inc)
    write(R/'RESEARCH_FINAL_STATE.json',dict(at=now(),state='research_complete_publication_pending',minimum12h_met=True,all_registered_tasks_complete=True,current_incumbent=inc))
    print(dict(state='research_complete_publication_pending',candidate=folder.name,scan_dice=metrics['scan_dice'],configurations=count))

if __name__=='__main__':main()
