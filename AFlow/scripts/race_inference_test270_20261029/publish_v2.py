"""Statistics-only merge, full saved-mask audit and truthful publication."""
import copy
import re
import numpy as np
from scripts.race_inference_test270_20261029.common import *


def block(name,body):
    path=ROOT/name;text=path.read_text();start='<!-- RACE164_TEST270_20261029 -->';end='<!-- END_RACE164_TEST270_20261029 -->'
    value=start+'\n'+body+'\n'+end
    if start in text:text=text[:text.index(start)]+value+text[text.index(end)+len(end):]
    else:
        first=text.index('\n');text=text[:first+1]+'\n'+value+'\n'+text[first+1:]
    path.write_text(text)


def main():
    verify();verify('EXECUTION_FREEZE_V2.json');ready=read(R/'CACHE_READY.json');index=read(CACHE/'index.json');case_rows=rows()
    labels=np.memmap(CACHE/'labels.u8',np.uint8,'r',shape=(270,*CANVAS))
    ranks=[read(R/f'RAW_RANK_{rank}.json') for rank in (0,1)]
    records={};metrics={};audited=0
    for scheme in ('raw','enhanced'):
        cases=sorted([x for rank in ranks for x in rank['per_scheme'][scheme]],key=lambda x:x['source_slot'])
        validate_rows(cases)
        for slot,row in enumerate(cases):
            path=ROOT/row['prediction_path'];assert sha(path)==row['prediction_sha256']
            saved=np.load(path);shape=tuple(map(int,saved['shape']));mask=np.unpackbits(saved['mask_bits'])[:int(np.prod(shape))].reshape(shape).astype(bool)
            point=saved['point'].tolist();g=index['geometry'][row['case_id']];target=np.array(labels[slot][slices(g)]>0)
            assert shape==target.shape and point==row['automatic_point']
            tp=int((mask&target).sum());pred=int(mask.sum());positive=int(target.sum());den=pred+positive
            score=float(np.float32(2*tp)/np.float32(den)) if den else 1.
            assert (tp,pred-tp,positive-tp)==(row['tp'],row['fp'],row['fn'])
            assert score==row['locator_dsc'] and bool(target[tuple(point)])==row['automatic_point_hits_gt']
            audited+=1
        metrics[scheme]=summary_ci(cases);metrics[scheme]['point_hits']=sum(x['automatic_point_hits_gt'] for x in cases)
        records[scheme]=cases
        write(R/f'RAW_TEST270_{scheme}.json',dict(at=now(),scheme=scheme,metrics=metrics[scheme],per_case=cases,
              checkpoint_sha256=CHECKPOINT_SHA,recipe_sha256=sha(R/'SELECTED_RECIPE.json'),test_access=True,test_tuned_history=True))
    old_base_path=ROOT/'archive/research_records/2026-10-28_segresnet_w32_e171_test270/RAW_TEST270.json'
    old_best_path=ROOT/'archive/research_records/2026-10-28_segresnet_auto_inference12h/candidates/0036_local_brain_joint/RAW_TEST270.json'
    old_base=read(old_base_path)['summary']['per_case'];old_best=read(old_best_path)['per_case']
    contrasts=dict(enhanced_minus_race_raw=paired(records['enhanced'],records['raw']),
          race_raw_minus_historical_w32_raw=paired(records['raw'],old_base),
          race_enhanced_minus_historical_w32_enhanced=paired(records['enhanced'],old_best))
    for value in contrasts.values():value['bootstrap']='10000 patient-cluster paired percentile95 seed20260918; descriptive reused/test-informed cohort;97.5 sensitivity not overall search correction'
    change=contrasts['enhanced_minus_race_raw']
    supported=change['scan_delta']>0 and change['scan_delta_ci95'][0]>0
    joint=supported and change['subject_delta']>=0 and change['point_hits_delta']>=0
    runtime=dict(case_forwards=sum(r['neural']['case_forwards'] for r in ranks),
          neural_batch_calls=sum(r['neural']['batch_calls'] for r in ranks),per_rank=[dict(rank=r['rank'],**r['neural'],decode_seconds=r['decode_seconds']) for r in ranks])
    assert runtime['case_forwards']==2160
    result=dict(at=now(),state='complete',checkpoint_epoch=164,checkpoint_sha256=CHECKPOINT_SHA,
          metrics=metrics,comparisons=contrasts,inference_Dice_support=supported,joint_nondecrease_gate=joint,
          engineering_amendment='INPUT_SCHEMA_AMENDMENT.json',active_execution_freeze_sha256=sha(R/'EXECUTION_FREEZE_V2.json'),training=False,test_access=True,current_test_parameter_search=False,inherited_test_tuned_recipe=True,
          new_outer_confirmation_access=False,cache=ready,batch=read(R/'BATCH_DECISION.json'),runtime=runtime,
          grid='native1mm original crop;160x208x168 canvas;all padding excluded',
          reference_reports={str(p.relative_to(ROOT)):sha(p) for p in (old_base_path,old_best_path)},
          no_automatic_continuation=True)
    write(R/'RESULT.json',result)
    write(R/'INDEPENDENT_MASK_AUDIT.json',dict(at=now(),status='passed',saved_masks_audited=audited,
          GT_counts_and_points_exact=True,neural_forwards_during_audit=0,whole_original_crop_scored=True))
    m=metrics['enhanced'];raw=metrics['raw'];oldm=summary_ci(old_best);oldb=summary_ci(old_base)
    report=f'''# RACE-Seg第164轮＋既有最佳推理：270例固定评测

已完成270扫描/119患者，单T1ce、原始1mm裁剪栅格去填充。网络训练0；权重由原108验证选择，唯一第164轮。迁移旧研究最高0036的全部参数，本次未用RACE test输出重新选规则。

| 同一270队列 | Scan Dice | Subject Dice | Point | 前向/例 |
|---|---:|---:|---:|---:|
| 历史w32第171轮原始 | {oldb['scan_dice']:.9f} | {oldb['subject_dice']:.9f} | {sum(x['automatic_point_hits_gt'] for x in old_base)}/270 | 1 |
| 历史w32＋旧最佳推理 | {oldm['scan_dice']:.9f} | {oldm['subject_dice']:.9f} | {sum(x['automatic_point_hits_gt'] for x in old_best)}/270 | 8 |
| RACE164原始 | {raw['scan_dice']:.9f} | {raw['subject_dice']:.9f} | {raw['point_hits']}/270 | 1 |
| RACE164＋旧最佳推理 | **{m['scan_dice']:.9f}** | **{m['subject_dice']:.9f}** | **{m['point_hits']}/270** | 8 |

同权重推理变化：Scan差{change['scan_delta']:+.9f}，患者配对描述性95%区间{change['scan_delta_ci95']}；Subject差{change['subject_delta']:+.9f}，Point差{change['point_hits_delta']:+d}。Dice支持门{supported}，三项不降联合门{joint}。相对旧w32最佳推理Scan差{contrasts['race_enhanced_minus_historical_w32_enhanced']['scan_delta']:+.9f}，区间{contrasts['race_enhanced_minus_historical_w32_enhanced']['scan_delta_ci95']}。

方案为8镜像、预测一致性加权logits、脑区约束的65536级预测直方图体积解码、26连接小于512体素组件去除、填充不超过1000体素封闭孔。不是精确RankDice复现，也不是论文世界SOTA声明。参照矩阵与图像支撑在翻转时同步变换，保持RACE的原训练参照规则；无训练标签或病例ID输入预测API。原始输出复用8视图中的identity，没有额外第9次前向。两卡各135例、共2160次病例前向，批量由纯合成吞吐选择；实际耗时/显存见RESULT.json，前向次数不等于FLOPs。

四个原生/128载荷SHA均与历史相同；新index只增加图像派生的参照几何，不能声称其文件SHA与旧index相同。评分保留整个原始裁剪GT，包括脑支撑外的阳性；原历史裁剪框外4体素口径不变。540份保存掩膜/TP-FP-FN/点独立核对通过。准备阶段字段及负步幅失败、已写载荷复用及N4尝试口径保留。首次评测入口缺少通道维，在预测器第一个形状断言失败，实际患者网络前向0；新增evaluate_v2补入通道维后才开始真实前向。原源码/访问/失败收据不改，见INPUT_SCHEMA_AMENDMENT.json；无重复神经前向，不伪造全阶段每例仅一次N4。

旧推理参数已直接在同一270队列大规模选择，因此本次迁移仍继承test-tuned/test-informed历史，非独立盲测或独立泛化确认。历史模型比较还包含种子/训练/执行差异，不能单独归因RACE模块。本轮结束，不自动重测、续搜、换权重或训练；没有新增outer/confirmation。

[固定方案](IDEA.md) · [参数](SELECTED_RECIPE.json) · [机器结果](RESULT.json) · [原始输出](RAW_TEST270_raw.json) · [增强输出](RAW_TEST270_enhanced.json) · [掩膜审计](INDEPENDENT_MASK_AUDIT.json)
'''
    (R/'RESULT.md').write_text(report)
    write(R/'COMPLETION_AUDIT.json',dict(at=now(),status='passed',checkpoint_sha256=sha(CHECKPOINT),
          result_sha256=sha(R/'RESULT.json'),report_sha256=sha(R/'RESULT.md'),recipe_sha256=sha(R/'SELECTED_RECIPE.json'),
          raw_report_sha256={s:sha(R/f'RAW_TEST270_{s}.json') for s in records},masks_audited=540,
          execution_freeze_sha256=sha(R/'EXECUTION_FREEZE_V2.json'),patient_case_forwards=2160))
    docs=['AGENTS.md','AFLOW_BRATS_HANDOFF_CURRENT.md','AFLOW_BRATS_METHOD_CATALOG.md','AFLOW_BRATS_RESULTS.md','AFLOW_BRATS_WEEKLY_REPORT.md','AFLOW_BRATS_LEADERBOARD.md','AFLOW_BRATS_LEADERBOARD.json']
    backup=R/'docs_before_publication';backup.mkdir(exist_ok=True)
    for name in docs:
        if not (backup/name).exists():shutil.copy2(ROOT/name,backup/name)
    board=read(ROOT/'AFLOW_BRATS_LEADERBOARD.json');before=copy.deepcopy(board);key='test270_tuned_native_rank_by_scan'
    ids=[]
    for scheme in ('raw','enhanced'):
        score=metrics[scheme];identifier=f'race_segresnet_e164_{scheme}_test270_20261029';ids.append(identifier)
        entry=dict(method=identifier,**score,point_denominator=270,cohort=dict(scans=270,subjects=119),
          checkpoint_epoch=164,checkpoint_path=str(CHECKPOINT.relative_to(ROOT)),checkpoint_sha256=CHECKPOINT_SHA,
          grid=result['grid'],score_grid='native',locator_forwards=1 if scheme=='raw' else 8,ensemble_size=1,
          forward_equivalents=(1 if scheme=='raw' else 8)*float(np.prod(CANVAS)/128**3),
          forward_cost_note='input-voxel equivalence only; not FLOPs or latency',
          threshold=0.5 if scheme=='raw' else 'prediction_only_volume_decode',tta=scheme=='enhanced',postprocessing=scheme=='enhanced',
          report_path=str((R/f'RAW_TEST270_{scheme}.json').relative_to(ROOT)),report_sha256=sha(R/f'RAW_TEST270_{scheme}.json'),
          result_path=str((R/'RESULT.json').relative_to(ROOT)),result_sha256=sha(R/'RESULT.json'),
          evidence_status='user-authorized fixed recipe transfer; inherited reused270 test-tuned history; no current test parameter search; not independent confirmation',
          test_tuned=True,current_test_retuning=False,input_modality='T1ce',uses_GT_for_localization=False,
          training_scope=dict(train_scans=972,train_subjects=444,validation_scans=108,validation_subjects=50,epochs=200,optimizer_updates=24400),
          new_outer_confirmation_access=False)
        board[key]=[v for v in board[key] if v['method']!=identifier]+[entry]
    board[key].sort(key=lambda v:(-v['scan_dice'],v['method']))
    for rank,row in enumerate(board[key],1):row['rank']=rank
    for old in before[key]:
        current=next(x for x in board[key] if x['method']==old['method'])
        assert {k:v for k,v in old.items() if k!='rank'}=={k:v for k,v in current.items() if k!='rank'}
    board['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    board['latest_race164_fixed_inference_test270_20261029']=dict(methods=ids,report_path=str((R/'RESULT.md').relative_to(ROOT)),current_test_parameter_search=False)
    write(ROOT/'AFLOW_BRATS_LEADERBOARD.json',board)
    path=ROOT/'AFLOW_BRATS_LEADERBOARD.md';text=path.read_text();a=text.index('<!-- TEST270_TUNED_NATIVE -->');b=text.index('<!-- END_TEST270_TUNED_NATIVE -->',a)
    section=text[a:b];table_start=section.index('| 排名 | 方法/快照 |')
    table='| 排名 | 方法/快照 | Scan Dice | Subject Dice | Point | 128³前向等价量 | 证据 |\n|---:|---|---:|---:|---:|---:|---|\n'
    table+='\n'.join(f"| {x['rank']} | {x['method']} | {x['scan_dice']:.9f} | {x['subject_dice']:.9f} | {x['point_hits']}/270 | {x.get('forward_equivalents',0):.3f}× | [报告]({x['report_path']}) |" for x in board[key])+'\n'
    text=text[:a]+section[:table_start]+table+text[b:]
    text=re.sub(r'更新时间：[^\n]+','更新时间：2026-10-29 UTC',text,count=1);path.write_text(text)
    link=str((R/'RESULT.md').relative_to(ROOT));body=f'## 2026-10-29：RACE164叠加既有最佳推理已完成\n\n原270/119，RACE原始Scan {raw["scan_dice"]:.6f}，固定8镜像＋解码＋后处理Scan {m["scan_dice"]:.6f}、Subject {m["subject_dice"]:.6f}、Point {m["point_hits"]}/270；同权重差{change["scan_delta"]:+.6f}，描述性95%区间{change["scan_delta_ci95"]}。训练0、2160病例前向，两个输出共用8视图。继承test-tuned历史，无当前新搜索或outer/confirmation。本轮结束不自动重测或续训。[结果]({link})。'
    for name in ('AGENTS.md','AFLOW_BRATS_HANDOFF_CURRENT.md','AFLOW_BRATS_RESULTS.md','AFLOW_BRATS_WEEKLY_REPORT.md'):block(name,body)
    card=body+'\n\n**一句话：** 固定模型，从多个方向看图后融合一致意见。\n**出发点：** RACE尚无270成绩，旧推理组合在w32上最好。\n**具体做法：** 固定164轮、8镜像一致性加权、图像脑区体积解码与两项结构后处理。\n**只改变/固定：** 仅推理流程；数据、权重、栅格固定，无训练。\n**是否支持直觉：** '+str(supported)+'；描述性复用队列证据，非独立泛化。\n**代价/下一步：** 每例8次前向，共享identity；结果保留，本轮结束。'
    block('AFLOW_BRATS_METHOD_CATALOG.md',card)
    write(ROOT/'archive/research_records/2026-10-28_race_seg/TEST_ACCESS_ADDENDUM_20261029.json',dict(at=now(),
          new_explicit_user_authorization=True,checkpoint_epoch=164,checkpoint_sha256=CHECKPOINT_SHA,
          report_path=link,test_scans=270,test_subjects=119,case_forwards=2160,
          historical_training_test_access_false_preserved=True,training_resumed=False))
    write(R/'PIPELINE_STATUS.json',dict(at=now(),state='complete',training=False,test_access=True,
          scans=270,subjects=119,inference_Dice_support=supported,current_test_parameter_search=False))


if __name__=='__main__':main()
