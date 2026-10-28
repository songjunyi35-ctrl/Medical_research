"""Statistics and publication only; never invokes a predictor."""
import copy
import datetime
import re
import subprocess
import numpy as np
from scripts.segresnet_inference_test270_20261028.common import *
from scripts.segresnet_inference_search_20261028.summarize import replace_block


def audit(raw):
    index = read(CACHE / 'index.json')
    labels = np.memmap(CACHE / 'labels.u8', np.uint8, 'r', shape=(270, *CANVAS))
    lookup = {d['case_id']:d for d in raw['predictions']}
    validate_rows(raw['per_case'])
    for row in raw['per_case']:
        entry = lookup[row['case_id']]
        assert sha(ROOT / entry['path']) == entry['sha256']
        saved = np.load(ROOT / entry['path'])
        shape = tuple(saved['shape'])
        mask = np.unpackbits(saved['mask_bits'])[:int(np.prod(shape))].reshape(shape).astype(bool)
        g = index['geometry'][row['case_id']]
        assert shape == tuple(g['extent'])
        sl = tuple(slice(o,o+e) for o,e in zip(g['canvas_offset'],g['extent']))
        target = labels[(row['source_slot'], *sl)] > 0
        tp, fp, fn = [int(a.sum()) for a in [mask & target, mask & ~target, ~mask & target]]
        assert [tp,fp,fn] == [row['tp'],row['fp'],row['fn']]
        den = 2*tp+fp+fn
        value = float(np.float32(2*tp)/np.float32(den)) if den else 1.
        assert value == row['locator_dsc']
        point = tuple(saved['point'])
        assert list(point) == row['automatic_point'] and bool(target[point]) == row['automatic_point_hits_gt']
    write(R / 'INDEPENDENT_MASK_AUDIT.json', dict(at=now(), status='passed', scans=270,
          masks_sha_and_counts_exact=True, points_exact=True, patient_model_forward=False))


def references():
    from scripts.segresnet_w32_test270_20261028.publish import references as older
    old = older()
    source = OLD / 'RAW_TEST270.json'
    records = read(source)['summary']['per_case']
    validate_rows(records)
    return dict(original_w32=dict(path=str(source.relative_to(ROOT)), sha256=sha(source), per_case=records),
                segresnet_fusion=old['segresnet_fusion'], nnunet_single525=old['nnunet_single525'])


def board_update(result):
    path = ROOT / 'AFLOW_BRATS_LEADERBOARD.json'
    board = read(path)
    before = copy.deepcopy(board)
    key = 'test270_tuned_native_rank_by_scan'
    original = next(d for d in board[key] if d['method']=='segresnet_width32_native_epoch171_test270_20261028')
    row = copy.deepcopy(original)
    row.update(method=ID, scan_dice=result['scan_dice'], subject_dice=result['subject_dice'],
          point_hit=result['point_hit'], point_hits=result['point_hits'],
          threshold=None, threshold_probability=None, raw_logit_threshold=None, threshold_raw_logit=None,
          threshold_space='per-case probability-volume adaptive, temperature2', comparison_operator='top-k',
          tta=True, postprocessing=True, locator_forwards=5, ensemble_size=1, views=[2,3,4,6,7],
          forward_equivalents=float(5*np.prod(CANVAS)/128**3),
          forward_cost_note='five forwards of one width32 model plus deterministic volume decoding; voxel equivalence, not FLOPs/latency',
          report_path=str((R/'RAW_TEST270.json').relative_to(ROOT)), report_sha256=sha(R/'RAW_TEST270.json'),
          result_path=str((R/'RESULT.json').relative_to(ROOT)), result_sha256=sha(R/'RESULT.json'),
          scan_ci95=result['scan_ci95'], subject_ci95=result['subject_ci95'],
          evidence_status='test-informed development; recipe frozen on reused108 validation before this270 access; no current test retuning',
          logit_formula='mean sigmoid of five inverse-mirrored logits; temperature2; expected-volume ratio top-k',
          paired_comparisons=result['comparisons'], inference_validation_utility_gate=True,
          selected_recipe_path=str((R/'SELECTED_RECIPE.json').relative_to(ROOT)),
          selected_recipe_sha256=sha(R/'SELECTED_RECIPE.json'), validation_gate_passed=False)
    ranked = [d for d in board[key] if d['method']!=ID] + [row]
    ranked.sort(key=lambda d:(-d['scan_dice'], d['forward_equivalents'], d['method']))
    for i,d in enumerate(ranked,1):d['rank']=i
    for old in before[key]:
        current=next(d for d in ranked if d['method']==old['method'])
        assert {k:v for k,v in old.items() if k!='rank'}=={k:v for k,v in current.items() if k!='rank'}
    board[key]=ranked
    board['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    board['latest_segresnet_inference_selected_test270_20261028']=dict(method=ID, rank=row['rank'],
          fixed_recipe=True, current_test_retuning=False)
    write(path,board)
    md=ROOT/'AFLOW_BRATS_LEADERBOARD.md'
    text=md.read_text()
    table='\n'.join(f"| {d['rank']} | {d['method']} | {d['scan_dice']:.9f} | {d['subject_dice']:.9f} | {d['point_hits']}/270 | {d['forward_equivalents']:.3f}× | [报告]({d['report_path']}) |" for d in ranked)
    block='''<!-- TEST270_TUNED_NATIVE -->
## 270例定向开发：原始栅格，test-tuned，独立分榜

本榜为复用内部探索，非独立泛化。五镜像＋体积解码配方仅在108验证选择后固定，本次不按270测试调参。统一原生1mm裁剪评分、排除填充；原始171轮阈值0.5历史行保留。前向等价量仅折算输入体素，不代表不同宽度/架构的FLOPs或实测耗时。

| 排名 | 方法/快照 | Scan Dice | Subject Dice | Point | 128³前向等价量 | 证据 |
|---:|---|---:|---:|---:|---:|---|
''' + table + '\n<!-- END_TEST270_TUNED_NATIVE -->'
    text=re.sub(r'<!-- TEST270_TUNED_NATIVE -->.*?<!-- END_TEST270_TUNED_NATIVE -->',lambda _:block,text,flags=re.S)
    md.write_text(text)
    write(R/'LEADERBOARD_UPDATE.json',dict(at=now(), method=ID, rank=row['rank'], all_historical_rows_preserved=True))


def main():
    verify()
    raw=read(R/'RAW_TEST270.json')
    audit(raw)
    refs=references()
    cases=raw['per_case']
    metrics=summary_ci(cases)
    metrics['point_hits']=sum(d['automatic_point_hits_gt'] for d in cases)
    contrasts={key:paired(cases,d['per_case']) for key,d in refs.items()}
    primary=contrasts['original_w32']
    gate=bool(primary['scan_delta']>0 and primary['scan_delta_ci95'][0]>0 and
              primary['subject_delta']>=0 and primary['point_hits_delta']>=0)
    result=dict(at=now(), status='complete_test270', method=ID, **metrics,
          comparisons=contrasts, descriptive_support_gate=gate, runtime=raw['runtime'],
          reference_summaries={k:summary_ci(v['per_case']) for k,v in refs.items()},
          reference_reports={k:{x:v[x] for x in ['path','sha256']} for k,v in refs.items()},
          checkpoint_sha256=CHECKPOINT_SHA, selected_recipe_sha256=sha(R/'SELECTED_RECIPE.json'),
          grid=raw['grid'], training=False, test_access=True, current_test_retuning=False,
          test_tuned_development=True, not_pristine_blind=True, new_outer_confirmation_access=False,
          original_training_validation_gate_passed=False, inference_validation_utility_gate=True,
          cache_payloads_and_index_historical_exact=True)
    write(R/'RESULT.json',result)
    backup=R/'docs_before_publication'
    backup.mkdir(exist_ok=True)
    for name in ['AGENTS.md','AFLOW_BRATS_HANDOFF_CURRENT.md','AFLOW_BRATS_METHOD_CATALOG.md',
                 'AFLOW_BRATS_RESULTS.md','AFLOW_BRATS_WEEKLY_REPORT.md','AFLOW_BRATS_LEADERBOARD.md','AFLOW_BRATS_LEADERBOARD.json']:
        if not (backup/name).exists():shutil.copy2(ROOT/name,backup/name)
    board_update(result)
    verdict=('配对区间支持本复用队列的探索性正差' if primary['scan_delta_ci95'][0]>0 else
             '配对区间为负，当前配方较原始推理差' if primary['scan_delta_ci95'][1]<0 else
             '配对区间跨0，尚未确认稳定增益或等效')
    lines=['# 固定五镜像＋体积解码：270测试完成','',
        '原270扫描/119患者，单T1ce、同32通道171轮权重和原生1mm裁剪评分。新增训练0，唯一验证选定配方，不作本次test参数搜索。','',
        '| 方案 | Scan Dice | Subject Dice | Point |','|---|---:|---:|---:|',
        f"| 历史原始171轮 | {result['reference_summaries']['original_w32']['scan_dice']:.9f} | {result['reference_summaries']['original_w32']['subject_dice']:.9f} | 256/270 |",
        f"| 五镜像＋体积解码 | **{metrics['scan_dice']:.9f}** | **{metrics['subject_dice']:.9f}** | **{metrics['point_hits']}/270** |",'',
        f"Scan差 **{primary['scan_delta']:+.9f}**，119患者整簇配对95%区间 **{primary['scan_delta_ci95']}**；Subject差{primary['subject_delta']:+.9f}，95%区间{primary['subject_delta_ci95']}，Point差{primary['point_hits_delta']:+d}。{verdict}，预先描述性支持门={gate}。",'',
        '视图2/3/4/6/7还原后等权概率平均；固定温度2，稳定排序＋CPU float64前缀累加的期望体积比值近似。不平滑、不删连通域、不采用固定二值阈值，标签只用于评分。不是精确RankDice。',
        f"单A10080GB，batch8/尾6；270例每例五视图恰一次，1350病例前向。包含输入加载、推理、解码、评分、压缩掩膜及journal耗时{raw['runtime']['seconds']:.2f}秒；排除模型加载和缓存复制。显存reserved峰值{raw['runtime']['peak_reserved_gib']:.2f}GiB。输入体素等价约{raw['runtime']['input_voxel_equivalents_128cubed']:.3f}个128立方前向，不等于FLOPs或延迟倍数。",'',
        '同历史原生图像、标签、index三项SHA全部一致；直接复制旧缓存，本次新增N4=0。沿用历史测试画布160×208×168，仅比验证画布末轴多8个零填充体素，以容纳元数据所示最大161体素裁剪；评分排除全部填充，框外历史4阳性体素处理不变。',
        '此队列已经多次复用，模型/方向test-informed；本配方在本次270模型访问前冻结，但不称独立盲测或独立泛化。与旧原始171报告作保存结果配对，硬件/批量及实现数值差异不冒称逐位匹配；没有额外重跑参考模型。保存270预测的SHA、TP/FP/FN、Dice和Point全部独立核对。',
        '无新outer/confirmation/外部访问；完整评估结束，不按分数扩搜、换模型/参数或自动重测。', '',
        '[原始逐例结果](RAW_TEST270.json) · [机器结果](RESULT.json) · [固定配方](SELECTED_RECIPE.json) · [独立掩膜复核](INDEPENDENT_MASK_AUDIT.json)']
    (R/'RESULT.md').write_text('\n'.join(lines)+'\n')
    rel=str(R.relative_to(ROOT))
    outcome=f"270/119 Scan{metrics['scan_dice']:.6f}、Subject{metrics['subject_dice']:.6f}、Point{metrics['point_hits']}/270；对同权重原始推理差{primary['scan_delta']:+.6f}，配对95%区间{primary['scan_delta_ci95']}，{verdict}。"
    marker='INFERENCE_SELECTED_TEST270_20261028'
    replace_block('AGENTS.md',marker,f'''## 2026-10-28五镜像＋体积解码固定270测试已完成

用户明确“你去270个test上推理就行了”，仅授权验证已选同一w32第171轮五镜像＋温度2体积解码。{outcome} 单A10080GB，原生载荷/index逐SHA相同，batch8/尾6，每例五视图恰一次；训练0，无本次test调参或新outer/confirmation。内部复用test-informed探索，完整评估结束，不自动扩搜/换参/重测。旧验证搜索的false访问标志保留为历史阶段，本次新测试访问见新归档；独立RACE训练不受影响。[结果]({rel}/RESULT.md)。''')
    replace_block('AFLOW_BRATS_HANDOFF_CURRENT.md',marker,f'''**2026-10-28固定五镜像＋体积解码270测试完成。** {outcome} 模型/配方从108选择后固定，本次无test调参，训练0、无新outer/confirmation。复用内部探索，非独立泛化；本批结束不扩搜/重测，不干预独立RACE训练。[报告]({rel}/RESULT.md)。''')
    replace_block('AFLOW_BRATS_METHOD_CATALOG.md',marker,f'''## 2026-10-28：无训练推理配方固定270测试｜complete

**一句话：** 用同一模型的五次镜像判断与概率体积解码改善输出。
**出发点：** 108验证搜索最高0.761779；不能将验证增量加在旧测试0.773228上。
**具体做法：** 固定同一171轮权重、视图2/3/4/6/7、温度2与确定性体积规则，仅新增270一次评估；单T1ce、缓存与原生评分不变。
**结果：** {outcome}
**是否支持直觉：** 描述性支持门={gate}；复用测试探索，不称独立泛化。
**代价/下一步：** 每例5次单模型前向与解码，训练0；全270掩膜独立核验，结束不自动扩搜或重测。[核对材料]({rel}/RESULT.md)。''')
    for name,title in [('AFLOW_BRATS_RESULTS.md','固定验证选出的推理配方测270'),('AFLOW_BRATS_WEEKLY_REPORT.md','不训练模型的推理改进固定测试')]:
        replace_block(name,marker,f'''## 2026-10-28：{title}

固定同一w32第171轮、验证选定五镜像＋温度2体积解码。{outcome} 原270/119、单T1ce、原生SHA不变，训练0，每例5前向；复用内部探索无本次test调参，非独立泛化，无新outer/confirmation，评估结束不自动重测。[报告]({rel}/RESULT.md)。''')
    write(SEARCH/'TEST_ACCESS_ADDENDUM_20261028.json',dict(at=now(), new_stage=str(R.relative_to(ROOT)),
          test_access_completed=True, scans=270, subjects=119, checkpoint_sha256=CHECKPOINT_SHA,
          validation_stage_reports_unchanged=True, no_current_test_retuning=True))
    checks={}
    for script in ['verify_aflow_brats_method_catalog.py','verify_aflow_brats_leaderboard.py','verify_aflow_brats_doc_layout.py','verify_aflow_root_artifacts.py']:
        p=subprocess.run([str(PY),str(ROOT/'scripts'/script)],cwd=ROOT,env=env(),capture_output=True,text=True)
        checks[script]=dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
    write(R/'DOC_CHECKS.json',checks)
    assert all(d['returncode']==0 for d in checks.values()), 'document/leaderboard verification failed'
    write(R/'COMPLETION_AUDIT.json',dict(at=now(),status='passed',scans=270,subjects=119,
          checkpoint_sha256=sha(CHECKPOINT),files={name:sha(R/name) for name in ['RESULT.md','RESULT.json','RAW_TEST270.json',
          'MODEL_PREACCESS.json','SELECTED_RECIPE.json','CACHE_READY.json','EXECUTION_FREEZE.json','INDEPENDENT_MASK_AUDIT.json']},
          model_forward_in_publication=False))
    status('complete',test_access_completed=True,scan_dice=metrics['scan_dice'],subject_dice=metrics['subject_dice'],point_hits=metrics['point_hits'])


if __name__=='__main__':main()
