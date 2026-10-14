"""Archive one verified four-output panel and mechanically update both leaderboards."""
import argparse
import copy
from datetime import datetime,timezone
import json
from pathlib import Path
import re
import tempfile
from scripts.structured_background_protocol import ROOT,checked,sha,write_new
from scripts.verify_aflow_brats_leaderboard import validate_leaderboard


def atomic_text(path,text):
    with tempfile.NamedTemporaryFile('w',dir=path.parent,delete=False,encoding='utf-8') as handle:
        handle.write(text);temporary=Path(handle.name)
    temporary.replace(path)


def table_region(markdown):
    section=markdown.index('## Reused internal test270：全自动direct locator')
    header='| 排名 | 方法 | Scan Dice | Subject Dice | Point hit | 成本 | 状态 |\n|---:|---|---:|---:|---:|---:|---|\n'
    start=markdown.index(header,section)+len(header)
    end=markdown.index('\n\n',start)
    return start,end


def parse_existing_table(markdown,rows):
    start,end=table_region(markdown);lines=markdown[start:end].splitlines()
    if len(lines)!=len(rows):raise ValueError('canonical Markdown/JSON row counts disagree')
    result={}
    for row,line in zip(rows,lines,strict=True):
        columns=[part.strip() for part in line.strip().strip('|').split('|')]
        if len(columns)!=7 or int(re.sub(r'[^0-9]','',columns[0]))!=row['rank']:raise ValueError('canonical Markdown rank mismatch')
        if f"{row['scan_dice']:.9f}" not in columns[2] or f"{row['subject_dice']:.9f}" not in columns[3]:raise ValueError('canonical Markdown metric mismatch')
        result[row['method']]=columns
    return result


def render_leaderboard(markdown,old_rows,ranked_rows,new_ids,report_link):
    old=parse_existing_table(markdown,old_rows);lines=[]
    for row in ranked_rows:
        if row['method'] in new_ids:
            status='本次固定面板；复用非盲；完整结果/CI见报告'
            columns=[str(row['rank']),row['display_name'],f"`{row['scan_dice']:.9f}`",f"`{row['subject_dice']:.9f}`",f"{row['point_hits']}/270 (`{row['point_hit']:.6f}`)",f"`{row['locator_forwards']}×`",status]
        else:
            columns=old[row['method']].copy();columns[0]=str(row['rank']);columns[1]=columns[1].replace('**','')
            for i in (2,3,4):columns[i]=columns[i].replace('**','')
            if not columns[6].startswith('历史状态：'):columns[6]='历史状态：'+columns[6]
        lines.append('| '+' | '.join(columns)+' |')
    start,end=table_region(markdown)
    best=ranked_rows[0];single=next(row for row in ranked_rows if row.get('locator_forwards')==1)
    note=(f"\n\n2026-09-10 当前同test270/119的整体Scan点估计最高：`{best['scan_dice']:.6f}`（{best['method']}）；"
          f"1× Scan最高：`{single['scan_dice']:.6f}`（{single['method']}）。只表示此复用队列排名，不自动表示显著性或外部泛化。"
          f"本次所有四输出（含对照、负结果、失败门）均已入表，[本次结果]({report_link})。"
          "以下旧关键增益及旧行状态文字保留其历史时点，现排名以本表/JSON rank为准；legacy160³行的栅格限制仍保留。")
    return re.sub(r'^更新时间：[^\n]*','更新时间：2026-09-10 UTC',markdown[:start]+'\n'.join(lines)+note+markdown[end:],count=1,flags=re.MULTILINE)


def archive(study,record,destination):
    result_path=destination/'RESULT.json'
    if result_path.exists():raise FileExistsError('test archive already exists; use --stage leaderboard for pending canonical update')
    raw_path=record/'RAW_TEST270.json';raw=json.loads(raw_path.read_text());verification_path=record/'TEST_VERIFICATION.json';verification=json.loads(verification_path.read_text())
    assert verification['status']=='passed';checked(raw_path,verification['raw_sha256'])
    pre_path=record/'TEST_PREACCESS.json';checked(pre_path,raw['test_preaccess_sha256']);pre=json.loads(pre_path.read_text())
    assert len(raw['results'])==4 and raw['new_forwards']==4320
    destination.mkdir(parents=True,exist_ok=True)
    for name,path in [('RAW_TEST270.json',raw_path),('TEST_PREACCESS.json',pre_path),('TEST_VERIFICATION.json',verification_path),('TEST_MASK_INDEX.json',record/'TEST_MASK_INDEX.json')]:
        with (destination/name).open('xb') as f:f.write(path.read_bytes())
        checked(destination/name,sha(path))
    full=study=='weak_elastic_full864'
    if full:
        seed=2026091031;fit_scans=864;fit_subjects=391;names={'full_control':'全864固定对照','full_elastic':'全864弱形变'}
        admission=json.loads((record/'TRAIN_PREACCESS.json').read_text());checked(record/'TRAIN_PREACCESS.json',pre['training_preaccess_sha256'])
        sampling='subject_equal_within_native' if admission['selected_sampling_policy']=='subject_elastic' else 'native_volume'
        motivation='先在fit684/dev180两seed检验弱形变；过门后用全授权864/391人从头固定配对重训，原dev成为训练，不再选优。两臂仅新增形变与否不同，抽样层由开发规则固定，无教师。'
        train_gpu_hours=sum(json.loads(Path(r['train_report']['path']).read_text())['elapsed_seconds'] for r in pre['reports'].values())/3600
    else:
        seed=2026090711;fit_scans=684;fit_subjects=313;names={'control':'fit684原硬标签对照','candidate':'fit684体积占比标签'}
        sampling='native_volume'
        motivation='仅把训练WT标签改成精确格内占比，图像/抽样/原损失/128³/阈值不变。开发两seed过门后，测试使用事前固定seed11而非较好seed；两模型只fit684/313，不称full864，无教师。'
        development=json.loads((record/'RESULT.json').read_text());train_gpu_hours=sum(v['seconds'] for v in development['checkpoints'].values())/3600
    leaderboard_rows=[]
    for key,summary in raw['results'].items():
        mode='flip8' if key.endswith('_flip8') else 'identity';arm=key[:-(len(mode)+1)]
        checkpoint=pre['reports'][arm]['checkpoint'];checked(checkpoint['path'],checkpoint['sha256'])
        method=f'{study}_20260910_{key}_epoch168_seed{seed}'
        row={'rank':0,'method':method,'display_name':f"{names[arm]}，seed{seed} e168，{'八镜像' if mode=='flip8' else '原输出'}",'scan_dice':summary['scan_dice'],'subject_dice':summary['subject_dice'],'point_hit':summary['point_hits']/270,'point_hits':summary['point_hits'],'point_denominator':270,'cohort':{'scans':270,'subjects':119},'locator_forwards':8 if mode=='flip8' else 1,'checkpoint_path':str(Path(checkpoint['path']).relative_to(ROOT)),'checkpoint_sha256':checkpoint['sha256'],'resolution':128,'threshold':0.,'tta':mode=='flip8','tta_formula':'ordered float32 mean of inverse-reflected raw logits over all8 axis reflection combinations' if mode=='flip8' else 'identity','sam':False,'postprocessing':False,'scan_dice_95ci':summary['scan_cluster_95ci'],'scan_ci_unit':'patient_cluster','subject_dice_95ci':summary['subject_95ci'],'bootstrap_samples':10000,'bootstrap_seed':pre['bootstrap']['seed'],'training_scope':{'fit_scans':fit_scans,'fit_subjects':fit_subjects,'former_development180_consumed_as_training':full,'fixed_epoch':168,'seed':seed,'teacher_checkpoints':[]},'report_path':str((destination/'RAW_TEST270.json').relative_to(ROOT)),'report_sha256':sha(destination/'RAW_TEST270.json'),'prediction_mask_index_path':str((destination/'TEST_MASK_INDEX.json').relative_to(ROOT)),'prediction_mask_index_sha256':sha(destination/'TEST_MASK_INDEX.json'),'evidence_status':'user_authorized_fixed_panel_after_reused_development_gates; reused_internal_test; not_pristine_blind; not_external; historical_test_informed_pipeline; no_new_panel_retuning','test_tuned':False,'test_tuning_scope':raw['test_tuning_scope'],'status':'completed_fixed_panel; no_result_dependent_retest_or_parameter_change','comparability_note':'same fixed128 braincrop label rasterization as the old128 internal rows; legacy160 row remains a different label rasterization'}
        row['training_scope']['sampling_policy']=sampling
        row['training_factor']=('weak_elastic_p0.2_max2vox' if arm=='full_elastic' else 'original_augmentation') if full else ('exact_cell_occupancy_target' if arm=='candidate' else 'original_nearest_exact_hard_target')
        leaderboard_rows.append(row)
    metrics={key:{k:v for k,v in summary.items() if k!='per_case'} for key,summary in raw['results'].items()}
    best=max(metrics,key=lambda key:metrics[key]['scan_dice']);best1=max((key for key in metrics if key.endswith('_identity')),key=lambda key:metrics[key]['scan_dice'])
    result={'schema':'aflow.verified_test270_panel.result','created_utc':datetime.now(timezone.utc).isoformat(),'study':study,'status':'completed_fixed_reused_test_panel','motivation_and_single_factor':motivation,'metrics':metrics,'comparisons':raw['comparisons'],'gates':raw['gates'],'candidate_overall_scientific_gate':raw['candidate_overall_scientific_gate'],'best_of_this_fixed_panel_by_scan':best,'best_1x_of_this_fixed_panel_by_scan':best1,'new_training_stage_gpu_hours':train_gpu_hours,'test_forwards':4320,'test_wall_seconds_including_mask_archival':raw['seconds'],'test_peak_gpu_allocated_bytes':raw['peak_gpu_allocated_bytes'],'prediction_masks_bytes':raw['prediction_masks']['bytes'],'raw_sha256':sha(destination/'RAW_TEST270.json'),'preaccess_sha256':sha(destination/'TEST_PREACCESS.json'),'verification_sha256':sha(destination/'TEST_VERIFICATION.json'),'mask_index_sha256':sha(destination/'TEST_MASK_INDEX.json'),'access':raw['access'],'test_tuning_scope':raw['test_tuning_scope'],'leaderboard_rows':leaderboard_rows,'further_test_tuning_or_retest_allowed':False}
    write_new(result_path,result)
    lines=['# 固定test270面板：'+study,'',motivation,'',
           '同一270扫描/119患者、旧128³硬GT/阈值0/最高logit Point。该队列已多次复用，历史测试和流水线已知；本次没有按新test结果调参，不是盲测、外部确认或公开SOTA。','',
           '| 固定输出 | Scan Dice | Subject Dice | Point /270 | forward |','|---|---:|---:|---:|---:|']
    for row in leaderboard_rows:lines.append(f"| {row['display_name']} | {row['scan_dice']:.9f} | {row['subject_dice']:.9f} | {row['point_hits']} | {row['locator_forwards']}× |")
    lines += ['',f"本次面板最高Scan输出为{best}（{metrics[best]['scan_dice']:.9f}），本次1×最高为{best1}（{metrics[best1]['scan_dice']:.9f}）；这不是根据test改选预定候选或部署方案。",'',
              f"预定candidate整体Scan是否超过旧0.7572696616942132：{raw['gates']['candidate_overall_scan_lead']}；candidate 1×是否超过旧0.7520031755617648：{raw['gates']['candidate_identity_1x_scan_lead']}；完整CI/Subject/Point门：{raw['candidate_overall_scientific_gate']}。点估计领先与统计支持分开。",'',
              '绝对与配对区间均为固定10,000次患者簇百分位95%CI，非多比较校正；不是训练随机性的完整不确定性。旧榜原绝对CI不被本区间替换。']
    for key,c in raw['comparisons'].items():lines.append(f"\n- {key}：Scan差{c['scan_delta']:+.6f} CI {c['scan_cluster_95ci']}；Subject差{c['subject_delta']:+.6f} CI {c['subject_95ci']}；Point差{c['point_delta']:+d}。")
    lines += ['',f"本后续训练阶段新增{train_gpu_hours:.3f}GPU小时；本面板4320forwards，耗时{raw['seconds']:.2f}s含保存掩膜，peak{raw['peak_gpu_allocated_bytes']/1024**3:.3f}GiB。保存二值预测270MiB，可从掩膜独立重算Dice；只核查Point坐标GT命中，未独立重算未保存logits的argmax。",'',
              '部署为4.70M单checkpoint，原输出1×或事前固定八镜像8×，不含N4预处理成本；四输出全部入榜，包括对照/负结果/失败门。未新增oldconfirmation、fixed validation、outer或external性能。训练来源Kaggle mirror与官方字节parity未证明限制仍保留。', '',
              '此测试面板关闭，不允许按其结果重选epoch、阈值、权重、病例或追加重测。机制是否获支持应分别看candidate对matched control和旧榜的预定对比，不以某一项点估计替代联合门。','',
              '[机器结果](RESULT.json)、[raw](RAW_TEST270.json)、[测试前收据](TEST_PREACCESS.json)、[独立掩膜/区间验证](TEST_VERIFICATION.json)、[预测索引](TEST_MASK_INDEX.json)。', '',
              f"[原实验七问](../2026-09-10_{study}/IDEA.md)、[原预注册](../2026-09-10_{study}/PREREG.md)。Raw SHA `{result['raw_sha256']}`；RESULT SHA `{sha(result_path)}`。"]
    for row in leaderboard_rows:
        if row['tta']:continue
        lines.append(f"\n- [{row['display_name']} checkpoint](../../../{row['checkpoint_path']}) SHA `{row['checkpoint_sha256']}`。")
    with (destination/'RESULT.md').open('x') as f:f.write('\n'.join(lines)+'\n')
    return result


def update_leaderboards(destination):
    result=json.loads((destination/'RESULT.json').read_text());new_rows=result['leaderboard_rows'];new_ids={r['method'] for r in new_rows}
    json_path=ROOT/'AFLOW_BRATS_LEADERBOARD.json';md_path=ROOT/'AFLOW_BRATS_LEADERBOARD.md'
    current_json=json_path.read_text();current_md=md_path.read_text();value=json.loads(current_json)
    existing={r['method'] for r in value['test270_direct_locator_rank_by_scan']}
    if existing&new_ids:
        if existing&new_ids!=new_ids:raise RuntimeError('partial existing panel rows; inspect before changing canonical files')
        staged_json=destination/'LEADERBOARD_STAGED.json';staged_md=destination/'LEADERBOARD_STAGED.md';before_md=destination/'LEADERBOARD_BEFORE.md'
        if sha(json_path)!=sha(staged_json):raise RuntimeError('canonical JSON changed after panel update; preserve and inspect')
        if sha(md_path)==sha(before_md):atomic_text(md_path,staged_md.read_text())
        elif sha(md_path)!=sha(staged_md):raise RuntimeError('canonical Markdown changed after panel update; preserve and inspect')
        validate_leaderboard(json_path,md_path);parse_existing_table(md_path.read_text(),value['test270_direct_locator_rank_by_scan'])
        print('panel already canonical; both formats verified',flush=True);return
    old_rows=copy.deepcopy(value['test270_direct_locator_rank_by_scan']);parse_existing_table(current_md,old_rows)
    for row in value['test270_direct_locator_rank_by_scan']:row['status_context_20260910']='legacy status text preserves its historical point; current rank field is authoritative'
    value['test270_direct_locator_rank_by_scan'].extend(copy.deepcopy(new_rows))
    ranked=value['test270_direct_locator_rank_by_scan'];ranked.sort(key=lambda row:row['scan_dice'],reverse=True)
    for rank,row in enumerate(ranked,1):row['rank']=rank
    value['updated_utc']='2026-09-10'
    value['latest_fixed_test_panel_20260910']={'study':result['study'],'result_path':str((destination/'RESULT.json').relative_to(ROOT)),'result_sha256':sha(destination/'RESULT.json'),'all_four_rows_recorded':True,'reused_nonblind':True,'historical_test_informed':True,'new_panel_retuning':False}
    value['current_internal_scan_leaders_20260910']={'overall':ranked[0]['method'],'one_forward':next(r['method'] for r in ranked if r.get('locator_forwards')==1)}
    new_md=render_leaderboard(current_md,old_rows,ranked,new_ids,str((destination/'RESULT.md').relative_to(ROOT)))
    new_json=json.dumps(value,indent=2,ensure_ascii=False)+'\n'
    artifacts={'LEADERBOARD_BEFORE.json':current_json,'LEADERBOARD_BEFORE.md':current_md,'LEADERBOARD_STAGED.json':new_json,'LEADERBOARD_STAGED.md':new_md}
    for name,body in artifacts.items():
        path=destination/name
        if path.exists():
            if path.read_text()!=body:raise RuntimeError(f'preserve conflicting prior staging: {path}')
        else:
            with path.open('x') as f:f.write(body)
    validate_leaderboard(destination/'LEADERBOARD_STAGED.json',destination/'LEADERBOARD_STAGED.md')
    parse_existing_table(new_md,ranked)
    if json_path.read_text()!=current_json or md_path.read_text()!=current_md:raise RuntimeError('concurrent canonical changes detected; nothing overwritten')
    atomic_text(json_path,new_json);atomic_text(md_path,new_md)
    validate_leaderboard(json_path,md_path);parse_existing_table(md_path.read_text(),ranked)
    receipt=destination/'LEADERBOARD_UPDATE.json'
    if not receipt.exists():write_new(receipt,{'status':'both_updated_and_validated','created_utc':datetime.now(timezone.utc).isoformat(),'json_sha256':sha(json_path),'markdown_sha256':sha(md_path),'new_methods':sorted(new_ids),'prior_rows_preserved':len(old_rows),'raw_sha256':result['raw_sha256']})
    print(json.dumps({'leaderboard_updated':True,'test_rows':len(ranked),'current_overall':ranked[0]['method'],'new_panel_methods':sorted(new_ids)},indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--study',choices=['weak_elastic_full864','fractional_occupancy'],required=True);parser.add_argument('--stage',choices=['archive','leaderboard','all'],default='all');args=parser.parse_args()
    record=ROOT/f'archive/research_records/2026-09-10_{args.study}';destination=ROOT/f'archive/research_records/2026-09-10_{args.study}_test270'
    if args.stage in ('archive','all'):archive(args.study,record,destination)
    if args.stage in ('leaderboard','all'):update_leaderboards(destination)


if __name__=='__main__':main()
