"""Scoped publication of this arm; preserve historical scores and evidence."""
import fcntl
import re
import subprocess
from scripts.nnunet_resenc_20261013.common import *

MARK='NNUNET_RESENC_20261013';REL=str(R.relative_to(ROOT))

def replace_block(path,body,append=False):
    path=Path(path);text=path.read_text();block=f'<!-- {MARK} -->\n{body}\n<!-- END_{MARK} -->'
    pattern=rf'<!-- {MARK} -->.*?<!-- END_{MARK} -->'
    if re.search(pattern,text,re.S):text=re.sub(pattern,lambda _:block,text,flags=re.S)
    elif append:text+='\n\n'+block+'\n'
    else:
        pos=text.find('\n');text=text[:pos+1]+'\n'+block+'\n'+text[pos+1:]
    path.write_text(text)

def progress(stage,detail=''):
    state=read(OUT/'STATUS.json') if (OUT/'STATUS.json').exists() else {}
    epoch=state.get('last_complete_epoch',0);target=state.get('target_epochs','待冻结')
    brief=f'**2026-10-13残差编码器参考：{stage}。** worker1132660双A10040GB，单T1ce原972/108，当前完整{epoch}/{target}轮。{detail} 270阶段以本组访问收据为准；无新outer/confirmation。[状态]({REL}/PIPELINE_STATUS.json) · [七问]({REL}/IDEA.md)。'
    replace_block(ROOT/'AFLOW_BRATS_HANDOFF_CURRENT.md',brief)
    replace_block(ROOT/'AGENTS.md',f'## 2026-10-13完整残差编码器参考：用户明确实现授权\n\n'
       f'- 用户明确“你去实现吧”，授权worker1132660双A10040GB、24小时内一组单T1ce完整nnU-Net残差编码器参考及条件有限融合。当前阶段：{stage}；{epoch}/{target}完整轮。{detail}\n'
       f'- 原972/108/270与患者隔离固定，仅972用于规划和梯度；108原生Scan选优。训练截止2026-10-14北京时间10:11:45，整体截止12:41:45；正式更新预算看EXECUTION_FREEZE，不套旧200轮。\n'
       f'- 最终唯一模型允许一次270探索性评测；条件六项融合规则见FUSION_PLAN，全部test-tuned，不称独立泛化；无outer/confirmation，不自动新增训练组或复活历史任务。真实状态见`{REL}/PIPELINE_STATUS.json`及workspace STATUS。')
    replace_block(ROOT/'AFLOW_BRATS_RESULTS.md',f'## 2026-10-13｜完整残差编码器参考\n\n'+brief,append=True)
    catalog=ROOT/'AFLOW_BRATS_METHOD_CATALOG.md'
    text=catalog.read_text()
    if f'<!-- {MARK} -->' not in text:
        anchor='## 达到0.8的新候选'
        pos=text.index(anchor)
        text=text[:pos]+f'<!-- {MARK} -->\n<!-- END_{MARK} -->\n\n'+text[pos:]
        catalog.write_text(text)
    support='候选仍在执行，不能预言达到0.8。'
    next_step='完成工程门后按冻结预算运行，所有结果如实归档。'
    if stage=='完成' and (R/'RESULT.json').exists():
        result=read(R/'RESULT.json')
        support=('本批内部定向开发点估计高于旧榜首，仍非独立泛化证据。' if result['improved_old_best'] else '本批最高未超过旧榜首，不支持这次固定配方带来总体提分；不否定整类方法。')
        next_step='本组结束，不自动扩组、换epoch或重测。'
    card=f'''## 方法8：完整残差编码器参考｜{stage}

**一句话：** 用一套不同的完整训练流程学习单T1ce，并检验与SegResNet的互补。  
**出发点：** 原生270最高融合0.775891，多个局部改动未形成明确突破。  
**具体做法：** 原972单独规划nnU-Net残差编码器L，固定108验证选优；整体改变网络、增强、采样及优化，不称单模块消融。  
**结果：** 当前完整{epoch}/{target}轮。{detail}  
**是否支持直觉：** {support}  
**代价/下一步：** 双A10040GB、资源24小时上限，较大网络与滑窗部署增加成本。{next_step}旧SegResNet与固定193对照保留。[协议与状态]({REL}/IDEA.md)。'''
    replace_block(catalog,card)

def checks():
    results=[]
    for s in ['verify_aflow_brats_method_catalog.py','verify_aflow_brats_leaderboard.py','verify_aflow_brats_doc_layout.py','verify_aflow_root_artifacts.py']:
        p=subprocess.run([str(BASE_PY),'scripts/'+s],cwd=ROOT,env=env(),capture_output=True,text=True)
        results.append({'script':s,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr})
    write(R/'DOC_CHECKS.json',results)
    if any(r['returncode'] for r in results):raise RuntimeError('documentation checks failed')

def leaderboard(reports,checkpoint,training):
    with (ROOT/'archive/reviews/leaderboard_publication.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        path=ROOT/'AFLOW_BRATS_LEADERBOARD.json';board=read(path);key='test270_tuned_native_rank_by_scan'
        entries=board[key];old={r['method']:{k:v for k,v in r.items() if k!='rank'} for r in entries}
        checkpoint_sha=sha(checkpoint)
        for report_path in reports:
            report=read(report_path);m=report['metrics'];identifier=report['method']
            if identifier in old:raise RuntimeError('new result already published; reconcile instead of overwrite')
            entries.append({'method':identifier,**{k:m[k] for k in ['scan_dice','subject_dice','point_hit','point_hits','scan_ci95','subject_ci95']},
               'point_denominator':270,'cohort':{'scans':270,'subjects':119},
               'grid':'native1mm crop; padding excluded','score_grid':'native','test_tuned':True,
               'checkpoint_path':str(checkpoint.relative_to(ROOT)),'checkpoint_sha256':checkpoint_sha,
               'checkpoint_epoch':training['best_epoch'],'threshold':report['threshold'],'tta':False,'postprocessing':False,
               'forward_equivalents':report['forward_equivalents'],
               'forward_cost_note':'input-voxel equivalents only; different architecture FLOPs and latency are not normalized',
               'role':'candidate' if report['single_model'] else 'fixed_output_panel',
               'logit_formula':report['formula'],'report_path':str(report_path.relative_to(ROOT)),
               'report_sha256':sha(report_path),'evidence_status':'test-tuned, reused internal, not independent generalization',
               'new_outer_confirmation_access':False,'training_scope':training,
               'paired_comparisons':report['comparisons']})
        entries.sort(key=lambda r:(-r['scan_dice'],r['forward_equivalents'],r['method']))
        for i,item in enumerate(entries,1):item['rank']=i
        for item in entries:
            if item['method'] in old and {k:v for k,v in item.items() if k!='rank'}!=old[item['method']]:raise RuntimeError('historical row changed')
        board[key]=entries;board['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();write(path,board)
        single=[r for r in entries if r.get('role')!='fixed_output_panel' and 'single' in r.get('logit_formula','')]
        board['latest_nnunet_resenc_20261013']={'overall_native_test_tuned':{k:entries[0][k] for k in ['method','scan_dice','subject_dice','point_hits']},
          'single_model_native_test_tuned':{k:max(single,key=lambda r:r['scan_dice'])[k] for k in ['method','scan_dice','subject_dice','point_hits']}}
        write(path,board)
        table='\n'.join(f"| {r['rank']} | {r['method']} | {r['scan_dice']:.9f} | {r['subject_dice']:.9f} | {r['point_hits']}/270 | {r['forward_equivalents']:.3f}× | [报告]({r['report_path']}) |" for r in entries)
        body=('<!-- TEST270_TUNED_NATIVE -->\n## 270例定向开发：原始栅格，test-tuned，独立分榜\n\n'
          '统一原生1mm裁剪栅格，排除填充；全部复用内部开发，非独立泛化。不同网络的输入体素前向等价量不代表计算量或延迟相同，真实成本见各报告。历史分数保留。\n\n'
          '| 排名 | 方法/快照 | Scan Dice | Subject Dice | Point | 128³前向等价量 | 证据 |\n'
          '|---:|---|---:|---:|---:|---:|---|\n'+table+'\n<!-- END_TEST270_TUNED_NATIVE -->')
        md=ROOT/'AFLOW_BRATS_LEADERBOARD.md';text=md.read_text()
        assert len(re.findall('<!-- TEST270_TUNED_NATIVE -->',text))==1
        md.write_text(re.sub(r'<!-- TEST270_TUNED_NATIVE -->.*?<!-- END_TEST270_TUNED_NATIVE -->',lambda _:body,text,flags=re.S))
