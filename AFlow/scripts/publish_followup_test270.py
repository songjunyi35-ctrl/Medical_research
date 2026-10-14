"""Publish fixed candidate scores and reference mapping; preserve every historical row."""
import sys,re,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.followup_test270_common import *
TITLES={'native':'公开预训练＋原分辨率组合166','patient':'预训练＋患者等权损失172'}
IDS={'native':'segresnet_pretrain_native_epoch166_test270_20260921','patient':'segresnet_patient_equal_epoch172_test270_20260921'}
REF_ID='segresnet_pretrained179_native_reference_20260921'


def block(path,mark,text):
    s=path.read_text();value=f'<!-- {mark} -->\n{text}\n<!-- END_{mark} -->';pattern=f'<!-- {mark} -->.*?<!-- END_{mark} -->'
    s=re.sub(pattern,lambda _:value,s,flags=re.S) if re.search(pattern,s,flags=re.S) else s.rstrip()+'\n\n'+value+'\n'
    path.write_text(s)


def row_for(name,d):
    r=RECORDS[name];rel=str(r.relative_to(ROOT));is_native=name=='native'
    return {'method':IDS[name],'scan_dice':d['scan_dice'],'subject_dice':d['subject_dice'],'point_hit':d['point_hit'],
      'point_hits':d['point_hits'],'point_denominator':270,'locator_forwards':1,'cohort':{'scans':270,'subjects':119},
      'checkpoint_epoch':EPOCHS[name],'checkpoint_path':rel+'/checkpoint_selected.pth','checkpoint_sha256':HASHES[name],
      'resolution':'native_1mm_crop_canvas_160x208x168' if is_native else 128,'score_grid':'native' if is_native else '128',
      'threshold':0.,'tta':False,'postprocessing':False,'sam':False,'distillation':False,
      'scan_dice_95ci':d['scan_ci95'],'subject_dice_95ci':d['subject_ci95'],
      'report_path':rel+'/RAW_TEST270.json','report_sha256':sha(r/'RAW_TEST270.json'),
      'training_scope':{'train_scans':972,'train_subjects':444,'validation_scans':108,'validation_subjects':50,'completed_epochs':200,
        'optimizer_updates':24400,'public_pretraining_scans':334,'public_pretraining_epochs':150},
      'evidence_status':'user_authorized_failed_validation_gate_exploratory; reused_internal; prior_test_informed; no_current_test_tuning; not_pristine_blind; not_external',
      'validation_gate_passed':False,'status':'fixed validation-selected checkpoint, one authorized test; paired CIs determine interpretation',
      'comparability_note':'native grid, padding excluded; compare only with native-grid rows' if is_native else 'same canonical128 test cache; native-grid scores are not directly comparable',
      'paired_comparisons':d['comparisons']}


def reference_row():
    raw=json.loads((S/'PRETRAINED179_NATIVE_RAW.json').read_text());m=summary_ci(raw['summary']['per_case'])
    return {'method':REF_ID,'scan_dice':m['scan_dice'],'subject_dice':m['subject_dice'],'point_hit':m['point_hit'],
      'point_hits':sum(x['automatic_point_hits_gt'] for x in raw['summary']['per_case']),'point_denominator':270,
      'locator_forwards':1,'cohort':{'scans':270,'subjects':119},'checkpoint_epoch':179,
      'checkpoint_path':str((PRETRAINED/'checkpoint_selected.pth').relative_to(ROOT)),'checkpoint_sha256':REF_SHA,
      'resolution':'128^3 inference -> native crop trilinear logit scoring','score_grid':'native','threshold':0.,'tta':False,
      'postprocessing':False,'sam':False,'scan_dice_95ci':m['scan_ci95'],'subject_dice_95ci':m['subject_ci95'],
      'report_path':str((S/'PRETRAINED179_NATIVE_RAW.json').relative_to(ROOT)),'report_sha256':sha(S/'PRETRAINED179_NATIVE_RAW.json'),
      'evidence_status':'already_tested_model_same_cohort_reference_reprojection; reused_internal; prior_test_informed; no_retuning; not_external',
      'status':'reference scoring view of existing pretrained179, not a new training candidate',
      'comparability_note':'native grid; same checkpoint as128 pretrained179 row, scores differ because evaluation raster differs; not an algorithm gain',
      'grid_mapping':{'from':[128,128,128],'to':'image-only native crop extent','mode':'trilinear','align_corners':False,'threshold_after_mapping':0.}}


def board_update(name,d):
    p=ROOT/'AFLOW_BRATS_LEADERBOARD.json';board=json.loads(p.read_text());old=board['test270_direct_locator_rank_by_scan']
    mdp=ROOT/'AFLOW_BRATS_LEADERBOARD.md';md=mdp.read_text();start=md.index('## Reused internal test270');end=md.find('\n## ',start+3);end=len(md) if end<0 else end
    section=md[start:end];lines=[s for s in section.splitlines() if re.match(r'^\|\s*\d+\s*\|',s)];assert len(lines)==len(old)
    rendered={r['method']:line for r,line in zip(old,lines)};new=[row_for(name,d)]
    primary='native173' if name=='native' else 'pretrained179';c=d['comparisons'][primary];ci=c['scan_delta_ci95']
    grid='native grid' if name=='native' else '128³'
    rendered[IDS[name]]=f"| 0 | {TITLES[name]}，**{grid}** | `{d['scan_dice']:.9f}` | `{d['subject_dice']:.9f}` | {d['point_hits']}/270 (`{d['point_hit']:.6f}`) | `1×` | 用户授权未过验证门探索；对{primary}差{c['scan_delta']:+.6f}，95%CI[{ci[0]:+.6f},{ci[1]:+.6f}]；不跨栅格相减 |"
    if name=='native':
        rr=reference_row();new.append(rr)
        rendered[REF_ID]=f"| 0 | 预训练179固定参考，128³预测→**native grid**评分 | `{rr['scan_dice']:.9f}` | `{rr['subject_dice']:.9f}` | {rr['point_hits']}/270 (`{rr['point_hit']:.6f}`) | `1×` | 同一已测试模型的不同评分栅格；非第三训练候选，不能把重评分差当方法增益 |"
    rows=[r for r in old if r['method'] not in {x['method'] for x in new}]+new;rows.sort(key=lambda r:-r['scan_dice'])
    for i,r in enumerate(rows,1):r['rank']=i
    candidates128=[r for r in rows if r.get('resolution')==128]
    native_ids={IDS['native'],REF_ID,'native_resolution_pure_segresnet_full972_epoch173_20260920'}
    candidates_native=[r for r in rows if r['method'] in native_ids]
    leaders={'128':max(candidates128,key=lambda r:r['scan_dice']),'native':max(candidates_native,key=lambda r:r['scan_dice'])}
    # Supersede current-record wording explicitly, retaining every original numeric row.
    if leaders['128']['method']!='segresnet_finetune_pretrained_20260921':
        oldid='segresnet_finetune_pretrained_20260921'
        if oldid in rendered:rendered[oldid]=rendered[oldid].replace('当前128³组总体及1× Dice最高点估计','历史128³组Dice纪录')
        for r in rows:
            if r['method']==oldid:r['status']='historical128 point-estimate record; current grid leader updated below; original failed gate and paired CIs preserved'
    if 'native_resolution_pure_segresnet_full972_epoch173_20260920' in rendered:
        k='native_resolution_pure_segresnet_full972_epoch173_20260920';rendered[k]=rendered[k].replace('栅格与其他行不同，不可直接相减','native grid；仅与同栅格行比较')
    table='\n'.join(re.sub(r'^\|\s*\d+\s*\|',f"| {r['rank']} |",rendered[r['method']]) for r in rows)
    first=section.index(lines[0]);last=section.index(lines[-1])+len(lines[-1]);md=md[:start]+section[:first]+table+section[last:]+md[end:]
    board['test270_direct_locator_rank_by_scan']=rows;board['updated_utc']=now()
    board['latest_followup_test270_20260921']={'score_grid_leaders':{k:{x:v[x] for x in ['method','scan_dice','subject_dice','point_hits']} for k,v in leaders.items()},
         'scope':'automatic same-cohort scores, overall and1x Scan point estimates per grid; no cross-grid or significance claim',
         'completed_candidates':[k for k,r in RECORDS.items() if (r/'RESULT.json').exists()]}
    write(p,board);md=re.sub(r'更新时间：[^\n]+','更新时间：2026-09-21 UTC',md,count=1);mdp.write_text(md)
    lead=f"最新固定候选评测后：128³组Scan最高点估计{leaders['128']['scan_dice']:.6f}（{leaders['128']['method']}）；native grid组为{leaders['native']['scan_dice']:.6f}（{leaders['native']['method']}）。均须按同栅格配对区间解释，参考179的native行是同模型重评分，不是新训练增益。"
    block(mdp,'FOLLOWUP_TEST270_GRID_LEADERS',lead)


def main(name):
    verify();r=RECORDS[name];d=json.loads((r/'RESULT.json').read_text());raw=json.loads((r/'RAW_TEST270.json').read_text())
    validate_rows(raw['summary']['per_case']);assert sha(r/'checkpoint_selected.pth')==d['checkpoint_sha256']
    for p,h in d['evidence'].items():assert sha(ROOT/p)==h,p
    rel=str(r.relative_to(ROOT));title=TITLES[name];grid='原始1mm裁剪栅格（native grid），排除填充' if name=='native' else '128³栅格'
    text=f'# {title}：270复用内部测试完成\n\n候选由108验证选定第{d["epoch"]}轮，用户授权后固定评测。270扫描/119患者，{grid}，Scan Dice **{d["scan_dice"]:.9f}**，95%区间{d["scan_ci95"]}；Subject **{d["subject_dice"]:.9f}**，95%区间{d["subject_ci95"]}；Point **{d["point_hits"]}/270**。\n\n| 同栅格参考 | 参考Scan | 当前−参考 | 患者配对95%区间 | 97.5%区间 | 点命中差 |\n|---|---:|---:|---|---|---:|\n'
    for key,c in d['comparisons'].items():text+=f'| {key} | {d["reference_summaries"][key]["scan_dice"]:.6f} | {c["scan_delta"]:+.6f} | {c["scan_delta_ci95"]} | {c["scan_delta_ci97_5"]} | {c["point_hits_delta"]:+d} |\n'
    text+='\n10000次患者整簇重采样，绝对seed20260917、配对seed20260918。区间跨0时不确定，单次点估计提高不等于稳定改进。验证明确提升门未过的状态保留；测试是用户另行授权的探索性评估，不能事后改写晋级门。不同栅格不能直接相减或据裸排名宣称优越。\n\n'
    text+='本组想法和固定因素见[七问](IDEA.md)。单T1ce、固定原972/108/270、纯SegResNet、阈值0、1次前向，无标签定位/点击、镜像平均、融合或后处理。组合仅改变预训练初始化加native输入；患者等权仅在训练中改损失系数，评测完全不加权。\n\n'
    text+='这是reused internal、prior-test-informed开发，非新盲测或外部确认；没有本次test选择/调参或额外outer/confirmation。没有重训、换epoch/阈值，结果正负均保留。\n\n'
    text+=f'本候选缓存就绪后推理{d["runtime"]["inference_seconds"]:.2f}秒，峰值{d["runtime"]["peak_allocated_bytes"]/2**30:.2f}GiB，独立快照{(r/"checkpoint_selected.pth").stat().st_size/2**20:.2f}MiB。native组另需约7.03GiB临时测试缓存及固定179参考前向，不能把这些成本隐去；128³复用原缓存。\n\n[机器结果](RESULT.json) · [逐例报告](RAW_TEST270.json) · [选优收据](SELECTION_RECEIPT.json) · [模型访问收据](MODEL_PREACCESS.json) · [模型](checkpoint_selected.pth)。\n'
    (r/'RESULT.md').write_text(text);board_update(name,d)
    primary='native173' if name=='native' else 'pretrained179';c=d['comparisons'][primary];ci=c['scan_delta_ci95']
    brief=f"{title}：test270/119完成，{grid}，Scan {d['scan_dice']:.6f}、Subject {d['subject_dice']:.6f}、Point {d['point_hits']}/270；对{primary}差{c['scan_delta']:+.6f}、95%CI[{ci[0]:+.6f},{ci[1]:+.6f}]。验证门未过，用户授权探索性测试；无当前test调优或新outer/confirmation。[完整报告]({rel}/RESULT.md)。"
    mark='FOLLOWUP_TEST_'+name.upper()
    for fname in ['AFLOW_BRATS_HANDOFF_CURRENT.md','AFLOW_BRATS_WEEKLY_REPORT.md','AFLOW_BRATS_RESULTS.md']:block(ROOT/fname,mark,brief)
    p=ROOT/'AFLOW_BRATS_METHOD_CATALOG.md';s=p.read_text();prefix='## 公开预训练初始化＋原始分辨率微调｜' if name=='native' else '## 预训练SegResNet患者等权损失｜'
    a=s.index(prefix);b=s.find('\n## ',a+3);b=len(s) if b<0 else b;card=s[a:b]
    card=re.sub(r'｜[^\n]+','｜200轮及用户授权test270完成',card,count=1)
    card=re.sub(r'\*\*结果：\*\*[^\n]+','**结果：** '+brief,card)
    verdict='该配对95%区间为正，但仍为复用队列探索。' if ci[0]>0 else ('该配对95%区间为负，当前候选较差。' if ci[1]<0 else '该配对区间跨0，未确认稳定增益。')
    card=re.sub(r'\*\*是否支持直觉：\*\*[^\n]+','**是否支持直觉：** '+verdict+'完整其他对照及定位结果见报告；不把test结果改写成验证门已通过，不否定一般方法。',card)
    card=re.sub(r'\*\*代价/下一步：\*\*[^\n]+',f'**代价/下一步：** 已完成训练及此次固定候选测试，部署1次前向、模型约18MiB；本次缓存就绪后270推理{d["runtime"]["inference_seconds"]:.1f}秒。native另有大画布/缓存成本。当前停止，不自动重测或调参。[核对材料]({rel}/RESULT.md)。',card)
    p.write_text(s[:a]+card+s[b:])
    train=TRAIN_RECORDS[name]
    write(train/'TEST_ACCESS_ADDENDUM.json',{'utc':now(),'experiment_test_access':True,'selected_epoch':d['epoch'],
          'result_path':rel+'/RESULT.json','result_sha256':sha(r/'RESULT.json'),'scope':'training RESULT/FINAL_RESULT false flags remain validation-stage history only'})
    tp=train/'RESULT.md';marker='后续用户授权test270已完成'
    if marker not in tp.read_text():tp.write_text(tp.read_text()+'\n**'+marker+'：** '+brief.replace(']('+rel,' ](../'+r.name)+' 上文无test只描述训练/验证阶段。\n')
    reports=[]
    for script in ['verify_aflow_brats_leaderboard.py','verify_aflow_brats_method_catalog.py','verify_aflow_brats_doc_layout.py','verify_aflow_root_artifacts.py']:
        x=subprocess.run([sys.executable,'scripts/'+script],cwd=ROOT,capture_output=True,text=True)
        reports.append({'name':script,'returncode':x.returncode,'stdout':x.stdout,'stderr':x.stderr})
    write(r/'DOC_CHECKS.json',reports)
    if any(x['returncode'] for x in reports):raise RuntimeError('document validation failed')
    write(r/'DONE.json',{'utc':now(),'status':'complete_test270','result_sha256':sha(r/'RESULT.json'),'all_four_validators_passed':True})


if __name__=='__main__':main(sys.argv[1])
