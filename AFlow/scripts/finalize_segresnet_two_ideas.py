"""One validation-selected test per candidate, then archive and update both boards."""
import sys, os, json, time, re, shutil, subprocess, traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from torch.utils.data import DataLoader
from scripts.segresnet_two_ideas_common import *
from scripts.evaluate_t1ce_wt_plain_locator_test270 import TestBrainCropDataset
TITLES={'semantic_regions':'前景语义分解SegResNet','finetune_pretrained':'公开334例预训练后微调SegResNet'}
HEADINGS={'semantic_regions':'前景语义分解后合成整体肿瘤','finetune_pretrained':'公开单T1ce病例预训练后微调'}
TEST_CACHE=ROOT/'workspace/BraTS/cache/t1ce_wt_n4_braincrop_128_20260817/test270'


def gate(c):
    return c['scan_delta']>=.01 and c['scan_delta_ci95'][0]>0 and c['subject_delta']>=0


def update_block(path,mark,body):
    s=path.read_text(); block=f'<!-- {mark} -->\n{body}\n<!-- END_{mark} -->'
    pattern=rf'<!-- {mark} -->.*?<!-- END_{mark} -->'
    s=re.sub(pattern,lambda _:block,s,flags=re.S) if re.search(pattern,s,flags=re.S) else s.rstrip()+'\n\n'+block+'\n'
    path.write_text(s)


def publish(name,d):
    rec=RECORDS[name];rel=str(rec.relative_to(ROOT)); title=TITLES[name];c=d['comparison_vs_e193'];v=d['validation']
    verdict='达到预设验证提升门' if v['promotion_gate_passed'] else '未达到预设验证提升门'
    ci=c['scan_delta_ci95']; evidence='差值区间为正' if ci[0]>0 else ('差值区间为负' if ci[1]<0 else '差值区间跨0')
    (rec/'RESULT.md').write_text(f'''# {title}：200轮与一次内部test270完成

验证选第{d['epoch']}轮；固定108扫描/50患者的Scan Dice为{v['scan_dice']:.9f}，相对193差{v['comparison_vs_e193']['scan_delta']:+.6f}，{verdict}。

复用内部270扫描/119患者，128³同栅格：Scan Dice **{d['scan_dice']:.9f}**，患者整簇95%区间{d['scan_ci95']}；Subject Dice **{d['subject_dice']:.9f}**，区间{d['subject_ci95']}；点命中{d['point_hits']}/270。

原193同128³基线0.759179，配对Scan差{c['scan_delta']:+.6f}、95%区间{ci}、两主比较校正用97.5%区间{c['scan_delta_ci97_5']}；{evidence}。该区间不消除反复使用验证/测试及单seed的局限。原始分辨率e173的0.768578是另一栅格，不能直接相减。

唯一方案因素、固定项、机制和代价见[七问](IDEA.md)。本结果只评价该整体配方，不证明内部各个部件单独有效。预训练候选若有提升，仍含额外数据与计算两方面，未设第三预算对照。保持单T1ce、原972/108/270、阈值0、1次前向，无镜像平均或后处理。

候选在测试前由验证固定；没有当前test调优、额外outer/confirmation或外部性能访问，用户的test-tuning许可未动用。历史test启发身份保留，不能称新盲测。完成后不自动加组、换epoch或追分。

本候选下游训练{d['training_seconds']/3600:.3f}小时，预训练另计{d['pretraining_seconds']/3600:.3f}小时；本次test GPU推理{d['runtime']['inference_seconds']:.2f}秒，峰值{d['runtime']['peak_allocated_bytes']/2**30:.3f}GiB；独立快照{d['checkpoint_bytes']/2**20:.2f}MiB。耗时不含历史N4或下载构建成本。

[机器结果](RESULT.json) · [原始测试](RAW_TEST270.json) · [验证结果](VALIDATION_RESULT.json) · [访问收据](PREACCESS.json) · [选优收据](SELECTION_RECEIPT.json) · [独立权重](checkpoint_selected.pth)
''')
    path=ROOT/'AFLOW_BRATS_LEADERBOARD.json';board=json.loads(path.read_text());old=board['test270_direct_locator_rank_by_scan']
    mdpath=ROOT/'AFLOW_BRATS_LEADERBOARD.md';md=mdpath.read_text();start=md.index('## Reused internal test270')
    end=md.find('\n## ',start+3);end=len(md) if end<0 else end
    section=md[start:end];lines=[s for s in section.splitlines() if re.match(r'^\|\s*\d+\s*\|',s)]
    assert len(lines)==len(old),(len(lines),len(old))
    rendered={r['method']:line for r,line in zip(old,lines)};method='segresnet_'+name+'_20260921'
    row={'method':method,'scan_dice':d['scan_dice'],'subject_dice':d['subject_dice'],'point_hit':d['point_hit'],
         'point_hits':d['point_hits'],'point_denominator':270,'locator_forwards':1,'cohort':{'scans':270,'subjects':119},
         'checkpoint_epoch':d['epoch'],'checkpoint_path':rel+'/checkpoint_selected.pth','checkpoint_sha256':d['checkpoint_sha256'],
         'resolution':128,'threshold':0.,'tta':False,'sam':False,'postprocessing':False,'distillation':False,
         'scan_dice_95ci':d['scan_ci95'],'subject_dice_95ci':d['subject_ci95'],
         'report_path':rel+'/RAW_TEST270.json','report_sha256':sha(rec/'RAW_TEST270.json'),
         'training_scope':{'train_scans':972,'train_subjects':444,'validation_scans':108,'validation_subjects':50,
                           'completed_epochs':200,'optimizer_updates':24400,'pretraining_scans':334 if name=='finetune_pretrained' else 0},
         'evidence_status':'user_authorized_validation_selected; reused_internal; prior_test_informed; no_current_test_tuning; not_pristine_blind; not_external',
         'status':verdict+'; '+evidence,'paired_comparisons':{'vs_e193':c},
         'comparability_note':'same N4 image-only braincrop128 test cache; different source initialization or semantic supervision; native-grid rows not directly comparable'}
    rows=[r for r in old if r['method']!=method]+[row];rows.sort(key=lambda r:-r['scan_dice'])
    rendered[method]=f"| 0 | {title}，epoch{d['epoch']}，128³ | `{d['scan_dice']:.9f}` | `{d['subject_dice']:.9f}` | {d['point_hits']}/270 (`{d['point_hit']:.6f}`) | `1×` | {verdict}；复用内部；对193差{c['scan_delta']:+.6f}，95%CI[{ci[0]:+.6f},{ci[1]:+.6f}] |"
    for rank,r in enumerate(rows,1):r['rank']=rank
    table='\n'.join(re.sub(r'^\|\s*\d+\s*\|',f"| {r['rank']} |",rendered[r['method']]) for r in rows)
    # Replace the existing contiguous data rows, preserving all historical prose/caveats.
    first=section.index(lines[0]);last=section.index(lines[-1])+len(lines[-1])
    md=md[:start]+section[:first]+table+section[last:]+md[end:]
    md=re.sub(r'更新时间：[^\n]+','更新时间：2026-09-21 UTC',md,count=1)
    board['test270_direct_locator_rank_by_scan']=rows;board['updated_utc']=now();write(path,board);mdpath.write_text(md)
    msg=f"{title}：200轮完成，选第{d['epoch']}轮；108验证Scan {v['scan_dice']:.6f}，{verdict}。270/119同128³测试Scan {d['scan_dice']:.6f}、Subject {d['subject_dice']:.6f}、Point {d['point_hits']}/270；对193差{c['scan_delta']:+.6f}、95%CI[{ci[0]:+.6f},{ci[1]:+.6f}]，{evidence}。单模型1次推理；无当前test调优或新outer/confirmation。[结果]({rel}/RESULT.md)。"
    for filename in ['AFLOW_BRATS_HANDOFF_CURRENT.md','AFLOW_BRATS_WEEKLY_REPORT.md','AFLOW_BRATS_RESULTS.md']:
        update_block(ROOT/filename,'TWO_IDEAS_'+name.upper(),msg)
    p=ROOT/'AFLOW_BRATS_METHOD_CATALOG.md';s=p.read_text();a=s.index('## '+HEADINGS[name]+'｜');b=s.find('\n## ',a+3);b=len(s) if b<0 else b
    card=s[a:b];card=re.sub(r'｜[^\n]+','｜200轮与一次test270完成',card,count=1)
    card=re.sub(r'\*\*结果：\*\*[^\n]+','**结果：** '+msg,card)
    card=re.sub(r'\*\*是否支持直觉：\*\*[^\n]+','**是否支持直觉：** '+verdict+'；'+evidence+'，只评价当前单seed实现，不作独立外部或内部部件因果声明。',card)
    card=re.sub(r'\*\*代价/下一步：\*\*[^\n]+',f'**代价/下一步：** 下游双卡{d["training_seconds"]/3600:.2f}小时，预训练另计{d["pretraining_seconds"]/3600:.2f}小时；推理1次，模型约{d["checkpoint_bytes"]/2**20:.1f}MiB。本候选完成，不自动重测或改参数。[核对材料]({rel}/RESULT.md)。',card)
    p.write_text(s[:a]+card+s[b:])
    checks={}
    for script in ['verify_aflow_brats_leaderboard.py','verify_aflow_brats_method_catalog.py','verify_aflow_brats_doc_layout.py','verify_aflow_root_artifacts.py']:
        x=subprocess.run([sys.executable,str(ROOT/'scripts'/script)],cwd=ROOT,capture_output=True,text=True)
        checks[script]={'returncode':x.returncode,'stdout':x.stdout,'stderr':x.stderr}
    write(rec/'DOC_CHECKS.json',checks)
    if any(x['returncode'] for x in checks.values()):raise RuntimeError('publication validator failed')
    write(rec/'DONE.json',{'utc':now(),'status':'complete','result_sha256':sha(rec/'RESULT.json'),'all_four_validators_passed':True})
    write(rec/'STATUS.json',{'utc':now(),'status':'complete_test270','completed_epochs':200,'test_access':True,
          'test_tuning_used':False,'result_sha256':sha(rec/'RESULT.json')})


def main(name):
    verify();rec=RECORDS[name];out=OUT/name
    if (rec/'RESULT.json').exists():publish(name,json.loads((rec/'RESULT.json').read_text()));return
    status=json.loads((out/'STATUS.json').read_text());assert status['status']=='completed' and status['last_complete_epoch']==200
    assert status['reload_check']['passed']
    sel=json.loads((out/'BEST.json').read_text());assert sha(out/'best.pth')==sel['checkpoint_sha256']
    expected=json.loads((out/sel['report']).read_text());assert expected['cases']==108 and expected['subjects']==50
    base=json.loads((BASELINE_RUN/'validation_e193.json').read_text());v=summary_ci(expected['per_case']);vc=paired(expected['per_case'],base['per_case'])
    v.update(comparison_vs_e193=vc,promotion_gate_passed=gate(vc),selected_epoch=sel['epoch'],role='validation108 reused',test_access=False)
    write(rec/'VALIDATION_RESULT.json',v);write(rec/'RAW_VALIDATION.json',expected)
    if not (rec/'checkpoint_selected.pth').exists():shutil.copy2(out/'best.pth',rec/'checkpoint_selected.pth')
    assert sha(rec/'checkpoint_selected.pth')==sel['checkpoint_sha256']
    if not (rec/'SELECTION_RECEIPT.json').exists():write(rec/'SELECTION_RECEIPT.json',{'utc':now(),**sel,'selection':'maximum validation Scan Dice; earliest tie','test_used_for_selection':False,
          'validation_result_sha256':sha(rec/'VALIDATION_RESULT.json'),'training_freeze_sha256':sha(R/'FREEZE.json')})
    if not (rec/'RAW_TEST270.json').exists():
        if (rec/'PREACCESS.json').exists():raise RuntimeError('test access already attempted; refuse automatic inference repeat')
        rows=read_rows(MANIFEST['test']);ds=TestBrainCropDataset(TEST_CACHE,[r['case_id'] for r in rows]);index=ds.index
        assert index['manifest_sha256']==sha(MANIFEST['test']) and index['model_shape']==[128,128,128]
        assert index['preprocessing']==json.loads((TRAIN_CACHE/'index.json').read_text())['preprocessing']
        write(rec/'TEST_FREEZE.json',{'utc':now(),'checkpoint_sha256':sel['checkpoint_sha256'],'epoch':sel['epoch'],
              'selection_receipt_sha256':sha(rec/'SELECTION_RECEIPT.json'),'code_freeze_sha256':sha(R/'FREEZE.json'),
              'grid':128,'threshold':0,'precision':'bfloat16','batch':2,'forwards':1,'tta':False,'postprocessing':False,
              'test_manifest_sha256':sha(MANIFEST['test']),'test_cache_index_sha256':sha(TEST_CACHE/'index.json')})
        write(rec/'PREACCESS.json',{'utc':now(),'test_access':True,'scope':'one frozen validation-selected candidate',
              'freeze_sha256':sha(rec/'TEST_FREEZE.json'),'current_test_tuning':False,'new_outer_confirmation_access':False})
        payload={}
        for filename,key,width in [('images.f32','images_sha256',4),('labels.u8','labels_sha256',1)]:
            p=TEST_CACHE/filename;assert p.stat().st_size==270*128**3*width;payload[filename]=sha(p);assert payload[filename]==index[key]
        write(rec/'CACHE_RECEIPT.json',{'utc':now(),'cache_index_sha256':sha(TEST_CACHE/'index.json'),'payload_sha256':payload})
        torch.set_num_threads(2);seed_all(BASELINE_SEED);device=torch.device('cuda:0');torch.cuda.set_device(device)
        model=model_for(device,name=='semantic_regions');state=torch.load(rec/'checkpoint_selected.pth',map_location='cpu',weights_only=False)
        assert state['epoch']==sel['epoch'];model.load_state_dict(state['model_state_dict'],strict=True);del state
        model.eval();model.requires_grad_(False)
        write(rec/'MODEL_PREACCESS.json',{'utc':now(),'checkpoint_sha256':sel['checkpoint_sha256'],'test_freeze_sha256':sha(rec/'TEST_FREEZE.json')})
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();tick=time.monotonic()
        result=validate(model,DataLoader(ds,batch_size=2,shuffle=False,num_workers=2,pin_memory=True),device);torch.cuda.synchronize()
        runtime={'inference_seconds':time.monotonic()-tick,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'gpu':torch.cuda.get_device_name()}
        assert result['cases']==270 and result['subjects']==119 and [r['case_id'] for r in result['per_case']]==[r['case_id'] for r in rows]
        assert all(np.isfinite(r['locator_dsc']) for r in result['per_case'])
        write(rec/'RAW_TEST270.json',{'utc':now(),'status':'complete_test270','summary':result,'runtime':runtime,'epoch':sel['epoch'],
              'checkpoint_sha256':sel['checkpoint_sha256'],'test_access':True,'current_test_tuning':False,'reused_internal':True})
    raw=json.loads((rec/'RAW_TEST270.json').read_text());cases=raw['summary']['per_case']
    base=json.loads((ROOT/'archive/research_records/2026-09-17_segresnet_plain972_e193_test270/RAW_TEST270.json').read_text())['summary']['per_case']
    d=summary_ci(cases);d.update(status='complete_test270',utc=now(),epoch=sel['epoch'],checkpoint_sha256=sel['checkpoint_sha256'],
          point_hits=sum(x['automatic_point_hits_gt'] for x in cases),point_denominator=270,validation=v,comparison_vs_e193=paired(cases,base),
          runtime=raw['runtime'],training_seconds=status['elapsed_seconds'],
          pretraining_seconds=json.loads((OUT/'pretrain/STATUS.json').read_text())['elapsed_seconds'] if name=='finetune_pretrained' else 0,
          checkpoint_bytes=(rec/'checkpoint_selected.pth').stat().st_size,test_access=True,not_pristine_blind=True,not_external=True,
          prior_test_informed_development=True,current_test_used_for_selection_or_retuning=False,new_outer_confirmation_access=False)
    evidence=[rec/x for x in ['RAW_TEST270.json','RAW_VALIDATION.json','checkpoint_selected.pth','SELECTION_RECEIPT.json','PREACCESS.json','MODEL_PREACCESS.json','CACHE_RECEIPT.json','TEST_FREEZE.json']]
    d['evidence']={str(p.relative_to(ROOT)):sha(p) for p in evidence};write(rec/'RESULT.json',d);publish(name,d)


if __name__=='__main__':
    try:main(sys.argv[1])
    except BaseException as e:
        write(RECORDS[sys.argv[1]]/'FINALIZATION_FAILURE.json',{'utc':now(),'error':repr(e),'traceback':traceback.format_exc()});raise
