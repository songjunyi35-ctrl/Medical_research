"""Archive combination validation only; no test entry point or automatic promotion."""
import sys,json,shutil,re,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.segresnet_pretrained_native_common import *
REL=str(R.relative_to(ROOT));MARK='PRETRAIN_NATIVE_COMBINATION'


def block(path,body):
    s=path.read_text();value=f'<!-- {MARK} -->\n{body}\n<!-- END_{MARK} -->'
    pattern=f'<!-- {MARK} -->.*?<!-- END_{MARK} -->'
    s=re.sub(pattern,lambda _:value,s,flags=re.S) if re.search(pattern,s,flags=re.S) else s.rstrip()+'\n\n'+value+'\n'
    path.write_text(s)


def checks():
    rows=[]
    for name in ['verify_aflow_brats_leaderboard.py','verify_aflow_brats_method_catalog.py','verify_aflow_brats_doc_layout.py','verify_aflow_root_artifacts.py']:
        x=subprocess.run([sys.executable,str(ROOT/'scripts'/name)],cwd=ROOT,capture_output=True,text=True)
        rows.append({'name':name,'returncode':x.returncode,'stdout':x.stdout,'stderr':x.stderr})
    write(R/'VALIDATORS.json',rows)
    if any(x['returncode'] for x in rows):raise RuntimeError('documentation validators failed')


def documents(stage,progress=None,result=None):
    brief=f'公开预训练＋原始分辨率组合：{stage}。固定公开150轮初始化，原972/108、单T1ce、200轮、每卡4/全局8、每轮122更新；按原始栅格验证选优。'
    if progress:brief+=f" 已完成{progress.get('last_complete_epoch',0)}/200轮，当前最佳第{progress.get('best_epoch',0)}轮Scan {progress.get('best_scan_dice',0):.6f}。"
    if result:
        brief+=f" 最终选第{result['selected_epoch']}轮，Scan {result['scan_dice']:.6f}、Subject {result['subject_dice']:.6f}；{'通过' if result['promotion_gate_passed'] else '未通过'}冻结组合提升门。"
    brief+=' 没有本候选test/outer/confirmation访问或test调优。[实验记录]('+REL+'/RESULT.md) · [实时状态]('+REL+'/PIPELINE_STATUS.json)。'
    for name in ['AFLOW_BRATS_HANDOFF_CURRENT.md','AFLOW_BRATS_WEEKLY_REPORT.md']:block(ROOT/name,brief)
    p=ROOT/'AFLOW_BRATS_METHOD_CATALOG.md';s=p.read_text();prefix='## 公开预训练初始化＋原始分辨率微调｜';a=s.index(prefix);b=s.find('\n## ',a+3);b=len(s) if b<0 else b
    card=s[a:b];card=re.sub(r'｜[^\n]+','｜'+stage,card,count=1);card=re.sub(r'\*\*结果：\*\*[^\n]+','**结果：** '+brief,card)
    if result:
        card=re.sub(r'\*\*是否支持直觉：\*\*[^\n]+','**是否支持直觉：** '+('达到冻结组合优势门，但仍为复用验证和单seed探索。' if result['promotion_gate_passed'] else '未达到冻结组合优势门，不证明所有组合均无效。')+'所有比较已统一原始栅格；非独立确认，不能直接把不同栅格涨幅相加。',card)
    p.write_text(s[:a]+card+s[b:])
    if result:block(ROOT/'AFLOW_BRATS_RESULTS.md','## 2026-09-21｜公开预训练与原分辨率组合验证完成\n\n'+brief)


def contrast(rows,refs,coefficients):
    ids=[r['case_id'] for r in rows];groups=sorted({r['subject_id'] for r in rows})
    maps={key:{r['case_id']:r for r in value} for key,value in refs.items()}
    assert all(set(m)==set(ids) for m in maps.values())
    values=np.array([sum(weight*maps[key][cid]['locator_dsc'] for key,weight in coefficients.items()) for cid in ids])
    n=np.array([sum(r['subject_id']==s for r in rows) for s in groups])
    sums=np.array([sum(v for r,v in zip(rows,values) if r['subject_id']==s) for s in groups])
    ix=np.random.default_rng(20260918).integers(0,len(groups),(10000,len(groups)))
    scan=sums[ix].sum(1)/n[ix].sum(1);subject=(sums/n)[ix].mean(1)
    return {'scan_delta':float(sums.sum()/n.sum()),'subject_delta':float((sums/n).mean()),
            'scan_delta_ci95':np.quantile(scan,[.025,.975]).tolist(),'scan_delta_ci97_5':np.quantile(scan,[.0125,.9875]).tolist(),
            'subject_delta_ci95':np.quantile(subject,[.025,.975]).tolist(),'coefficients':coefficients}


def main():
    verify();status=json.loads((OUT/'STATUS.json').read_text());assert status['status']=='completed' and status['last_complete_epoch']==200
    assert json.loads((R/'RELOAD_CHECK.json').read_text())['status']=='passed'
    logs=[json.loads(x) for x in (OUT/'TRAIN.jsonl').read_text().splitlines() if x.strip()]
    assert [x['epoch'] for x in logs]==list(range(1,201)) and all(x['unique_scans']==972 and x['optimizer_updates']==122 for x in logs)
    sel=json.loads((OUT/'BEST.json').read_text());report=OUT/sel['report'];assert sha(report)==sel['report_sha256'] and sha(OUT/'best.pth')==sel['checkpoint_sha256']
    raw=json.loads(report.read_text());rows=raw['per_case'];d=summary_ci(rows)
    panel=json.loads((R/'REFERENCE_NATIVE_VALIDATION.json').read_text())['references'];refs={k:v['per_case'] for k,v in panel.items()};refs['combination']=rows
    comparisons={k:contrast(rows,refs,{'combination':1,k:-1}) for k in panel}
    for key,c in comparisons.items():
        c['point_hits_delta']=sum(x['automatic_point_hits_gt'] for x in rows)-sum(x['automatic_point_hits_gt'] for x in refs[key])
    strong=max(panel[k]['summary']['scan_dice'] for k in ['native173','pretrained179'])
    passed=d['scan_dice']>=strong+.01 and all(comparisons[k]['scan_delta_ci97_5'][0]>0 for k in ['native173','pretrained179'])
    checkpoint=R/'checkpoint_selected.pth'
    if not checkpoint.exists():shutil.copy2(OUT/'best.pth',checkpoint)
    assert sha(checkpoint)==sel['checkpoint_sha256'];write(R/'RAW_SELECTED_VALIDATION.json',raw)
    write(R/'SELECTION_RECEIPT.json',{'utc':now(),**sel,'selection_role':'reused validation108 native grid','test_access':False})
    d.update(status='completed200_validation_only',created_utc=now(),selected_epoch=sel['epoch'],completed_epochs=200,
        optimizer_updates=24400,checkpoint_sha256=sel['checkpoint_sha256'],comparisons=comparisons,
        interaction_descriptive=contrast(rows,refs,{'combination':1,'native173':-1,'pretrained179':-1,'pure193':1}),
        promotion_gate_passed=passed,baseline_reference_summaries={k:v['summary'] for k,v in panel.items()},
        grid='native crop, padding excluded',training_seconds=sum(x['training_seconds'] for x in logs),
        elapsed_seconds=status['elapsed_seconds'],peak_allocated_bytes_per_gpu=max(x['peak_allocated_bytes_per_gpu'] for x in logs),
        new_test_pixels_access=False,new_outer_confirmation_access=False,prior_test_informed_development=True,test_tuning_used=False)
    paths=[checkpoint,report,OUT/'initial.pth',OUT/'latest.pth',OUT/'TRAIN.jsonl',R/'FREEZE.json',R/'TRAIN_PREACCESS.json',
           R/'VALIDATION_PREACCESS.json',R/'DATA_AUDIT.json',R/'REFERENCE_PREACCESS.json',R/'REFERENCE_NATIVE_VALIDATION.json',
           R/'RELOAD_CHECK.json',R/'RAW_SELECTED_VALIDATION.json',R/'SELECTION_RECEIPT.json']
    d['evidence']={str(p.relative_to(ROOT)):sha(p) for p in paths};write(R/'RESULT.json',d);write(R/'FINAL_RESULT.json',d)
    text=f'# 公开预训练＋原分辨率：200轮完成，仅108验证\n\n最佳第{sel["epoch"]}轮，Scan {d["scan_dice"]:.9f}、Subject {d["subject_dice"]:.9f}；Scan患者整簇95%区间{d["scan_ci95"]}。固定108扫描/50患者，原始裁剪栅格，排除填充。\n\n| 同原始栅格参考 | 参考Scan | 组合−参考 | 配对95%区间 | 两主比较97.5%区间 |\n|---|---:|---:|---|---|\n'
    for name,c in comparisons.items():text+=f'| {name} | {panel[name]["summary"]["scan_dice"]:.6f} | {c["scan_delta"]:+.6f} | {c["scan_delta_ci95"]} | {c["scan_delta_ci97_5"]} |\n'
    text+=f'\n冻结组合提升门：{"通过" if passed else "未通过"}；要求比更强单项至少+0.01且对两个单项的97.5%区间下界均>0。仍是复用验证选优、单seed探索，不能宣称独立确认或严格因果协同。交互差仅描述性：{d["interaction_descriptive"]}。\n\n'
    text+='复用公开334预训练150轮，原972以1mm画布微调200轮；主要研究变化相对原native为初始化来源。训练保持native数值设置，本组/参考的验证统一确定性评分，历史参考选择模式差异已披露。[七问与冻结](IDEA.md)。\n\n'
    text+=f'双A100微调约{d["elapsed_seconds"]/3600:.2f}小时，峰值{d["peak_allocated_bytes_per_gpu"]/2**30:.2f}GiB/卡；模型约{checkpoint.stat().st_size/2**20:.2f}MiB，部署1次原始分辨率前向。无新test/outer/confirmation，未按test调参；不自动重复或测试其他模型。\n\n[机器结果](RESULT.json) · [逐例验证](RAW_SELECTED_VALIDATION.json) · [固定模型](checkpoint_selected.pth) · [选优收据](SELECTION_RECEIPT.json)\n'
    (R/'RESULT.md').write_text(text);documents('200轮完成，仅验证',status,d);checks()
    write(R/'DONE.json',{'utc':now(),'status':'complete_validation_only','result_sha256':sha(R/'RESULT.json'),'test_access':False})


if __name__=='__main__':main()
