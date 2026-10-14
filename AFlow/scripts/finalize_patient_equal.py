import sys,json,shutil,re,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.patient_equal_common import *
REL=str(R.relative_to(ROOT));MARK='PATIENT_EQUAL_LOSS'


def block(path,body):
    s=path.read_text();text=f'<!-- {MARK} -->\n{body}\n<!-- END_{MARK} -->';pattern=f'<!-- {MARK} -->.*?<!-- END_{MARK} -->'
    s=re.sub(pattern,lambda _:text,s,flags=re.S) if re.search(pattern,s,flags=re.S) else s.rstrip()+'\n\n'+text+'\n';path.write_text(s)


def documents(stage,result=None):
    text='患者等权损失128³组：'+stage+'；单T1ce、公开150轮初始化、全972各一次/每轮122更新、单A100/global8、目标200轮。'
    if result:text+=f" 最佳第{result['selected_epoch']}轮，108验证Scan {result['scan_dice']:.6f}，{'通过' if result['promotion_gate_passed'] else '未通过'}明确增益门。"
    text+=' 无本候选test/outer/confirmation或test定向训练。[实验记录]('+REL+'/RESULT.md) · [阶段]('+REL+'/PIPELINE_STATUS.json)。'
    for name in ['AFLOW_BRATS_HANDOFF_CURRENT.md','AFLOW_BRATS_WEEKLY_REPORT.md']:block(ROOT/name,text)
    p=ROOT/'AFLOW_BRATS_METHOD_CATALOG.md';s=p.read_text();prefix='## 预训练SegResNet患者等权损失｜';a=s.index(prefix);b=s.find('\n## ',a+3);b=len(s) if b<0 else b
    card=s[a:b];card=re.sub(r'｜[^\n]+','｜'+stage,card,count=1);card=re.sub(r'\*\*结果：\*\*[^\n]+','**结果：** '+text,card)
    if result:card=re.sub(r'\*\*是否支持直觉：\*\*[^\n]+','**是否支持直觉：** '+('达到冻结门，仍是单seed复用验证探索。' if result['promotion_gate_passed'] else '当前实现未通过明确提升门，不证明患者平衡一般无效。')+'扫描Dice仍为主指标，没有改评分口径追求正结果。',card)
    p.write_text(s[:a]+card+s[b:])
    if result:block(ROOT/'AFLOW_BRATS_RESULTS.md','## 2026-09-21｜患者等权损失验证完成\n\n'+text)


def checks():
    reports=[]
    for s in ['verify_aflow_brats_leaderboard.py','verify_aflow_brats_method_catalog.py','verify_aflow_brats_doc_layout.py','verify_aflow_root_artifacts.py']:
        p=subprocess.run([sys.executable,'scripts/'+s],cwd=ROOT,capture_output=True,text=True)
        reports.append({'name':s,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr})
    write(R/'VALIDATORS.json',reports)
    if any(x['returncode'] for x in reports):raise RuntimeError('documentation validators failed')


def main():
    verify();state=json.loads((OUT/'STATUS.json').read_text());assert state['status']=='completed' and state['last_complete_epoch']==200
    assert json.loads((R/'RELOAD_CHECK.json').read_text())['status']=='passed'
    logs=read_rows(OUT/'TRAIN.jsonl');assert [x['epoch'] for x in logs]==list(range(1,201))
    assert all(x['unique_scans']==972 and x['updates']==122 for x in logs)
    selected=json.loads((OUT/'BEST.json').read_text());p=OUT/selected['report'];assert sha(p)==selected['report_sha256'] and sha(OUT/'best.pth')==selected['checkpoint_sha256']
    raw=json.loads(p.read_text());d=summary_ci(raw['per_case']);control=json.loads(PRETRAINED_REPORT.read_text());base=json.loads(BASELINE_REPORT.read_text())
    c=paired(raw['per_case'],control['per_case']);b=paired(raw['per_case'],base['per_case'])
    passed=c['scan_delta']>=.01 and c['scan_delta_ci95'][0]>0 and c['subject_delta']>=0
    snapshot=R/'checkpoint_selected.pth'
    if not snapshot.exists():shutil.copy2(OUT/'best.pth',snapshot)
    assert sha(snapshot)==selected['checkpoint_sha256']
    write(R/'RAW_SELECTED_VALIDATION.json',raw);write(R/'SELECTION_RECEIPT.json',{'utc':now(),**selected,'test_access':False})
    d.update(status='completed200_validation_only',utc=now(),selected_epoch=selected['epoch'],completed_epochs=200,optimizer_updates=24400,
        comparison_vs_pretrained179=c,comparison_vs_pure193=b,promotion_gate_passed=passed,elapsed_seconds=state['elapsed_seconds'],
        peak_allocated_bytes=max(x['peak_allocated_bytes'] for x in logs),new_test_pixels_access=False,new_outer_confirmation_access=False,
        prior_test_informed_development=True,test_tuning_used=False,checkpoint_sha256=selected['checkpoint_sha256'])
    evidence=[snapshot,OUT/'initial.pth',OUT/'latest.pth',OUT/'TRAIN.jsonl',p,R/'FREEZE.json',R/'WEIGHTS.json',R/'TRAIN_PREACCESS.json',
              R/'VALIDATION_PREACCESS.json',R/'DATA_AUDIT.json',R/'RELOAD_CHECK.json',R/'RAW_SELECTED_VALIDATION.json',R/'SELECTION_RECEIPT.json']
    d['evidence']={str(p.relative_to(ROOT)):sha(p) for p in evidence};write(R/'RESULT.json',d);write(R/'FINAL_RESULT.json',d)
    (R/'RESULT.md').write_text(f'''# 患者等权损失：200轮完成，仅108验证

最佳第{selected['epoch']}轮，Scan Dice {d['scan_dice']:.9f}、Subject {d['subject_dice']:.9f}、Point {d['point_hit']:.6f}。固定108扫描/50患者、128³未加权评分。

相对预训练179：Scan差{c['scan_delta']:+.6f}、患者配对95%CI{c['scan_delta_ci95']}、97.5%CI{c['scan_delta_ci97_5']}；相对固定193差{b['scan_delta']:+.6f}、95%CI{b['scan_delta_ci95']}。{'通过' if passed else '未通过'}冻结明确提升门；复用验证与单seed不能当独立确认。

训练从同公开150轮初始化，只对原Dice+0.1BCE使用训练患者扫描数逆权重。每轮972各一次，不更换数据或主指标；单GPU/global8相对历史双rank参考的随机数和数值差异已事前记录。单卡耗时{d['elapsed_seconds']/3600:.2f}小时，峰值{d['peak_allocated_bytes']/2**30:.2f}GiB，部署仍1次前向、约18MiB；没有新大影像缓存。

本候选没有test/outer/confirmation访问，没有test定向训练；不自动加组、调权重、改epoch或启动测试。[七问](IDEA.md) · [机器结果](RESULT.json) · [逐例验证](RAW_SELECTED_VALIDATION.json) · [权重表](WEIGHTS.json) · [模型](checkpoint_selected.pth)。
''')
    documents('200轮完成，仅验证',d);checks();write(R/'DONE.json',{'utc':now(),'status':'complete_validation_only','result_sha256':sha(R/'RESULT.json'),'test_access':False})


if __name__=='__main__':main()
