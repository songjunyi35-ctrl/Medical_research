"""Complete108-only valid-region experiment from committed raw validation reports."""
import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.segresnet_pretrained_native_common import paired

R = ROOT / 'archive/research_records/2026-10-10_segresnet_valid_region_supervision'
OUT = ROOT / 'workspace/BraTS/segresnet_t1ce_wt/valid_region_20261010'
OLD = ROOT / 'workspace/BraTS/segresnet_t1ce_wt/native_multiscale_resume50_20260929/validation_e166.json'
PURE = ROOT / 'archive/research_records/2026-09-21_segresnet_pretrain_native_res/REFERENCE_NATIVE_VALIDATION.json'
MARK = 'VALID_REGION_20261010'


def now(): return datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
def read(path): return json.loads(Path(path).read_text())
def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''): digest.update(chunk)
    return digest.hexdigest()
def write(path, data):
    path = Path(path); temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n'); temp.replace(path)


def publish_progress(stage, details=''):
    state = read(OUT / 'STATUS.json') if (OUT / 'STATUS.json').exists() else {}
    epoch = state.get('last_complete_epoch', 0)
    rel = str(R.relative_to(ROOT))
    brief = (f'**2026-10-10有效区域监督：{stage}。** worker1127648双A10040GB，单组原972/108、单T1ce原生多尺度SegResNet；'
             f'当前完成{epoch}/200轮，唯一变化为几何有效区域损失，global8/每轮122更新。{details}'
             f'无本候选新270/outer/confirmation访问，test-informed开发。[真实状态]({rel}/PIPELINE_STATUS.json) · [方案与报告]({rel}/RESULT.md)。')
    if not (R/'FINAL_RESULT.json').exists():
        write(R/'RESULT.json',{'updated_at':now(),'state':stage,'last_complete_epoch':epoch,
              'completed_updates':state.get('completed_updates',epoch*122),
              'final_accuracy_available':False,'new_test_model_forward':False,
              'new_outer_confirmation_access':False})
        (R/'RESULT.md').write_text('# 有效区域监督执行进度\n\n'+brief+'\n\n当前阶段没有最终精度结论。CPU与双卡合成检查见[CPU](CPU_CHECKS_V2.json)、[GPU](GPU_SMOKE.json)，配方见[七问](IDEA.md)。\n')
    for name in ['AFLOW_BRATS_HANDOFF_CURRENT.md', 'AFLOW_BRATS_WEEKLY_REPORT.md', 'AFLOW_BRATS_RESULTS.md']:
        path = ROOT / name; value = path.read_text()
        pattern = f'<!-- {MARK} -->.*?<!-- END_{MARK} -->'
        if len(re.findall(pattern, value, flags=re.S)) != 1: raise RuntimeError('publication block mismatch: ' + name)
        body = brief if name != 'AFLOW_BRATS_RESULTS.md' else '## 2026-10-10｜有效区域监督单组执行\n\n' + brief
        path.write_text(re.sub(pattern, lambda _: f'<!-- {MARK} -->\n{body}\n<!-- END_{MARK} -->', value, flags=re.S))


def main():
    stop = read(R / 'TRAIN_STOP.json'); reload = read(R / 'RELOAD_CHECK.json'); frozen = read(R / 'EXECUTION_FREEZE.json')
    if stop['last_complete_epoch'] != 200 or stop['completed_updates'] != 24400 or reload['status'] != 'passed':
        raise RuntimeError('approved full completion/reload boundary not reached')
    for path, digest in frozen['source_sha256'].items():
        if sha(ROOT / path) != digest: raise RuntimeError('frozen source changed: ' + path)
    selected = read(OUT / f"validation_e{stop['best_epoch']:03d}.json")
    baseline = read(OLD); pure_rows = read(PURE)['references']['pure193']['per_case']
    if selected['cases'] != 108 or selected['subjects'] != 50: raise RuntimeError('wrong validation cohort')
    comparison = paired(baseline['per_case'], selected['per_case'])
    vs_pure = paired(pure_rows, selected['per_case'])
    passed = (selected['locator_dsc_mean'] >= .7630921331986233 and comparison['scan_delta_ci95'][0] > 0 and
              selected['locator_subject_dsc_mean'] >= .7396687182784081 and selected['automatic_point_hit_rate'] >= 94/108)
    result = {'updated_at': now(), 'status': 'completed200', 'last_complete_epoch': 200, 'completed_updates': 24400,
              'best_epoch': stop['best_epoch'], 'best_validation': {k:v for k,v in selected.items() if k != 'per_case'},
              'paired_current_multiscale166_to_candidate': comparison, 'paired_fixed_pure193_native_to_candidate': vs_pure,
              'promotion_gate_passed': passed, 'target_scan': .7630921331986233, 'point_floor': 94,
              'checkpoint_path': str((OUT/'best.pth').relative_to(ROOT)), 'checkpoint_sha256': sha(OUT/'best.pth'),
              'selected_validation_report_sha256': sha(OUT/f"validation_e{stop['best_epoch']:03d}.json"),
              'source_pretrain_sha256': frozen['pretrain_sha256'], 'data_audit_sha256': sha(R/'DATA_AUDIT.json'),
              'gpu_smoke_sha256': sha(R/'GPU_SMOKE.json'), 'reload_check_sha256': sha(R/'RELOAD_CHECK.json'),
              'test_tuned_development': True, 'new_test_model_forward': False, 'new_outer_confirmation_access': False,
              'worker':1127648, 'research_change':'valid-region main/auxiliary loss support only',
              'no_score_based_epoch50_budget_stop':True, 'auxiliary_heads_discarded_for_deployment':True}
    write(R/'FINAL_RESULT.json',result); write(R/'RESULT.json',result)
    c = comparison; ci = c['scan_delta_ci95']; subject=selected['locator_subject_dsc_mean']; hits=round(selected['automatic_point_hit_rate']*108)
    support='达到事前明确验证收益门；仍需按复用验证/test-informed开发解释' if passed else '未达到事前明确验证收益门；只评价当前固定配方，不证明所有有效区域策略无效'
    (R/'RESULT.md').write_text(f'''# SegResNet有效区域监督结果

单组完成200轮/24400更新，原972扫描/444患者训练，以固定108扫描/50患者原生1mm裁剪栅格选优。选第{stop['best_epoch']}轮：Scan **{selected['locator_dsc_mean']:.9f}**、Subject **{subject:.9f}**、Point **{hits}/108**。

对现有多尺度166：Scan差{c['scan_delta']:+.9f}，50患者配对95%CI{ci}；Subject差{c['subject_delta']:+.9f}，CI{c['subject_delta_ci95']}；Point差{c['point_hits_delta']:+d}。108中{c['improved_scans']}例改善、{c['worse_scans']}例变差、{c['tied_scans']}例相同。对固定纯193的同栅格参考差{vs_pure['scan_delta']:+.9f}，CI{vs_pure['scan_delta_ci95']}；历史参考仅复用报告，未增加模型前向。

{support}。冻结门为Scan至少0.763092133、配对95%区间下界正、Subject至少0.739668718、Point至少94/108。没有第50轮分数早停，也没有按中途分数改辅助权重、BCE权重、seed、阈值或预算。

唯一研究变化是图像几何有效区域损失。单T1ce、公开334的已有150轮预训练初始化、原972/108/270、主网络与半/四分之一辅助头、global8/每轮122更新及原学习率均保留。掩膜与空间增强/空前景回退同步，框内真实背景仍参与监督；主/粗尺度损失按有效覆盖归约。图像和标签原生载荷SHA与旧收据一致。bf16合成门的一次数值归约修复在患者访问前完成，原失败源码/收据保留，数值门未放宽。

两卡40GB；实际峰值和墙钟见GPU_SMOKE与workspace TRAIN日志，缓存约26.8GiB只放worker/tmp；主模型约18MiB、部署仍一次整脑前向。训练与108验证已完成且best/latest严格重载通过。此前270信息影响方向设计，标test-informed开发；**本候选没有新270/outer/confirmation像素或模型访问**，不能把这里的108成绩当270成绩。原270榜首融合0.775891与八块单模型0.772830均为历史同格test-tuned点估计。

[七问](IDEA.md) · [机器结果](FINAL_RESULT.json) · [GPU合成](GPU_SMOKE.json) · [数值工程修订](NUMERIC_ENGINEERING_AMENDMENT_001.json) · [原生载荷核验](DATA_AUDIT.json) · [执行冻结](EXECUTION_FREEZE.json) · [重载核验](RELOAD_CHECK.json) · [最佳主模型](../../../{result['checkpoint_path']})。本组结束，不自动续训、回放公开病例、扩组、换epoch或进入270。
''')
    details=f"选{stop['best_epoch']}轮108 Scan{selected['locator_dsc_mean']:.6f}、Subject{subject:.6f}、Point{hits}/108；对旧多尺度166差{c['scan_delta']:+.6f}、95%CI[{ci[0]:+.6f},{ci[1]:+.6f}]。{support}。"
    publish_progress('完成200轮',details)
    path=ROOT/'AFLOW_BRATS_METHOD_CATALOG.md';value=path.read_text();a=value.index('## SegResNet有效区域监督｜');b=value.find('\n## ',a+3);b=len(value) if b<0 else b
    card=value[a:b];card=re.sub(r'｜[^\n]+','｜完成200轮',card,count=1)
    card=re.sub(r'\*\*结果：\*\*[^\n]+','**结果：** '+details+' 本组只有训练/108验证，无新270访问。',card)
    card=re.sub(r'\*\*是否支持直觉：\*\*[^\n]+','**是否支持直觉：** '+support+'，不能把几何补零比例当成性能归因。',card)
    card=re.sub(r'\*\*代价/下一步：\*\*[^\n]+',f'**代价/下一步：** 双A10040GB训练，约26.8GiB临时缓存；部署主模型一次前向。本组停止，不自动扩组、改配方或测试270。[最终报告]({str(R.relative_to(ROOT))}/RESULT.md)。',card)
    path.write_text(value[:a]+card+value[b:])
    agents=ROOT/'AGENTS.md';value=agents.read_text();heading='## 2026-10-10 有效区域监督单组：用户最新授权\n'
    if heading not in value:raise RuntimeError('AGENTS active scope missing')
    value=value.replace(heading,heading+'\n- **本轮终态：** '+details+' 已完成200轮/24400更新及严格重载，仅108验证、无新270/outer/confirmation。本组不自动恢复/扩组/测试。\n',1);agents.write_text(value)
    checks=[]
    for script in ['verify_aflow_brats_method_catalog.py','verify_aflow_brats_leaderboard.py','verify_aflow_brats_doc_layout.py','verify_aflow_root_artifacts.py']:
        p=subprocess.run([sys.executable,'scripts/'+script],cwd=ROOT,capture_output=True,text=True)
        checks.append({'script':script,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr})
    write(R/'FINAL_DOC_CHECKS.json',checks)
    if any(x['returncode'] for x in checks):raise RuntimeError('final publication checks failed')
    write(R/'COMPLETION_AUDIT.json',{'updated_at':now(),'status':'passed','completed_epochs':200,'completed_updates':24400,
          'selected_epoch':stop['best_epoch'],'checkpoint_sha256':sha(OUT/'best.pth'),'result_sha256':sha(R/'FINAL_RESULT.json'),
          'source_hashes_exact':True,'all_four_document_checks_passed':True,'no_new_test_or_outer_confirmation_access':True})


if __name__=='__main__':main()
