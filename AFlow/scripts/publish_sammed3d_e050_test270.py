"""Publish this one named test cohort without touching the active training recipe."""
import sys,json,re,subprocess,os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.sammed3d_graymean_single972 import ROOT,sha,write,now
R=ROOT/'archive/research_records/2026-09-17_sammed3d_graymean_e050_test270'
REL=str(R.relative_to(ROOT));KEY='test270_sammed3d_gtcentered_best5_rank_by_scan'


def main():
    d=json.loads((R/'RESULT.json').read_text());sel=json.loads((R/'SELECTION_RECEIPT.json').read_text());raw=json.loads((R/'RAW_TEST270.json').read_text())
    assert d['status']=='complete_270_case_test' and raw['status']=='complete_270_case_test'
    c=d['best5']['candidate'];b=d['best5']['baseline'];delta=d['best5']['paired_delta'];f=d['final5']['candidate']
    outcome='超过旧基线点估计' if d['point_estimate_exceeds_baseline'] else '未超过旧基线'
    brief=f"第50轮灰度模型在复用内部270扫描/119患者上的五点击最佳Scan Dice {c['scan']:.6f}、Subject {c['subject']:.6f}；旧基线Scan {b['scan']:.6f}，差{delta['scan']:+.6f}，患者整簇配对95%区间[{delta['scan_ci95'][0]:+.6f},{delta['scan_ci95'][1]:+.6f}]，{outcome}。"
    text=f"# 第50轮SAM-Med3D灰度模型：270内部测试结果\n\n{brief}\n\n| 指标 | 灰度第50轮 | 旧基础模型 | 配对差值95%区间 |\n|---|---:|---:|---|\n| 五点击最佳Scan | {c['scan']:.9f} | {b['scan']:.9f} | {delta['scan_ci95']} |\n| 五点击最佳Subject | {c['subject']:.9f} | {b['subject']:.9f} | {delta['subject_ci95']} |\n| 第五点击Scan | {f['scan']:.9f} | {d['final5']['baseline']['scan']:.9f} | {d['final5']['paired_delta']['scan_ci95']} |\n\n候选五点击最佳Scan自身患者整簇95%区间{c['scan_ci95']}，Subject区间{c['subject_ci95']}。10000次患者成对整簇bootstrap、seed20260917。逐扫描改善/变差/持平={d['best5']['scan_improved']}/{d['best5']['scan_worse']}/{d['best5']['scan_tied']}。CI仅适用这份复用内部队列及已冻结候选，不代表外部泛化。\n\n**直觉和解释：** 用区域平均亮度帮助肿瘤轮廓学习；此次结果{outcome}。它检验了这个具体第50轮模型，不能由单组历史比较证明灰度项的独立因果效应，也不能否定所有灰度约束。本轮更大batch、seed及选优指标与旧基线不同；模型不等同于论文作者完整实现。\n\n**可比与访问边界：** 固定972/108/270，候选在测试之前由108验证第五点击选为第50轮且由用户指定；整批测试图像/标签张量SHA与旧基线缓存完全一致。复用旧五点击函数和128³/1.5mm栅格、raw logit>0、无增强/后处理。真实标签辅助裁剪和模拟点击，属于交互模型，禁止混入全自动榜。Point hit不适用，记null。旧基线按best5验证选优，本轮按final5选优；整推理文件哈希不同已预先披露，不冒称完整训练配方一致。\n\n用户授权本次测试，非新盲测或外部确认；无新增outer/confirmation。独立评估进程已访问test，训练执行器自身未读test，不能用其test_access:false否认本次访问。200轮训练继续，测试不改变损失/学习率/batch/阈值/选优，不自动测试别的epoch。\n\n**成本：** 并行使用现有一张A100余量，batch1，推理峰值已分配{d['runtime']['peak_allocated_bytes']/2**30:.3f}GiB；预处理{d['runtime']['preprocess_seconds']:.1f}秒、推理{d['runtime']['inference_seconds']:.1f}秒，共{d['runtime']['total_seconds']:.1f}秒。独立checkpoint {d['checkpoint_bytes']/2**20:.1f}MiB，worker临时缓存{d['cache_bytes']/2**30:.3f}GiB；每例一次编码+五次解码，无灰度额外前向。共享GPU会影响训练壁钟但不改训练参数。\n\n**下一步：** 保持已冻结200轮任务及验证选优；本次负/正结果均存档，不以test启动调参。\n\n[机器报告](RESULT.json) · [原始270例](RAW_TEST270.json) · [访问前冻结](FREEZE.json) · [访问收据](PREACCESS.json) · [缓存索引](CACHE_INDEX.json) · [独立模型](checkpoint_epoch050.pth) · [七问](IDEA.md)。\n"
    (R/'RESULT.md').write_text(text)
    leaderboard=ROOT/'AFLOW_BRATS_LEADERBOARD.json';j=json.loads(leaderboard.read_text())
    common={'cohort':{'scans':270,'subjects':119,'role':'reused_internal_test'},'point_hit':None,'point_hit_reason':'not applicable: GT-centered interactive clicks, not automatic locator','resolution':'128^3 after1.5mm resampling','threshold_raw_logit':0,'tta':False,'postprocessing':False,'forward_cost':'1 image encoder + 5 interactive decoder calls per scan','reused':True,'test_tuned':False,'external':False,'evidence_status':'user_authorized_reused_internal_test; no_test_tuning; GT_centered_interactive; grid matched via tensor SHA; historical recipe differs'}
    baseline_report=json.loads((ROOT/sel['baseline_report']).read_text())
    rows=[{**common,'method':'upstream_literal_best_of_five','scan_dice':b['scan'],'subject_dice':b['subject'],'scan_dice_ci95':b['scan_ci95'],'subject_dice_ci95':b['subject_ci95'],'checkpoint_epoch':21,'checkpoint_sha256':baseline_report['model']['checkpoint_sha256'],'report_path':sel['baseline_report'],'report_sha256':sel['baseline_report_sha256'],'selection':'historical validation best5 epoch21 after200epochs'},
          {**common,'method':'sammed3d_graymean_single972_epoch050','scan_dice':c['scan'],'subject_dice':c['subject'],'scan_dice_ci95':c['scan_ci95'],'subject_dice_ci95':c['subject_ci95'],'checkpoint_epoch':50,'checkpoint_path':REL+'/checkpoint_epoch050.pth','checkpoint_sha256':d['checkpoint_sha256'],'report_path':REL+'/RAW_TEST270.json','report_sha256':sha(R/'RAW_TEST270.json'),'selection':'fixed validation108 final5 epoch50; user fixed interim candidate; training continues to200','paired_delta_vs_vanilla':delta}]
    rows.sort(key=lambda x:-x['scan_dice'])
    for rank,row in enumerate(rows,1):row['rank']=rank
    j[KEY]=rows;j['updated_utc']=now();j['latest_graymean_e050_test270']={'result_path':REL+'/RESULT.json','result_sha256':sha(R/'RESULT.json'),'epoch_fixed_before_test':50,'new_test_access':True,'ongoing_training_unchanged':True,'historical_baseline_row_precision_note':'noncomparable_test_history rounded values preserved; new matched-protocol table uses original raw full precision'}
    write(leaderboard,j)
    p=ROOT/'AFLOW_BRATS_LEADERBOARD.md';s=p.read_text();s=re.sub(r'更新时间：[^\n]+','更新时间：2026-09-17 UTC',s,count=1)
    block='<!-- SAM_GRAY_E050_TEST270 -->\n## 内部test270：SAM-Med3D标签中心裁剪、五点击最佳\n\n同270扫描/119患者，独立于全自动主榜；只有两行是这里的同推理口径比较，保留旧历史表。全部复用内部测试，非外部/新盲测；无测试调参。\n\n| 排名 | 模型 | Scan Dice（95% CI） | Subject Dice（95% CI） | Point hit | 前向成本 |\n|---:|---|---:|---:|---|---|\n'
    for row in rows:
        name='灰度模型，第50轮' if row['checkpoint_epoch']==50 else '旧基础模型，第21轮'
        block+=f"| {row['rank']} | {name} | {row['scan_dice']:.9f} [{row['scan_dice_ci95'][0]:.6f}, {row['scan_dice_ci95'][1]:.6f}] | {row['subject_dice']:.9f} [{row['subject_dice_ci95'][0]:.6f}, {row['subject_dice_ci95'][1]:.6f}] | 不适用（交互） | 1编码+5解码 |\n"
    block+='\n'+brief+' 固定128³/1.5mm，阈值0，无增强/后处理；重建测试张量哈希与旧基线完全一致。训练batch/seed/选优规则不完全相同，非灰度因果消融。模型/原始报告哈希见JSON。[完整证据]('+REL+'/RESULT.md)。\n<!-- END_SAM_GRAY_E050_TEST270 -->\n\n'
    if '<!-- SAM_GRAY_E050_TEST270 -->' in s:s=re.sub(r'<!-- SAM_GRAY_E050_TEST270 -->.*?<!-- END_SAM_GRAY_E050_TEST270 -->\n*',lambda m:block,s,flags=re.S)
    else:s=s.replace('## 当前决策',block+'## 当前决策',1)
    p.write_text(s)
    # This is a separate test card, outside the training publisher's replaced section.
    p=ROOT/'AFLOW_BRATS_METHOD_CATALOG.md';s=p.read_text();start=s.index('## SAM-Med3D灰度第50轮：用户指定test270｜');end=s.find('\n## ',start+3)
    card='## SAM-Med3D灰度第50轮：用户指定test270｜已完成，'+outcome+'\n\n**一句话：** 用同一套测试题检验灰度模型。 **出发点：** 验证与测试分数不能直接比较。\n**具体做法：** 独立冻结第50轮，固定270/119、原预处理/五点击/阈值，无增强或后处理；200轮训练不变。\n**结果：** '+brief+'\n**是否支持直觉：** '+outcome+'；单组历史比较不能证明灰度因果效应，也不代表其他灰度方法不可能有效。\n**代价/下一步：** 一卡batch1并行推理，一次编码+五次解码；维持200轮训练与验证选优，不按test调参。[核对材料]('+REL+'/RESULT.md)\n'
    p.write_text(s[:start]+card+(s[end:] if end>=0 else ''))
    p=ROOT/'AFLOW_BRATS_HANDOFF_CURRENT.md';s=p.read_text();pattern=r'\*\*2026-09-17独立测试授权：\*\*[^\n]*'
    note='**2026-09-17独立测试已完成：** '+brief+' [完整报告]('+REL+'/RESULT.md)。训练照原方案继续；下方训练监控及训练归档的“未新增测试/test_access:false”仅指训练执行器，实验整体已访问test。本次测试不得反向调参，也不自动授权其他epoch测试。'
    s=re.sub(pattern,lambda m:note,s);p.write_text(s)
    p=ROOT/'AFLOW_BRATS_RESULTS.md';s=p.read_text();entry='\n## 2026-09-17｜用户指定灰度第50轮test270评估\n\n**问题：** 想确认是否超过0.594226，但此前只有验证分数。 **做法：** 用户明确授权，冻结第50轮并行测270例，原输入/推理/阈值固定。 **结果：** '+brief+' **决定：** 结果归档并更新交互榜；训练不调参继续至200轮，无新增outer/confirmation。[核对材料]('+REL+'/RESULT.md)\n'
    if '## 2026-09-17｜用户指定灰度第50轮test270评估' not in s:p.write_text(s+entry)
    p=ROOT/'AFLOW_BRATS_WEEKLY_REPORT.md';s=p.read_text();line='\n2026-09-17第50轮独立测试：'+brief+' 保持200轮冻结训练，不用test调参；属于交互模型及复用内部证据。[核对材料]('+REL+'/RESULT.md)。\n'
    if '2026-09-17第50轮独立测试：' not in s:p.write_text(s+line)
    print(brief)

if __name__=='__main__':main()
