"""Archive and rank the one fixed pure epoch193 candidate without losing old rows."""
import json,re,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.segresnet_plain972_common import ROOT,sha,write,now
R=ROOT/'archive/research_records/2026-09-17_segresnet_plain972_e193_test270';REL=str(R.relative_to(ROOT));METHOD='pure_segresnet_full972_epoch193_20260917'


def main():
    d=json.loads((R/'RESULT.json').read_text());raw=json.loads((R/'RAW_TEST270.json').read_text());assert d['status']=='complete_test270' and len(raw['summary']['per_case'])==270
    comparisons=d['comparisons'];delta=comparisons['flip8_kd168'];plain=comparisons['plain168']
    (R/'RESULT.md').write_text(f'''# 原972纯SegResNet第193轮：270内部测试完成

**Scan Dice {d['scan_dice']:.9f}、Subject Dice {d['subject_dice']:.9f}，自动点命中{d['point_hits']}/270（{d['point_hit']:.6f}）。** 全部270扫描/119患者完成；候选为训练200轮后固定108验证选出的第193轮，并由用户明确授权测试。

| 指标 | 结果 | 患者整簇95%区间 |
|---|---:|---|
| Scan Dice | {d['scan_dice']:.9f} | [{d['scan_ci95'][0]:.6f}, {d['scan_ci95'][1]:.6f}] |
| Subject Dice | {d['subject_dice']:.9f} | [{d['subject_ci95'][0]:.6f}, {d['subject_ci95'][1]:.6f}] |
| 自动点命中 | {d['point_hits']}/270 | 诊断指标，非网络输入 |

## 固定历史比较

| 历史模型 | Scan Dice | 当前减历史 | 配对Scan95%区间 |
|---|---:|---:|---|
| 756纯模型epoch99 | {comparisons['plain99']['baseline_scan_dice']:.9f} | {comparisons['plain99']['scan_delta']:+.6f} | {comparisons['plain99']['scan_ci95']} |
| 756纯模型epoch168 | {plain['baseline_scan_dice']:.9f} | {plain['scan_delta']:+.6f} | {plain['scan_ci95']} |
| Full972蒸馏epoch168，1次前向 | {comparisons['full972_kd168']['baseline_scan_dice']:.9f} | {comparisons['full972_kd168']['scan_delta']:+.6f} | {comparisons['full972_kd168']['scan_ci95']} |
| fit864优化器支线，1次前向 | {comparisons['gcsam184']['baseline_scan_dice']:.9f} | {comparisons['gcsam184']['scan_delta']:+.6f} | {comparisons['gcsam184']['scan_ci95']} |
| Full972蒸馏＋8镜像 | {delta['baseline_scan_dice']:.9f} | {delta['scan_delta']:+.6f} | {delta['scan_ci95']} |

全部为测试前锁定的历史原始报告配对，未重新运行历史模型。10000次患者整簇成对bootstrap，seed20260917；描述性多重比较未作多重性校正，不作独立临床或外部显著性声明。

本模型在当前复用内部test270自动主榜取得总体与单次前向Scan/Subject最高点估计。相对旧8镜像Scan仅高0.001909、Subject高0.003731，两差值区间均跨0，不能称显著胜出。Point242低于旧8镜像265，也低于旧Full972蒸馏262；不称所有指标最佳。相对旧纯168，Scan差区间为正，但训练池756→972、抽样、选择轮次等不相同，不把提升拆成单一因素。

## 想法、固定项与解释

本组目标是补齐固定972/108/270上的纯SegResNet自动基线。470万参数、随机初始化、Dice+0.1交叉熵、完整200轮，每轮972唯一扫描/122更新；无蒸馏、灰度项、中心头、参数平均、融合、镜像增强或后处理。结果支持这条直接自动分割路线作为当前场景的实用基线，不证明网络架构普遍优于SAM，也不证明去掉蒸馏有独立因果收益。

测试输入仍为单T1ce/WT1,2,3排除4，N4校正、图像自身脑区裁剪到128³、非零体素标准化；整个cache图像/标签SHA已重验，病例/患者与原972训练和108验证隔离。bfloat16 batch2、固定raw logit>0，每扫描仅1次前向，无GT中心/点击。与SAM交互模型栅格和权限不同，不混排为纯架构消融。

## 访问与成本

本次为用户明确授权的一次193候选test270评估。训练/选优不使用本次测试；此前研究已看过该测试，故是reused internal、prior-test-informed development，不是pristine blind或外部确认。本次没有新增outer/confirmation评估；不得因本结果调整阈值、loss、epoch、分辨率、TTA或后处理，也不自动重测。

已有缓存原位复用，无新影像缓存；独立权重约18MiB。GPU纯推理{d['runtime']['inference_seconds']:.2f}秒（缓存已备），含hash/加载约{d['runtime']['total_seconds']:.2f}秒；峰值分配{d['runtime']['peak_allocated_bytes']/2**30:.3f}GiB。该耗时不含历史N4缓存构建，不能冒充原始MRI到输出端到端耗时。对比旧8镜像仅需1/8前向次数；准确壁钟比未在本次同机配对测量。

## 下一步

固定本模型为本项目当前纯自动基线并保留证据。模型选型如追求无需点击的自动分割，可优先考虑该路线；若研究交互纠错，应另按同使用权限设计评估。当前不自动训练、调参或增加测试访问。

[机器报告及配对区间](RESULT.json) · [原始270例](RAW_TEST270.json) · [访问前冻结](FREEZE.json) · [选优收据](SELECTION_RECEIPT.json) · [访问收据](PREACCESS.json) · [第193轮模型](checkpoint_epoch193.pth)。
''')
    p=ROOT/'AFLOW_BRATS_LEADERBOARD.json';j=json.loads(p.read_text());oldrows=list(j['test270_direct_locator_rank_by_scan'])
    mdpath=ROOT/'AFLOW_BRATS_LEADERBOARD.md';md=mdpath.read_text();start=md.index('## Reused internal test270');tail=md.index('关键增益：',start)
    section=md[start:tail];mdrows=[line for line in section.splitlines() if re.match(r'^\|\s*\d+\s*\|',line)]
    if METHOD not in [x['method'] for x in oldrows]:assert len(mdrows)==len(oldrows)
    rendered={x['method']:line for x,line in zip(oldrows,mdrows)}
    row={'method':METHOD,'scan_dice':d['scan_dice'],'subject_dice':d['subject_dice'],'point_hit':d['point_hit'],'point_hits':d['point_hits'],'point_denominator':270,'locator_forwards':1,'cohort':{'scans':270,'subjects':119},'training_scope':{'train_scans':972,'train_subjects':444,'validation_scans':108,'validation_subjects':50,'completed_epochs':200,'unique_scans_per_epoch':972,'optimizer_updates':24400},'checkpoint_epoch':193,'checkpoint_path':REL+'/checkpoint_epoch193.pth','checkpoint_sha256':d['checkpoint_sha256'],'resolution':128,'threshold':0.0,'tta':False,'sam':False,'postprocessing':False,'distillation':False,'scan_dice_95ci':d['scan_ci95'],'subject_dice_95ci':d['subject_ci95'],'bootstrap':'10000 patient cluster percentile95 seed20260917','evidence_status':'user_authorized_validation_selected_epoch193; reused_internal_test; prior_test_informed_development; no_current_test_selection_or_retuning; not_pristine_blind; not_external','report_path':REL+'/RAW_TEST270.json','report_sha256':sha(R/'RAW_TEST270.json'),'status':'highest_reused_internal_test_scan_and_subject_point_estimates_overall_and_1x; not_highest_point_hit; paired_difference_vs_previous_flip8_CI_crosses_zero','comparability_note':'image-derived N4 braincrop128; same test cache as prior true128 rows; historical training/sample recipes differ; legacy160 rasterization caveat retained','paired_comparisons':d['comparisons']}
    rows=[x for x in oldrows if x['method']!=METHOD]+[row];rows.sort(key=lambda x:-x['scan_dice'])
    for i,x in enumerate(rows,1):
        x['rank']=i
        if x['method']=='full972_random_init_voxel_kd_epoch168_flip8_tta':x['status']='historical former highest scan/subject; retains highest overall point_hit265/270;8x_inference'
        elif x['method']=='gcsam_inner_fit864_epoch184':x['status']='historical former highest1x scan; superseded by pure_full972_epoch193; fit864 not full972'
        elif x['method']=='full972_random_init_voxel_kd_epoch168':x['status']='historical former highest1x subject; retains highest1x point_hit262/270; failed_gate_exploratory'
    j['test270_direct_locator_rank_by_scan']=rows;j['updated_utc']=now();j['latest_plain972_e193_test270']={'result_path':REL+'/RESULT.json','result_sha256':sha(R/'RESULT.json'),'all270_complete':True,'reused_internal':True,'prior_test_informed_development':True,'current_test_used_for_selection':False,'rank_scope':'automatic direct locators; overall and1x Scan/Subject point-estimate best; Point not best'};write(p,j)
    rendered[METHOD]=f"| 0 | 纯SegResNet，原972训练，epoch193 | `{d['scan_dice']:.9f}` | `{d['subject_dice']:.9f}` | {d['point_hits']}/270 (`{d['point_hit']:.6f}`) | `1×` | 用户指定验证选优；复用内部；无蒸馏/TTA；Scan/Subject点估计最高，Point非最高 |"
    table='## Reused internal test270：全自动direct locator\n\n同一270 scans /119 subjects，按Scan Dice降序。复用内部测试，非pristine blind、外部确认或公开SOTA；legacy160³栅格限制仍保留。\n\n| 排名 | 方法 | Scan Dice | Subject Dice | Point hit | 成本 | 状态 |\n|---:|---|---:|---:|---:|---:|---|\n'
    for x in rows:table+=re.sub(r'^\|\s*\d+\s*\|','| '+str(x['rank'])+' |',rendered[x['method']])+'\n'
    table+=f"\n2026-09-17当前同test270/119自动系统总体及单次前向Scan/Subject最高点估计：纯SegResNet原972 epoch193，**{d['scan_dice']:.6f}/{d['subject_dice']:.6f}**，仅1次前向。Scan95%CI [{d['scan_ci95'][0]:.6f},{d['scan_ci95'][1]:.6f}]；Point242/270非最高。相对旧Flip8 Scan+0.001909的患者配对95%CI [-0.004567,+0.009086]跨0，不称显著胜出。最高Point仍为旧Flip8的265/270，最高1×Point仍为旧Full972 KD的262/270。[本次完整结果]({REL}/RESULT.md)。\n\n以下旧关键增益保留历史时点，不作为当前最高声明；新结果不触发重测/调参。\n\n"
    md=md[:start]+table+md[tail:];md=re.sub(r'更新时间：[^\n]+','更新时间：2026-09-17 UTC',md,count=1)
    md=re.sub(r'- 最高内部test总体点估计：[^\n]+',f'- 最高内部test自动总体及1× Scan/Subject点估计：原972纯SegResNet epoch193，{d["scan_dice"]:.6f}/{d["subject_dice"]:.6f}；与旧8×最高差值CI跨0，Point非最高，复用内部而非盲测/外部确认。',md)
    md=re.sub(r'- 最高内部test 1× Scan点估计仍为[^\n]+','- 最高内部test 1× Scan/Subject为原972纯SegResNet193；1× Point最高仍为Full972 KD168的262/270，总体Point最高仍为其Flip8的265/270。',md)
    mdpath.write_text(md)
    train_r=ROOT/'archive/research_records/2026-09-17_segresnet_plain972_baseline'
    write(train_r/'TEST_ACCESS_ADDENDUM.json',{'utc':now(),'test_access':True,'test_checkpoint_epoch':193,'result_path':REL+'/RESULT.json','result_sha256':sha(R/'RESULT.json'),'preaccess_sha256':sha(R/'PREACCESS.json'),'scope':'training FINAL_RESULT.json remains frozen historical validation-only receipt; experiment-wide access now includes this test','no_new_outer_confirmation_access':True})
    current=json.loads((train_r/'RESULT.json').read_text());current.update(test_evaluation_completed=True,experiment_test_access=True,test_result_path=REL+'/RESULT.json',test_scan_dice=d['scan_dice'],test_subject_dice=d['subject_dice'],inherited_access_flag_scope='new_test_pixels_access in training FINAL_RESULT describes training/validation stage only');write(train_r/'RESULT.json',current)
    p=train_r/'RESULT.md';s=p.read_text();s+='\n**后续用户授权测试已完成：** 第193轮test270/119 Scan '+f"{d['scan_dice']:.6f}"+'、Subject '+f"{d['subject_dice']:.6f}"+'，Point242/270；上文“未测试”是训练完成时的历史状态，[最新测试结果](../2026-09-17_segresnet_plain972_e193_test270/RESULT.md)与访问补记优先。\n';p.write_text(s)
    p=ROOT/'AFLOW_BRATS_METHOD_CATALOG.md';s=p.read_text();a=s.index('## SegResNet固定972纯基线｜');b=s.find('\n## ',a+3)
    card='## SegResNet固定972纯基线｜训练200轮、193轮test270完成\n\n**一句话：** 小型三维残差网络直接给出自动肿瘤掩膜。 **出发点：** 旧纯模型756训练，缺当前972划分纯baseline。\n**具体做法：** 原972/108/270、随机初始化纯网络、200轮且每轮972各一次；验证选193轮，用户授权后固定一次test，1×无蒸馏/灰度/融合/增强后处理。\n**结果：** 复用内部270/119 Scan '+f"{d['scan_dice']:.6f}"+'、Subject '+f"{d['subject_dice']:.6f}"+'，Point242/270；自动总体与1× Dice最高点估计，较旧8×仅+0.001909且CI跨0。\n**是否支持直觉：** 支持纯自动基线的实用性；非受控架构/采样消融，不能单独归因模块或声称外部泛化。\n**代价/下一步：** 训练双卡约1小时43分，部署1次前向；当前停止，不按test调参或重测。[核对材料]('+REL+'/RESULT.md)\n'
    p.write_text(s[:a]+card+(s[b:] if b>=0 else ''))
    p=ROOT/'AFLOW_BRATS_RESULTS.md';s=p.read_text();s+='\n## 2026-09-17｜原972纯SegResNet193一次test270完成\n\n**问题：** 用户指定验证最佳193轮测试。 **做法：** 访问前固定模型/缓存/阈值/原五份历史对照，完整270例1次前向。 **结果：** Scan '+f"{d['scan_dice']:.6f}"+'、Subject '+f"{d['subject_dice']:.6f}"+'、Point242/270；自动总体及1× Dice最高点估计，旧8镜像差值CI跨0，Point下降。 **决定：** 纯自动基线证据已建立；复用内部、非盲测/外部，无新增outer/confirmation，不继续test调参。[材料]('+REL+'/RESULT.md)。\n';p.write_text(s)
    p=ROOT/'AFLOW_BRATS_WEEKLY_REPORT.md';s=p.read_text();lines=s.splitlines();lines.insert(2,'**2026-09-17最新测试结论：** 原972纯SegResNet第193轮test270/119 Scan0.759179、Subject0.744985，1次前向取得当前自动总体及单次前向Dice最高点估计。较旧8镜像仅+0.001909且差值CI跨0，Point242低于265，不称全面/显著胜出。复用内部，非外部盲测。[完整结果]('+REL+'/RESULT.md)。');p.write_text('\n'.join(lines)+'\n')
    print(json.dumps({'new_rank':next(x['rank'] for x in rows if x['method']==METHOD),'rows_preserved':len(oldrows),'new_total_rows':len(rows),'scan_dice':d['scan_dice']}))
if __name__=='__main__':main()
