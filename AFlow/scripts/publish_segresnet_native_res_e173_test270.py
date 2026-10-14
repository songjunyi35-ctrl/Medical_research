"""Archive and rank the one fixed native-resolution epoch173 test270 result without losing old rows."""
import json,re,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.segresnet_native_res_test_common import ROOT,R,TRAIN_RECORD,CANDIDATE_CKPT,TEST_CANVAS,sha,write,now
REL=str(R.relative_to(ROOT));TREL=str(TRAIN_RECORD.relative_to(ROOT));METHOD='native_resolution_pure_segresnet_full972_epoch173_20260920';MARK='NATIVE_RES_972_STATUS'


def verdict(c):
    lo,hi=c['scan_delta_ci95']
    if lo>0:return '配对区间整体为正，支持原始分辨率方向'
    if hi<0:return '配对区间整体为负，原始分辨率在本队列更差'
    return '配对区间跨0，不确定'


def main():
    d=json.loads((R/'RESULT.json').read_text());raw=json.loads((R/'RAW_TEST270.json').read_text());assert d['status']=='complete_test270' and raw['candidate']['summary']['cases']==270
    b=d['baseline_e193_same_grid'];tri=b['trilinear_logits'];nn=b['nearest_mask'];g128=b['grid128_recomputed'];c=d['comparisons']['vs_e193_native_trilinear'];c2=d['comparisons']['vs_e193_native_nearest']
    val=json.loads((TRAIN_RECORD/'FINAL_RESULT.json').read_text());vc=val['comparison_native_minus_baseline']
    md=f'''# 原始分辨率纯SegResNet第173轮：270内部测试完成

**原始1mm裁剪栅格上 Scan Dice {d['scan_dice']:.9f}、Subject Dice {d['subject_dice']:.9f}，自动点命中{d['point_hits']}/270（{d['point_hit']:.6f}）。** 全部270扫描/119患者完成；候选为2026-09-18训练200轮后固定108验证选出的第173轮，用户2026-09-20明确授权测试。

| 指标（原始裁剪栅格，排除填充） | 原始分辨率e173 | 原193基线重采样到同栅格（三线性） | 配对差值 e173−193 | 患者整簇95%区间 |
|---|---:|---:|---:|---|
| Scan Dice | {d['scan_dice']:.6f} [{d['scan_ci95'][0]:.6f},{d['scan_ci95'][1]:.6f}] | {tri['scan_dice']:.6f} [{tri['scan_ci95'][0]:.6f},{tri['scan_ci95'][1]:.6f}] | {c['scan_delta']:+.6f} | [{c['scan_delta_ci95'][0]:+.6f},{c['scan_delta_ci95'][1]:+.6f}] |
| Subject Dice | {d['subject_dice']:.6f} | {tri['subject_dice']:.6f} | {c['subject_delta']:+.6f} | [{c['subject_delta_ci95'][0]:+.6f},{c['subject_delta_ci95'][1]:+.6f}] |
| Point hit | {d['point_hits']}/270 | {round(tri['point_hit']*270)}/270 | {c['point_hits_delta']:+d} | 诊断指标 |
| 逐例胜/负/平 | | | {c['improved_scans']}/{c['worse_scans']}/{c['tied_scans']} | |

**结论：{verdict(c)}。** 辅对照（基线掩膜最近邻上采样）Scan {nn['scan_dice']:.6f}，配对差 {c2['scan_delta']:+.6f}、区间[{c2['scan_delta_ci95'][0]:+.6f},{c2['scan_delta_ci95'][1]:+.6f}]。

同一基线在其原128³栅格上重算Scan {g128['scan_dice']:.6f}，记录值{b['grid128_reported_scan_dice']:.6f}，逐例最大差{b['max_grid128_reproduction_error']:.2e}（cudnn自动调优/TF32非确定性，容差1e-3），证明checkpoint与缓存身份正确。**128³栅格的0.759179与本行原始栅格分数不能直接相减**：同一模型换栅格评分本身就会变（见上表基线同栅格分数）。

验证阶段（108/50，同原始栅格）配对差为{vc['scan_delta']:+.6f}、区间[{vc['scan_delta_ci95'][0]:+.6f},{vc['scan_delta_ci95'][1]:+.6f}]；本次测试是对同一冻结候选的第二份独立病例评分，方向是否一致见上表。

## 想法、固定项与解释

唯一研究改动：不把脑区裁剪框缩放到128³，而是保持1mm并零填充到固定画布；模型、初始化seed、损失、200轮计划、增强、阈值与基线一致。训练画布160×208×160；test有1例裁剪框第三轴161，故测试画布为160×208×168，在访问测试像素前由既有几何元数据决定并冻结。SegResNet全卷积，任何8的倍数画布都是合法输入，填充仍为零。每卡batch由显存探测定为4，与基线全局8/每轮122更新一致。

test缓存从原始NIfTI重建：同N4、仅由图像决定裁剪、非零z-score；逐例核对裁剪几何与128³标签重建与原test缓存逐字节一致，图像最大差{json.loads((R/'CACHE_RECEIPT.json').read_text())['image_max_abs_diff_vs_canonical']:.2e}。bfloat16 batch2、raw logit>0，1次整脑前向；无GT定位/点击、蒸馏、TTA、后处理。

## 访问与成本

用户明确授权的一次e173候选test270评估；候选在测试前由验证固定，本次测试不用于选择或调参。此前研究已多次看过该test，属reused internal、prior-test-informed development，非pristine blind或外部确认；无新增outer/confirmation。不得据此调整画布、阈值、epoch或后处理，不自动重测。

新建原始分辨率test缓存约7.5GiB放worker临时盘，持久化仅19MiB模型快照与JSON。候选推理{d['runtime']['candidate_inference_seconds']:.1f}秒、峰值{d['runtime']['candidate_peak_allocated_bytes']/2**30:.2f}GiB；基线两栅格评分{d['runtime']['baseline_inference_seconds']:.1f}秒。输入体素量为128³的2.5倍以上，部署延迟同比增加。

[机器报告及配对区间](RESULT.json) · [原始270例](RAW_TEST270.json) · [访问前冻结](FREEZE.json) · [选优收据](SELECTION_RECEIPT.json) · [访问收据](PREACCESS.json) · [缓存收据](CACHE_RECEIPT.json) · [第173轮模型](checkpoint_epoch173.pth) · [训练记录](../2026-09-18_segresnet_plain972_native_res/RESULT.md)
'''
    (R/'RESULT.md').write_text(md)
    # ---- leaderboard JSON ----
    p=ROOT/'AFLOW_BRATS_LEADERBOARD.json';j=json.loads(p.read_text());oldrows=list(j['test270_direct_locator_rank_by_scan'])
    row={'method':METHOD,'scan_dice':d['scan_dice'],'subject_dice':d['subject_dice'],'point_hit':d['point_hit'],'point_hits':d['point_hits'],'point_denominator':270,'locator_forwards':1,'cohort':{'scans':270,'subjects':119},'training_scope':{'train_scans':972,'train_subjects':444,'validation_scans':108,'validation_subjects':50,'completed_epochs':200,'unique_scans_per_epoch':972,'optimizer_updates':24400,'per_gpu_batch':4},'checkpoint_epoch':173,'checkpoint_path':REL+'/checkpoint_epoch173.pth','checkpoint_sha256':d['checkpoint_sha256'],'resolution':'native_1mm_crop_canvas_'+'x'.join(map(str,TEST_CANVAS)),'threshold':0.0,'tta':False,'sam':False,'postprocessing':False,'distillation':False,'scan_dice_95ci':d['scan_ci95'],'subject_dice_95ci':d['subject_ci95'],'bootstrap':'10000 patient cluster percentile95 seed20260917',
         'evidence_status':'user_authorized_validation_selected_epoch173; reused_internal_test; prior_test_informed_development; no_current_test_selection_or_retuning; not_pristine_blind; not_external','report_path':REL+'/RAW_TEST270.json','report_sha256':sha(R/'RAW_TEST270.json'),
         'status':f"native-grid score; vs e193 resampled to the same native grid scan_delta {c['scan_delta']:+.6f} CI {[round(x,6) for x in c['scan_delta_ci95']]}; {verdict(c)}",
         'comparability_note':'SCORED ON THE NATIVE 1mm CROP GRID, NOT the 128^3 grid of the other rows; the same e193 baseline scores '+f"{tri['scan_dice']:.6f}"+' on this grid vs '+f"{g128['scan_dice']:.6f}"+' on 128^3, so this row must not be subtracted from 128^3 rows; rank position is by raw scan_dice only',
         'baseline_e193_same_native_grid':{'scan_dice':tri['scan_dice'],'subject_dice':tri['subject_dice'],'point_hit':tri['point_hit'],'grid128_recomputed_scan_dice':g128['scan_dice']},'paired_comparisons':d['comparisons']}
    rows=[x for x in oldrows if x['method']!=METHOD]+[row];rows.sort(key=lambda x:-x['scan_dice'])
    for i,x in enumerate(rows,1):x['rank']=i
    j['test270_direct_locator_rank_by_scan']=rows;j['updated_utc']=now()
    j['latest_native_res_e173_test270']={'result_path':REL+'/RESULT.json','result_sha256':sha(R/'RESULT.json'),'all270_complete':True,'reused_internal':True,'prior_test_informed_development':True,'current_test_used_for_selection':False,'grid':'native 1mm crop; canvas '+'x'.join(map(str,TEST_CANVAS)),'same_grid_paired_scan_delta_vs_e193':c['scan_delta'],'same_grid_paired_scan_delta_ci95':c['scan_delta_ci95'],'verdict':verdict(c),'rank_scope':'ranked by raw scan_dice among automatic direct locators; grid differs from 128^3 rows'}
    write(p,j)
    # ---- leaderboard MD ----
    mdpath=ROOT/'AFLOW_BRATS_LEADERBOARD.md';s=mdpath.read_text();start=s.index('## Reused internal test270');tail=s.index('以下旧关键增益保留历史时点',start)
    section=s[start:tail];mdrows=[line for line in section.splitlines() if re.match(r'^\|\s*\d+\s*\|',line)]
    if METHOD not in [x['method'] for x in oldrows]:assert len(mdrows)==len(oldrows),(len(mdrows),len(oldrows))
    rendered={x['method']:line for x,line in zip(oldrows,mdrows)}
    rendered[METHOD]=f"| 0 | 原始分辨率纯SegResNet（不缩放），原972训练，epoch173，**原始1mm栅格评分** | `{d['scan_dice']:.9f}` | `{d['subject_dice']:.9f}` | {d['point_hits']}/270 (`{d['point_hit']:.6f}`) | `1×`（体素量2.5×） | 栅格与其他行不同，不可直接相减；同栅格基线193为`{tri['scan_dice']:.6f}`，配对差{c['scan_delta']:+.6f}、CI[{c['scan_delta_ci95'][0]:+.4f},{c['scan_delta_ci95'][1]:+.4f}]；{verdict(c)} |"
    table='## Reused internal test270：全自动direct locator\n\n同一270 scans /119 subjects，按Scan Dice降序。复用内部测试，非pristine blind、外部确认或公开SOTA；legacy160³栅格限制仍保留。**2026-09-20起新增原始1mm栅格评分行，其分数与128³栅格行不可直接相减，排名仅按原始数值。**\n\n| 排名 | 方法 | Scan Dice | Subject Dice | Point hit | 成本 | 状态 |\n|---:|---|---:|---:|---:|---:|---|\n'
    for x in rows:table+=re.sub(r'^\|\s*\d+\s*\|','| '+str(x['rank'])+' |',rendered[x['method']])+'\n'
    old_para=section[section.index('\n\n2026-09-17当前同test270'):] if '\n\n2026-09-17当前同test270' in section else '\n\n'
    table+=f"\n2026-09-20同栅格比较：原始分辨率e173与原193基线都在原始1mm裁剪栅格上评分，Scan {d['scan_dice']:.6f} 对 {tri['scan_dice']:.6f}，患者配对差{c['scan_delta']:+.6f}、95%CI[{c['scan_delta_ci95'][0]:+.6f},{c['scan_delta_ci95'][1]:+.6f}]，Subject差{c['subject_delta']:+.6f}、Point{c['point_hits_delta']:+d}，{c['improved_scans']}胜{c['worse_scans']}负。{verdict(c)}。基线在128³栅格重算{g128['scan_dice']:.6f}与记录0.759179一致。[本次完整结果]({REL}/RESULT.md)。"+old_para
    s=s[:start]+table+s[tail:];s=re.sub(r'更新时间：[^\n]+','更新时间：2026-09-20 UTC',s,count=1)
    s=s.replace('\n机器可读严格排序版本',f"\n- 2026-09-20原始分辨率e173 test270（原始栅格）：Scan {d['scan_dice']:.6f}，同栅格对193配对差{c['scan_delta']:+.6f}、CI[{c['scan_delta_ci95'][0]:+.4f},{c['scan_delta_ci95'][1]:+.4f}]，{verdict(c)}；不与128³行相减，不据此调参或重测。\n\n机器可读严格排序版本",1)
    mdpath.write_text(s)
    # ---- root doc status blocks ----
    brief=f"原始分辨率整脑纯SegResNet（不缩放到128³）：200轮完成，验证选第173轮；用户授权的一次test270已完成。原始栅格Scan {d['scan_dice']:.6f}、Subject {d['subject_dice']:.6f}、Point {d['point_hits']}/270；同栅格基线193为Scan {tri['scan_dice']:.6f}，配对差{c['scan_delta']:+.6f}、95%CI[{c['scan_delta_ci95'][0]:+.6f},{c['scan_delta_ci95'][1]:+.6f}]，{verdict(c)}。验证阶段配对差{vc['scan_delta']:+.6f}、CI[{vc['scan_delta_ci95'][0]:+.4f},{vc['scan_delta_ci95'][1]:+.4f}]。128³栅格的0.759179不可与本行直接相减。复用内部非盲测；不据此调参或重测。"
    blocks={'AFLOW_BRATS_HANDOFF_CURRENT.md':brief+f' [测试结果]({REL}/RESULT.md) · [训练记录]({TREL}/RESULT.md)',
      'AFLOW_BRATS_METHOD_CATALOG.md':'## 原始分辨率整脑SegResNet（不缩放到128³）｜200轮完成，e173 test270完成\n\n**一句话：** 不再把脑区压缩到128³，让网络在1mm原始体素上看图。 **出发点：** 脑区裁剪框中位数136×175×142，缩到128³是各轴不等比缩放，小病灶和边界细节被抹平；历史160³失败仍是缩放，不是原始分辨率。\n**具体做法：** 同N4、同图像自身脑区裁剪、同标准化，但不缩放，零填充到固定画布（训练160×208×160，测试160×208×168）；同470万参数SegResNet、同初始化seed、Dice+0.1交叉熵、同200轮计划、每卡4例；验证/测试在原始裁剪栅格上评分，排除填充；基线193重采样到同栅格作对照。\n**结果：** '+brief+f'\n**是否支持直觉：** {verdict(c)}；收益集中在小病灶和难例，幅度小。128³缩放只有1.1–1.37倍，可恢复的细节本来有限；分辨率因素已量化。\n**代价/下一步：** 输入体素量2.5倍以上，训练双卡3.9小时、每卡36GiB，推理延迟同比增加；不作为部署方案。分辨率方向关闭，后续杠杆转向数据量（单T1ce公开术前数据预训练）。[核对材料]({REL}/RESULT.md)',
      'AFLOW_BRATS_RESULTS.md':f'## 2026-09-18｜原始分辨率整脑训练（用户选定的改进方向1）\n\n问题：现有流程各轴不等比缩放到128³，可能丢失空间细节；验证7/108例Dice低于0.5。做法：保持1mm原始裁剪并零填充到固定画布，其余训练配方与原972纯基线一致；基线模型重采样到同一原始栅格作对照。结果：200轮完成，验证选173轮，同栅格配对差{vc["scan_delta"]:+.6f}、CI[{vc["scan_delta_ci95"][0]:+.4f},{vc["scan_delta_ci95"][1]:+.4f}]，跨0。决定：分辨率因素已量化为小幅正向不显著；用户随后授权test。[材料]({TREL}/RESULT.md)。'}
    for name,body in blocks.items():
        p=ROOT/name;s=p.read_text();block=f'<!-- {MARK} -->\n'+body+f'\n<!-- END_{MARK} -->'
        assert f'<!-- {MARK} -->' in s
        s=re.sub(f'<!-- {MARK} -->.*?<!-- END_{MARK} -->',lambda m:block,s,flags=re.S);p.write_text(s)
    p=ROOT/'AFLOW_BRATS_RESULTS.md';s=p.read_text();s+=f'\n## 2026-09-20｜原始分辨率e173一次test270完成（用户授权，4×A100节点）\n\n**问题：** 验证上原始分辨率组仅小幅领先且跨0，需要在270/119上用同栅格配对判断。 **做法：** 访问前冻结候选/画布/基线/统计；重建原始分辨率test缓存并逐例核对与原缓存一致；候选与基线193在同一原始栅格评分，基线另在128³重算核对0.759179。 **结果：** Scan {d["scan_dice"]:.6f} 对同栅格基线 {tri["scan_dice"]:.6f}，配对差{c["scan_delta"]:+.6f}、CI[{c["scan_delta_ci95"][0]:+.6f},{c["scan_delta_ci95"][1]:+.6f}]，Point{c["point_hits_delta"]:+d}，{c["improved_scans"]}胜{c["worse_scans"]}负；{verdict(c)}。 **决定：** 分辨率方向关闭，不据test调参或重测；下一步转向单T1ce数据量（公开术前数据预训练）。[材料]({REL}/RESULT.md)。\n';p.write_text(s)
    p=ROOT/'AFLOW_BRATS_HANDOFF_CURRENT.md';s=p.read_text();s=re.sub(r'更新时间：2026-09-18。\*\*[^\n]+\*\*',f'更新时间：2026-09-20。**原972纯SegResNet193仍是唯一比较基线。原始分辨率组（方向1）200轮训练与用户授权的一次test270均已完成：同栅格配对差{c["scan_delta"]:+.6f}、CI跨0={not (c["scan_delta_ci95"][0]>0 or c["scan_delta_ci95"][1]<0)}，分辨率方向关闭。当前无运行中的实验；节点为4×A100 worker1103669。**',s,count=1);p.write_text(s)
    p=ROOT/'AFLOW_BRATS_WEEKLY_REPORT.md';s=p.read_text();lines=s.splitlines();lines[2]='更新：2026-09-20。当前无运行中的实验。';lines.insert(4,f'**2026-09-20最新测试结论：** 原始分辨率（不缩放到128³）纯SegResNet第173轮test270/119在原始1mm栅格上Scan {d["scan_dice"]:.6f}、Subject {d["subject_dice"]:.6f}；原193基线重采样到同栅格为{tri["scan_dice"]:.6f}，配对差{c["scan_delta"]:+.6f}、95%CI[{c["scan_delta_ci95"][0]:+.4f},{c["scan_delta_ci95"][1]:+.4f}]，{verdict(c)}。不能与128³栅格的0.759179直接相减。分辨率因素已量化并关闭。[完整结果]({REL}/RESULT.md)。');p.write_text('\n'.join(lines)+'\n')
    write(TRAIN_RECORD/'TEST_ACCESS_ADDENDUM.json',{'utc':now(),'test_access':True,'test_checkpoint_epoch':173,'result_path':REL+'/RESULT.json','result_sha256':sha(R/'RESULT.json'),'preaccess_sha256':sha(R/'PREACCESS.json'),'scope':'training FINAL_RESULT.json remains the frozen validation-only receipt; experiment-wide access now includes this one test270 evaluation','no_new_outer_confirmation_access':True})
    p=TRAIN_RECORD/'RESULT.md';s=p.read_text();s+=f'\n**后续用户授权测试已完成（2026-09-20）：** 第173轮test270/119原始栅格Scan {d["scan_dice"]:.6f}，同栅格对193配对差{c["scan_delta"]:+.6f}、CI[{c["scan_delta_ci95"][0]:+.6f},{c["scan_delta_ci95"][1]:+.6f}]；上文“无test”是训练完成时的历史状态，[测试结果](../2026-09-20_segresnet_native_res_e173_test270/RESULT.md)与访问补记优先。\n';p.write_text(s)
    print(json.dumps({'new_rank':next(x['rank'] for x in rows if x['method']==METHOD),'rows_total':len(rows),'scan_dice':d['scan_dice'],'baseline_same_grid':tri['scan_dice'],'paired':c['scan_delta'],'ci':c['scan_delta_ci95'],'verdict':verdict(c)},ensure_ascii=False))

if __name__=='__main__':main()
