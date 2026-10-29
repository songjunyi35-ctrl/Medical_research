"""Text-only audit/report from completed108 records; no arrays or models loaded."""
from scripts.race_v2_inference_a_20261029.common import *
from scripts.race_v2_inference_a_20261029.statistics import paired,summary_ci
def main():
    rank=read(R/'RANKING.json');data=read(R/'RAW_VALIDATION108.json')['per_configuration'];best=rank['best'];records=data[best['id']]
    assert len(data)==32 and all(len(values)==108 for values in data.values())
    comparisons={}
    for name in ('raw_identity_t05','equal8_logit_t05','legacy_full'):
        reference=data[name];right={r['case_id']:r for r in reference}
        changed=[dict(case_id=r['case_id'],subject_id=r['subject_id'],dice=r['locator_dsc'],reference_dice=right[r['case_id']]['locator_dsc'],delta=r['locator_dsc']-right[r['case_id']]['locator_dsc']) for r in records]
        comparisons[name]=dict(paired=paired(records,reference),all_worsened=sorted([r for r in changed if r['delta']<0],key=lambda r:r['delta']),all_changed=changed)
    old=read(ROOT/'archive/research_records/2026-10-28_race_seg/race/validation_e164.json')['per_case'];lookup={r['case_id']:r for r in old}
    delta=[dict(case_id=r['case_id'],historical_dice=lookup[r['case_id']]['locator_dsc'],current_dice=r['locator_dsc'],dice_delta=r['locator_dsc']-lookup[r['case_id']]['locator_dsc'],point_equal=r['automatic_point']==lookup[r['case_id']]['automatic_point']) for r in data['raw_identity_t05']]
    write(R/'BASELINE_EXECUTION_COMPARISON.json',dict(at=now(),cases=delta,current=summary_ci(data['raw_identity_t05']),paired_vs_historical=paired(data['raw_identity_t05'],old),note='same weights and exact input payloads; historical validation batch1 versus new batch8/tail4 and hardware; numerical execution may differ, no score retuning'))
    write(R/'ALL_COMPARISONS.json',dict(at=now(),best_id=best['id'],references=comparisons,intervals_descriptive=True,test_tuned_history=True))
    refs={r['id']:r for r in rank['references']};fwd=read(R/'ATOMS_READY.json');ci=rank['best_metrics_ci'];contrast=comparisons['legacy_full']['paired']
    table_rows=[refs['raw_identity_t05'],refs['equal8_logit_t05'],refs['legacy_full']]
    if best['id'] not in {r['id'] for r in table_rows}:table_rows.append(best)
    table='\n'.join('| '+r['id']+' | '+f"{r['scan_dice']:.9f} | {r['subject_dice']:.9f} | {r['point_hits']}/108 | {r['neural_case_forwards']} |" for r in table_rows)
    text=f'''# RACE164专属108验证推理校准完成

固定RACE164、单T1ce、原108/50、原生1mm裁剪全GT评分，32个事前冻结全局配置。最高配置为`{best['id']}`，Scan **{best['scan_dice']:.9f}**、Subject **{best['subject_dice']:.9f}**、Point **{best['point_hits']}/108**。全部为既有test-tuned研究历史上的复用验证选择，非独立泛化确认。

| 配置 | Scan | Subject | Point | 部署前向/例 |
|---|---:|---:|---:|---:|
{table}

对当前完整配方Scan差{contrast['scan_delta']:+.9f}，患者整簇配对描述性95%区间{contrast['scan_delta_ci95']}；Subject差{contrast['subject_delta']:+.9f}，区间{contrast['subject_delta_ci95']}。Point少1，Subject配对区间跨0，不能称三项联合稳定提升。最高配置Scan区间{ci['scan_ci95']}。均为10000次患者整簇bootstrap、seed20261029，未校正32配置选择。

固定5折按患者隔离，四折选参/留出折汇总Scan {rank['heldout_selected_metrics']['scan_dice']:.9f}、Subject {rank['heldout_selected_metrics']['subject_dice']:.9f}、Point {rank['heldout_selected_metrics']['point_hits']}/108；留出折既有配方Scan {rank['heldout_legacy_metrics']['scan_dice']:.9f}。各折与全部候选表现保存在RANKING.json，并非新独立确认。

108使用160×208×160历史画布，4个实际载荷SHA与全部几何匹配旧108。RACE镜像参照矩阵T R T共轭、非零支撑同步翻转。一次864病例前向，batch{fwd['batch']}/尾4，耗时{fwd['seconds']:.1f}秒、reserved峰值{fwd['peak_reserved_gib']:.3f}GiB；32解码不增加网络前向。Point的raw/equal采用logit argmax；完整配方及变体保持历史概率argmax和饱和并列首位置规则，二者没有混为同一定位规则。

全部32配置逐例TP/FP/FN、Point、实际阈值和视图权重保存在RAW_VALIDATION108.json；相对三个参考的全部变差病例保存在ALL_COMPARISONS.json。三个预定108槽位0/53/107的旧完整配方CPU/GPU解码掩膜、Point和量化cut全部相同，神经前向增加0。原始推理与旧训练验证的执行差异见BASELINE_EXECUTION_COMPARISON.json，不把新batch/hardware声称为逐位同轨迹。

旧270四载荷、index、权重SHA及2160/2160既有logits的SHA/shape/dtype/finite全部核验通过，可在总控冻结最多两个候选后复用。本阶段新增270前向0、新270解码评分0。

[计划](PLAN.json) · [32配置逐例](RAW_VALIDATION108.json) · [排名与分折](RANKING.json) · [完整比较](ALL_COMPARISONS.json) · [入选配方](SELECTED_RECIPE.json) · [270缓存只读审计](EXISTING270_CACHE_AUDIT.json)
'''
    (R/'RESULT.md').write_text(text)
    write(R/'REPORT_AUDIT.json',dict(at=now(),status='passed',configurations=32,rows_per_configuration=108,all_references_and_worsened_saved=True,raw_sha256=sha(R/'RAW_VALIDATION108.json'),ranking_sha256=sha(R/'RANKING.json'),source_sha256=sha(Path(__file__)),model_forwards_added=0,new270_scoring=False))
if __name__=='__main__':main()
