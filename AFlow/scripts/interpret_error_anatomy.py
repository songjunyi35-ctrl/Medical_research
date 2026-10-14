"""Data-derived plain-language interpretation without changing verified results."""
from datetime import datetime,timezone
import json
import statistics
from scripts import structured_background_protocol as p
from scripts.publish_research_card import publish


def main():
    rec=p.ROOT/'archive/research_records/2026-09-10_development_error_anatomy'
    if (rec/'INTERPRETATION.json').exists():raise FileExistsError('already interpreted')
    result=json.loads((rec/'RESULT.json').read_text());verification=json.loads((rec/'VERIFICATION.json').read_text());p.checked(rec/'RESULT.json',verification['result_sha256'])
    values={}
    for key,m in result['summary'].items():
        wt=sum(m['class_target_voxels'][str(i)] for i in [1,2,3]);misses=180-m['point_hits'];hist=m['point_class_histogram']
        values[key]={'class2_share_of_target':m['class_target_voxels']['2']/wt,'class2_share_of_fn':m['class_recall']['2']['false_negative_share'],'class2_pooled_recall':m['class_recall']['2']['pooled_voxel_recall'],'class3_pooled_recall':m['class_recall']['3']['pooled_voxel_recall'],'point_misses':misses,'rc_point_misses':hist.get('4',0),'ordinary_background_point_misses':hist.get('0',0),'rc_share_of_point_misses':hist.get('4',0)/misses,'rc_share_of_fp_voxels':m['rc_share_of_false_positive_voxels']}
    report={'created_utc':datetime.now(timezone.utc).isoformat(),'result_sha256':p.sha(rec/'RESULT.json'),'values':values,'new_forwards':0,'new_model_or_parameter_selection':False,'source_definition_url':'https://raw.githubusercontent.com/rachitsaluja/BraTS-2024-Metrics/main/metrics_GLI.py','not_causal_or_upper_bound':True}
    p.write_new(rec/'INTERPRETATION.json',report)
    lines=['# 什么错误最值得解释？','',
      '这是四个旧模型在同一reused dev180/78上的描述，不是四份独立验证。720例新推理与历史预测计数/点位置exact，720个保存掩膜已独立核对。','',
      '大范围漏画与少数错误最高峰需要分开处理：','',
      '| 模型 | label2占GT | label2占漏画 | label2召回 | label3召回 | 错点中切除腔 |','|---|---:|---:|---:|---:|---:|']
    for key,v in values.items():lines.append(f"| {key} | {v['class2_share_of_target']:.2%} | {v['class2_share_of_fn']:.2%} | {v['class2_pooled_recall']:.2%} | {v['class3_pooled_recall']:.2%} | {v['rc_point_misses']}/{v['point_misses']} ({v['rc_share_of_point_misses']:.2%}) |")
    lines+=['',
      '标签2本来就是大类（83.82%左右），所以约92%漏画来自它不能单独证明更难；同时看它约75%的体素召回，而标签3约90%—92%，才说明覆盖确有明显差异。患者等权召回详见RESULT.json，不能拿总体体素比代替患者Dice。',
      '切除腔内部只占约11%—12%的误画体素，却占约38%—65%的错误最高分点。这里说的是标签4内部，不是测量过“距切除腔边缘多远”；本诊断没有做距离分析。它支持区分面积误差与极值定位误差，不证明新的切除腔方法一定有效。旧辅助系数0.1失败仍保留，不据此重跑该配方。',
      '官方标签2是周围非增强T2/FLAIR高信号区域；FLAIR是液体衰减反转恢复MRI序列，项目当前只输入增强T1图像。术后该类可能含不同组织变化，不能笼统叫所有体素都是活肿瘤/单纯水肿。缺另一序列是信息限制的合理担忧，但本报告不能量化不可约误差、证明达到上限或批准改变单T1ce任务。',
      '当前已注册旋转/模糊参数与触发条件保持原样。下一阶段的问题应是：能否改善周围异常区域覆盖，同时抑制切除腔内错误高峰，而不是继续搜索一个平均分最好看的种子。这里仅提出研究问题，不新增训练配方或test授权。','',
      '成本：720旧模型开发前向40.53秒，另fit8 smoke；类别cache构建阶段报告29.78秒不包含此前远端源文件哈希读取，不能称完整流水线总耗时29.78秒。临时类标签360MiB、mask180MiB。CLASS_CACHE.rows.class_sha256字段实际保存的是源label文件SHA，名称不准确；整体类cache有独立sha256，原文件保留并以本说明澄清。','',
      '[数值](INTERPRETATION.json)、[已验证结果](RESULT.json)、[验证范围](VERIFICATION.json)、[官方类别实现](https://raw.githubusercontent.com/rachitsaluja/BraTS-2024-Metrics/main/metrics_GLI.py)。']
    with (rec/'INTERPRETATION.md').open('x') as f:f.write('\n'.join(lines)+'\n')
    card=(rec/'CARD.md').read_text()
    card+='\n**结果解释补记：** label2占目标约84%却占漏画91%—92%，其召回约75%（增强组织约90%—92%）；切除腔占误画11%—12%却占错误最高点38%—65%。面积与极值问题不同，不证明达到上限，不重开旧0.1辅助头，不改当前旋转/模糊参数。[详细解释与字段澄清](archive/research_records/2026-09-10_development_error_anatomy/INTERPRETATION.md)。\n'
    (rec/'CARD.md').write_text(card);publish(str(rec.relative_to(p.ROOT)))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
