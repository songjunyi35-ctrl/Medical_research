"""Validate saved diagnostic aggregates; no images, model, or parameter changes."""
import json
from pathlib import Path
import numpy as np
from scripts import weak_elastic_protocol as p


def main():
    raw_path=p.RECORD/'FIT_AUGMENTATION_AUDIT.json';d=json.loads(raw_path.read_text())
    pre=p.RECORD/'FIT_AUGMENTATION_AUDIT_PREACCESS.json';p.checked(pre,d['preaccess_sha256'])
    for name,digest in json.loads(pre.read_text())['files'].items():p.checked(p.ROOT/name,digest)
    fit,_=p.check_roles();rows=d['per_case'];s=d['summary']
    assert [(r['case_id'],r['subject_id']) for r in rows]==[(r['case_id'],r['subject_id']) for r in fit]
    assert len(rows)==s['scans']==684 and s['subjects']==313
    assert all(r['warped_voxels']>0 and abs(r['relative_volume_change']-(r['warped_voxels']-r['original_voxels'])/r['original_voxels'])<1e-12 for r in rows)
    missing=[v for r in rows for v in r['missing_component_sizes']]
    assert len(missing)==s['missing_component_count'] and sum(missing)==s['missing_component_voxels']
    assert sum(r['components'] for r in rows)==s['original_components']
    assert sum(r['original_voxels'] for r in rows)==s['original_wt_voxels']
    assert abs(np.mean([r['overlap_dice_not_model_score'] for r in rows])-s['mean_deformed_label_overlap_not_model_dice'])<1e-12
    expected=np.quantile([abs(r['relative_volume_change']) for r in rows],[.5,.95,1])
    assert np.allclose(expected,[s['absolute_relative_volume_change_percentiles'][k] for k in ('p50','p95','max')],rtol=0,atol=1e-12)
    assert d['new_model_forwards']==0 and not any(v for k,v in d['access'].items() if k!='fit_labels')
    verification=p.RECORD/'FIT_AUGMENTATION_AUDIT_VERIFICATION.json'
    p.write_new(verification,{'status':'passed','raw_sha256':p.sha(raw_path),'preaccess_sha256':p.sha(pre),'verifier_sha256':p.sha(Path(__file__)),'scans':684,'summary_recomputed':True,'patient_order_exact':True,'new_model_forwards':0,'access':'metadata_and_saved_diagnostic_only'})
    lines=['# 冻结弱形变：全684训练标签压力诊断','',
           '全684扫描/313人完成，全部整WT仍非空，二值/位移/梯度/零边界合同通过。正式训练参数完全不变；没有模型/开发/测试前向。','',
           f"原2277个128³连通块中5个ID消失，合计6个输入体素、最大2体素、涉及5例；占原WT {s['original_wt_voxels']}体素的{6/s['original_wt_voxels']*100:.8f}%。不是5例肿瘤整体消失。",'',
           '绝对相对体积变化中位1.6468%、p95 5.3217%、最大12.7193%。最小WT228→199（BraTS-GLI-02236-100）；应记录边界离散影响，不能说形变完全无损。','',
           '变形前后掩膜重叠Dice均值0.913754、最低0.289030；最低例BraTS-GLI-02932-101为484→464体素。坐标被有意改变，低重叠不等于模型失败、标签错配或大比例标签消失。100%例各变形一次的压力样本也不等于正式20%抽样下的真实发生率。','',
           f"成本{d['seconds']:.2f}秒、1CPU进程2线程、peak RSS {d['peak_rss_kib']/1024**2:.3f}GiB（含mmap驻留，高于事前<2GiB估计，非既定停止门）；0GPU，报告{raw_path.stat().st_size} bytes，无新影像cache。",'',
           '结论：工程安全直觉得到此诊断支持，少量微小块与离散体积变化确实存在；精度收益只能等待冻结dev门。无强度/概率修订，不新增病例排除。','',
           '[七问](FIT_AUGMENTATION_AUDIT_IDEA.md)、[preaccess](FIT_AUGMENTATION_AUDIT_PREACCESS.json)、[raw](FIT_AUGMENTATION_AUDIT.json)、[验证](FIT_AUGMENTATION_AUDIT_VERIFICATION.json)。','',f'Raw SHA `{p.sha(raw_path)}`；preaccess SHA `{p.sha(pre)}`。']
    with (p.RECORD/'FIT_AUGMENTATION_AUDIT.md').open('x') as f:f.write('\n'.join(lines)+'\n')
    print(json.dumps({'status':'passed','raw_sha256':p.sha(raw_path),'verification_sha256':p.sha(verification),'peak_gib':d['peak_rss_kib']/1024**2},indent=2))


if __name__=='__main__':main()
