"""Re-read every saved soft target and independently check its archived invariants."""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
from scripts import fractional_occupancy_protocol as p


def main():
    output=p.RECORD/'CACHE_VERIFICATION.json'
    if output.exists():raise FileExistsError(output)
    p.source.check_freeze();fit,_=p.check_roles()
    path=p.RECORD/'CACHE_RECEIPT.json';receipt=json.loads(path.read_text());assert receipt['status']=='passed'
    pre=p.RECORD/'CACHE_PREACCESS.json';p.checked(pre,receipt['cache_preaccess_sha256'])
    for name,digest in json.loads(pre.read_text())['files'].items():p.checked(p.ROOT/name,digest)
    p.checked(Path(receipt['target_path']),receipt['target_sha256']);p.checked(p.SOFT_CACHE/'index.json',receipt['index_sha256']);p.checked(p.RECORD/'CACHE_INDEX.json',receipt['index_sha256'])
    index=json.loads((p.RECORD/'CACHE_INDEX.json').read_text())
    assert index['case_ids']==[r['case_id'] for r in fit] and len(receipt['details'])==684
    assert receipt['bytes']==684*128**3*2==Path(receipt['target_path']).stat().st_size
    counts=[];start=time.monotonic();fd=os.open(receipt['target_path'],os.O_RDONLY)
    try:
        for i,(meta,row) in enumerate(zip(fit,receipt['details'],strict=True)):
            assert (meta['case_id'],meta['subject_id'])==(row['case_id'],row['subject_id'])
            body=os.pread(fd,128**3*2,i*128**3*2);assert len(body)==128**3*2
            assert hashlib.sha256(body).hexdigest()==row['target_array_sha256']
            values=np.frombuffer(body,dtype=np.float16)
            assert np.isfinite(values).all() and values.min()>=0 and values.max()<=1
            total=float(values.sum(dtype=np.float64));assert total==row['stored_float16_sum'] and total>0
            expected=row['crop_native_wt_voxels']*128**3/np.prod(row['crop_shape'])
            error=abs(total-expected)/expected
            assert abs(expected-row['expected_occupancy_sum'])<1e-12 and error==row['relative_volume_error_float16'] and error<=.001
            fractional=int(((values>0)&(values<1)).sum());hardened=int((values>.5).sum())
            assert fractional==row['fractional_voxels'] and hardened==row['hardened_half_voxels']
            assert float(values.max())==row['soft_max'] and row['hard_cache_byte_exact']
            counts.append({'case_id':row['case_id'],'relative_mass_error_float16':error,'relative_mass_error_old_hard':abs(row['hard_cache_voxels']-expected)/expected,'fractional_voxels':fractional,'hardened_half_voxels':hardened})
    finally:os.close(fd)
    summary={'scans':684,'subjects':313,'all_case_bytes_and_statistics_exact':True,'soft_mass_error_mean':float(np.mean([r['relative_mass_error_float16'] for r in counts])),'soft_mass_error_max':max(r['relative_mass_error_float16'] for r in counts),'old_hard_mass_error_mean':float(np.mean([r['relative_mass_error_old_hard'] for r in counts])),'fractional_voxels_total':sum(r['fractional_voxels'] for r in counts),'hardened_half_empty_count':sum(r['hardened_half_voxels']==0 for r in counts),'mean_hardened_half_vs_old_hard_dice_not_model':float(np.mean([r['hardened_half_vs_old_hard_dice_not_model'] for r in receipt['details']])),'outside_common_crop_voxels':sum(r['outside_common_crop_wt_voxels'] for r in receipt['details'])}
    p.write_new(output,{'status':'passed','created_utc':datetime.now(timezone.utc).isoformat(),'summary':summary,'cache_receipt_sha256':p.sha(path),'cache_preaccess_sha256':p.sha(pre),'target_sha256':p.sha(Path(receipt['target_path'])),'cache_index_sha256':receipt['index_sha256'],'verifier_sha256':p.sha(Path(__file__)),'verify_seconds':time.monotonic()-start,'new_model_forwards':0,'access':'generated_fit_targets_and_metadata_only','training_admitted':False})
    lines=['# 精确占比训练标签：缓存工程完成','',
           'fit684扫描/313人全部完成，原生label SHA与旧记录一致，原nearest-exact硬WT逐例byte exact。生成的占比文件已逐例重读哈希与统计，范围[0,1]、非空总量、坐标/顺序和量化门全部通过。原影像、硬标签与评分协议没有修改。','',
           f"float64最大相对质量误差{receipt['max_relative_volume_error_float64']:.3g}；float16最大{summary['soft_mass_error_max']:.3g}、平均{summary['soft_mass_error_mean']:.3g}。旧硬标签在同一输出格体积换算下平均质量误差{summary['old_hard_mass_error_mean']:.6f}，只是数据表示诊断，不是模型Dice。",'',
           f"部分占比格总数{summary['fractional_voxels_total']}；占比>.5后整WT为空{summary['hardened_half_empty_count']}例。该硬化结果与旧hard的平均掩膜重叠{summary['mean_hardened_half_vs_old_hard_dice_not_model']:.6f}，不用于改阈值/核。旧公共crop外仍有{summary['outside_common_crop_voxels']}个WT体素，原样保留记录、不排例；质量守恒指裁剪内。",'',
           f"成本：builder {receipt['seconds']:.2f}s、peak RSS {receipt['peak_rss_kib']/1024**2:.3f}GiB；标签文件{receipt['bytes']}bytes=2.672GiB，位于临时目录，0GPU/model/dev/test。原先数分钟估计偏乐观，实际成本如实记录。",'',
           '这只证明构造正确且保留几何占比信息，不是软标签提升分割的证据，也不是医师信心或概率校准。它是原标注单位格恒定占据的数学模型，并非完整MRI扫描点扩散函数模型。图像原三线性重采样仍固定，不能声称整条成像模型已纠正。','',
           '下一步仍为条件性训练：弱形变不能晋级且时间充足才执行两个固定seed；否则作为已准备的备用数据表示保留。部署不需要此标签cache；/tmp不保证跨环境长期保留，已归档builder/原生label SHA/index/receipt以便重建。','',
           '[七问](IDEA.md)、[预注册](PREREG.md)、[cache前冻结](CACHE_PREACCESS.json)、[cache收据](CACHE_RECEIPT.json)、[index](CACHE_INDEX.json)、[验证](CACHE_VERIFICATION.json)。','',f"Target SHA `{receipt['target_sha256']}`；cache receipt SHA `{p.sha(path)}`。"]
    with (p.RECORD/'CACHE_RESULT.md').open('x') as f:f.write('\n'.join(lines)+'\n')
    print(json.dumps({'status':'passed','summary':summary,'cache_receipt_sha256':p.sha(path),'verification_sha256':p.sha(output)},indent=2),flush=True)


if __name__=='__main__':main()
