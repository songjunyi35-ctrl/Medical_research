"""Verify saved count identities and aggregate the completed label-only audit."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import numpy as np
from scripts import structured_background_protocol as p

RECORD = p.ROOT / 'archive/research_records/2026-09-08_fit_spatial_roundtrip_v2'
BINS = ('1', '2-8', '9-27', '28-125', '126-1000', '>1000')


def verify_row(row):
    components = row['components']
    details = components['details']
    assert [d['id'] for d in details] == list(range(1, len(details) + 1))
    assert len(details) == components['native_count_26']
    assert sum(d['native_voxels'] for d in details) == row['native_WT_voxels']
    assert sum(d['crop_voxels'] for d in details) == row['crop_WT_voxels']
    assert sum(d['cached_voxels'] for d in details) == row['cached_WT_voxels']
    assert row['crop_lost_WT_voxels'] == row['native_WT_voxels'] - row['crop_WT_voxels']
    assert abs(row['crop_only_dice'] - 2 * row['crop_WT_voxels'] / (row['native_WT_voxels'] + row['crop_WT_voxels'])) < 1e-12
    for d in details:
        assert 0 <= d['crop_voxels'] <= d['native_voxels'] and d['native_voxels'] > 0 and d['cached_voxels'] >= 0
        assert d['represented'] == (d['cached_voxels'] > 0)
        assert d['fully_crop_lost'] == (d['crop_voxels'] == 0)
        assert d['resampling_lost'] == (d['crop_voxels'] > 0 and d['cached_voxels'] == 0)
        assert d['partially_cropped'] == (0 < d['crop_voxels'] < d['native_voxels'])
        assert d['size_bin'] == BINS[sum(d['native_voxels'] > t for t in (1, 8, 27, 125, 1000))]
        assert all(0 <= v <= d['native_voxels'] for v in d['restored_native_overlap_voxels'].values())
    for key, field in [('represented_native_components', 'represented'), ('fully_crop_lost_components', 'fully_crop_lost'),
                       ('resampling_lost_components', 'resampling_lost')]:
        assert components[key] == sum(d[field] for d in details)
    for mode, metric in row['restorations'].items():
        n, pred, inter = metric['target_voxels'], metric['reconstructed_voxels'], metric['intersection']
        assert n == row['native_WT_voxels'] and 0 <= inter <= min(n, pred)
        assert sum(d['restored_native_overlap_voxels'][mode] for d in details) == inter
        assert abs(metric['dice'] - 2 * inter / (n + pred)) < 1e-12
        denom = row['crop_WT_voxels'] + pred
        assert abs(metric['crop_internal_dice'] - (2 * inter / denom if denom else 1)) < 1e-12
        assert metric['false_negative_voxels'] == n - inter and metric['false_positive_voxels'] == pred - inter
        assert abs(metric['relative_volume_error'] - (pred - n) / n) < 1e-12
        assert abs(metric['absolute_relative_volume_error'] - abs(pred - n) / n) < 1e-12
        assert 0 <= metric['surface_dice_one_native_voxel'] <= 1
        for key in ('hd95_native_voxels', 'mean_surface_gt_to_pred_native_voxels', 'mean_surface_pred_to_gt_native_voxels'):
            value = metric[key]
            assert (value is None and pred == 0) or (value is not None and np.isfinite(value) and value >= 0)
    assert row['cache_byte_exact'] and row['metric_spacing_native_voxels'] == [1., 1., 1.]


def main():
    if (RECORD / 'VERIFICATION.json').exists():
        raise FileExistsError('already finalized')
    result_path = RECORD / 'RESULT.json'
    result = json.loads(result_path.read_text())
    assert result['status'] == 'completed_fit_only_diagnostic'
    pre_path = RECORD / 'PREACCESS.json'
    p.checked(pre_path, result['preaccess_sha256'])
    pre = json.loads(pre_path.read_text())
    for path, digest in pre['files'].items():
        p.checked(Path(path), digest)
    p.checked(p.CACHE / 'labels.u8', pre['labels_sha256'])
    p.checked(p.CACHE / 'index.json', p.INDEX_SHA)
    raw_path = RECORD / 'RAW_CASES.jsonl'
    p.checked(raw_path, result['raw_sha256'])
    rows = [json.loads(line) for line in raw_path.read_text().splitlines()]
    fit, _ = p.check_roles()
    assert len(rows) == 684 and len({r['subject_id'] for r in rows}) == 313
    assert [(r['case_id'], r['subject_id']) for r in rows] == [(r['case_id'], r['subject_id']) for r in fit]
    for row in rows:
        verify_row(row)
    summary = result['summary']
    for mode, metrics in summary['restorations'].items():
        for name, saved in metrics.items():
            if name == 'empty_reconstructions':
                assert saved == sum(r['restorations'][mode]['empty_prediction'] for r in rows)
                continue
            values = [r['restorations'][mode][name] for r in rows]
            finite = [v for v in values if v is not None]
            assert saved['undefined_scans'] == len(rows) - len(finite)
            grouped = defaultdict(list)
            for row, value in zip(rows, values):
                if value is not None:
                    grouped[row['subject_id']].append(value)
            if finite:
                assert abs(statistics.fmean(finite) - saved['scan_mean']) < 1e-12
                assert abs(statistics.fmean(statistics.fmean(v) for v in grouped.values()) - saved['subject_mean']) < 1e-12
                assert np.allclose(np.quantile(finite, [0, .05, .5, .95, 1]), saved['min_p05_median_p95_max'], atol=1e-12, rtol=0)
    all_components = [d for r in rows for d in r['components']['details']]
    derived_bins = {}
    for name in BINS:
        selected = [d for d in all_components if d['size_bin'] == name]
        saved = summary['component_bins'][name]
        actual = {'components': len(selected), 'fully_crop_lost': sum(d['fully_crop_lost'] for d in selected),
                  'resampling_lost': sum(d['resampling_lost'] for d in selected),
                  'represented_at_128': sum(d['represented'] for d in selected),
                  'native_voxels': sum(d['native_voxels'] for d in selected)}
        assert actual == saved
        derived_bins[name] = {**actual, 'resampling_lost_native_voxels': sum(d['native_voxels'] for d in selected if d['resampling_lost'])}
    for key, field in [('native_components_26', 'native_count_26'), ('resampling_lost_components', 'resampling_lost_components'),
                       ('fully_crop_lost_components', 'fully_crop_lost_components'),
                       ('coarse_components_merging_multiple_native', 'coarse_components_merging_multiple_native'),
                       ('native_components_split_in_coarse_grid', 'native_components_split_in_coarse_grid')]:
        assert summary[key] == sum(r['components'][field] for r in rows)
    assert summary['crop_lost_WT_voxels'] == sum(r['crop_lost_WT_voxels'] for r in rows)
    assert summary['cases_with_crop_loss'] == sum(r['crop_lost_WT_voxels'] > 0 for r in rows)
    assert summary['cases_with_resampling_lost_component'] == sum(r['components']['resampling_lost_components'] > 0 for r in rows)
    events = [json.loads(line) for path in (RECORD / 'worker_access').glob('*.jsonl') for line in path.read_text().splitlines()]
    fit_ids = {r['case_id'] for r in fit}
    assert {e['case_id'] for e in events} == fit_ids
    stages = Counter(e['stage'] for e in events)
    assert stages['header_begin'] == stages['label_array_decode_begin'] == stages['completed'] == 685
    first = json.loads((RECORD / 'FIRST_CASE.json').read_text())
    assert {k: v for k, v in first.items() if k not in ('compute_seconds', 'worker_peak_rss_bytes')} == {
        k: v for k, v in rows[0].items() if k not in ('compute_seconds', 'worker_peak_rss_bytes')}
    total_native = sum(r['native_WT_voxels'] for r in rows)
    lost_mass = sum(d['native_voxels'] for d in all_components if d['resampling_lost'])
    receipt = {'created_utc': datetime.now(timezone.utc).isoformat(), 'status': 'passed',
               'result_sha256': p.sha(result_path), 'raw_sha256': p.sha(raw_path),
               'preaccess_sha256': p.sha(pre_path), 'case_count': 684, 'subject_count': 313,
               'row_count_identities_and_aggregate_recomputed': True,
               'first_case_repeat_exact': True, 'source_cache_labels_rehashed': True,
               'access_event_counts': dict(stages), 'unique_fit_labels': len(fit_ids),
               'component_bins_with_mass': derived_bins, 'resampling_lost_component_native_voxels': lost_mass,
               'native_WT_voxels_total': total_native, 'lost_component_mass_fraction': lost_mass / total_native,
               'new_forwards': 0, 'verifier_sha256': p.sha(Path(__file__)),
               'limits': 'Surface metrics checked finite and repeatable on first case, not independently recomputed with a second surface algorithm.'}
    p.write_new(RECORD / 'VERIFICATION.json', receipt)
    lines = ['# fit空间往返保真诊断：全684例结果', '',
             '状态：完成并核验。只读fit684扫描/313人，0模型/0开发或测试性能。以下是GT标签变换保真度，不是模型分割成绩或可达上限。', '',
             '原生WT=labels1/2/3，固定旧image-only crop→nearest-exact128。684/684与旧缓存byte exact；首例独立进程重复一致。', '',
             '| 还原算子 | Scan Dice均值 | Subject Dice均值 | Dice中位数 | HD95中位/95分位（原生体素） | 表面Dice：1体素容差均值 |',
             '|---|---:|---:|---:|---:|---:|']
    for mode, label in [('nearest-exact', '最近邻'), ('trilinear', '三线性二值指示图>0.5')]:
        m = summary['restorations'][mode]
        h = m['hd95_native_voxels']['min_p05_median_p95_max']
        lines.append(f"| {label} | {m['dice']['scan_mean']:.6f} | {m['dice']['subject_mean']:.6f} | {m['dice']['min_p05_median_p95_max'][2]:.6f} | {h[2]:.4f}/{h[3]:.4f} | {m['surface_dice_one_native_voxel']['scan_mean']:.6f} |")
    lines += ['', '三线性还原用的是GT生成的0/1指示图，不是真实网络logit；两算子都按冻结协议报告，不择优成一个模型分数。无bootstrap或显著性检验；这是全训练池的确定性工程描述。', '',
              f"裁剪阶段：{summary['cases_with_crop_loss']}例有WT体素在固定裁剪外，共{summary['crop_lost_WT_voxels']}体素。发现损失仍保留所有病例。", '',
              '## 原生26邻接连通块', '',
              '| 原生体素数 | 原生块数 | 缩小后无自身ID取样 | 完全在crop外 | 缩小丢失块原生体素总数 |',
              '|---|---:|---:|---:|---:|']
    for name, v in derived_bins.items():
        lines.append(f"| {name} | {v['components']} | {v['resampling_lost']} | {v['fully_crop_lost']} | {v['resampling_lost_native_voxels']} |")
    lines += ['', f"原生共{summary['native_components_26']}个连通块，{summary['resampling_lost_components']}个在crop内但缩小后没有自身ID取样，涉及{summary['cases_with_resampling_lost_component']}例；这些块共{lost_mass}个原生WT体素，占全部WT体素{lost_mass/total_native:.6%}。", '',
              f"低分辨率网格中有{summary['coarse_components_merging_multiple_native']}个连通块关联多个原生块；有{summary['native_components_split_in_coarse_grid']}个原生块在低网格分裂。", '',
              '连通块不是临床病灶，可能包含标注碎片；无自身ID取样也不意味着还原二值图一定完全不覆盖它，相邻块可能覆盖其位置。不能把此比例叫模型病灶漏检率。', '',
              '## 单位、验证与代价', '',
              'V1要求所有header显式mm，在保存2行后停止；额外fit预取次数未完整记录，旧记录保留。V2全队列改用原生体素距离，没有猜测毫米或删除14例双unknown；Dice/体积/插值/组件算法不改。', '',
              f"V2耗时{result['elapsed_seconds']:.2f}秒，4 CPU workers；单worker最高RSS {result['max_worker_peak_rss_bytes']/1024**3:.3f}GiB（不是总并发峰值），原始统计文件{result['raw_report_bytes']} bytes，不保存重建体积。访问日志685次标签读取尝试/完成（首例重复一次），唯一fit标签684，0影像像素/模型/开发/test/outer/confirmation/external数组访问。", '',
              '独立验证从保存计数重算Dice、体积、组件分类及所有聚合；表面值验证有限性且首例跨进程复现，没有用第二套表面算法重算全队列。', '',
              '[七问](IDEA.md)、[协议](PREREG.md)、[单位修订](AMENDMENT.md)、[机器结果](RESULT.json)、[逐例/组件原始记录](RAW_CASES.jsonl)、[验证](VERIFICATION.json)、[访问前冻结](PREACCESS.json)。']
    with (RECORD / 'RESULT.md').open('x') as stream:
        stream.write('\n'.join(lines) + '\n')
    paths = [RECORD / name for name in ('IDEA.md', 'PREREG.md', 'AMENDMENT.md', 'PREACCESS.json', 'FIRST_CASE.json', 'RAW_CASES.jsonl', 'RESULT.json', 'RESULT.md', 'VERIFICATION.json')]
    paths += sorted((RECORD / 'worker_access').glob('*.jsonl'))
    paths += [p.CACHE / 'index.json', p.FIT, Path(__file__)]
    p.write_new(RECORD / 'ARCHIVE_SEAL.json', {'status': 'passed', 'files': {str(path): p.sha(path) for path in paths}, 'new_forwards': 0})
    print(json.dumps({k: v for k, v in receipt.items() if k != 'component_bins_with_mass'}, indent=2))


if __name__ == '__main__':
    main()
