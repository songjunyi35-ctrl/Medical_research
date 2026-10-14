"""Check prior external raw/receipt/leaderboard linkage without images or inference."""
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics

from scripts import structured_background_protocol as p

RECORD = p.ROOT / 'archive/reviews/2026-09-07_30h_completion_audit'
PANELS = [
    ('official_float_source_correction_20260906', 'external_ucsd75_direct_locator_rank_by_scan', 'VERIFICATION_RECEIPT.json', 75, 53),
    ('brats_africa95_external_20260906', 'external_brats_africa95_direct_locator_rank_by_scan', 'VERIFICATION_RECEIPT_V3.json', 95, 95),
    ('cptac33_africa_informed_confirmation_20260906', 'external_cptac33_direct_locator_rank_by_scan', 'VERIFICATION_RECEIPT.json', 33, 33),
    ('waw_gliomai152_external_20260906', 'external_waw_gliomai152_direct_locator_rank_by_scan', 'VERIFICATION_RECEIPT.json', 152, 152),
    ('waw_postop_only31_external_20260906', 'external_waw_postop_only31_direct_locator_rank_by_scan', 'VERIFICATION_RECEIPT.json', 31, 31),
]


def recompute(summary):
    rows = summary['per_case']
    assert rows and len({r['case_id'] for r in rows}) == len(rows)
    subjects = defaultdict(list)
    overlap_checks = 0
    for row in rows:
        assert row['target_wt_voxels'] > 0 and 0 <= row['locator_dsc'] <= 1
        subjects[row['subject_id']].append(row['locator_dsc'])
        if 'intersection_voxels' in row:
            inter, pred, target = (row[k] for k in ('intersection_voxels', 'predicted_wt_voxels', 'target_wt_voxels'))
            assert 0 <= inter <= min(pred, target)
            assert abs(2 * inter / (pred + target) - row['locator_dsc']) < 1e-12
            overlap_checks += 1
    result = {'scans': len(rows), 'subjects': len(subjects),
              'scan_dice': statistics.fmean(r['locator_dsc'] for r in rows),
              'subject_dice': statistics.fmean(statistics.fmean(v) for v in subjects.values()),
              'point_hits': sum(r['automatic_point_hits_gt'] for r in rows),
              'dice_recomputed_from_intersection_rows': overlap_checks}
    assert result['scans'] == summary['cases']
    assert abs(result['scan_dice'] - summary['locator_dsc_mean']) < 1e-12
    assert abs(result['subject_dice'] - summary['locator_subject_dsc_mean']) < 1e-12
    assert result['point_hits'] == summary['automatic_point_hits']
    return result


def main():
    out = RECORD / 'EXTERNAL_RESULT_LINKS.json'
    if out.exists():
        raise FileExistsError(out)
    leaderboard_path = p.ROOT / 'AFLOW_BRATS_LEADERBOARD.json'
    board = json.loads(leaderboard_path.read_text())
    results = {}
    for metadata_key, list_key, receipt_name, scans, subjects in PANELS:
        metadata = board[metadata_key]
        raw_path = p.ROOT / metadata['raw_report_path']
        p.checked(raw_path, metadata['raw_report_sha256'])
        raw = json.loads(raw_path.read_text())
        receipt_path = raw_path.parent / receipt_name
        receipt = json.loads(receipt_path.read_text())
        assert receipt['status'] == 'passed'
        assert receipt['cases'] == scans and receipt['subjects'] == subjects
        assert receipt.get('raw_report_sha256', receipt.get('raw_sha256')) == p.sha(raw_path)
        index_sha = receipt['cache_index_archive_sha256']
        index_paths = [f for f in raw_path.parent.glob('*INDEX*.json') if p.sha(f) == index_sha]
        assert index_paths, 'archived cache index absent or mismatched'
        relevant = [r for r in board[list_key] if r['report_path'] == metadata['raw_report_path']]
        panels = {}
        for model_key in ('full972_kd168', 'gcsam184'):
            model = raw[model_key]
            styles = [('identity', 'summary', 'bootstrap')] if 'summary' in model else [
                (view, view, view + '_bootstrap') for view in ('identity', 'flip8') if view in model]
            for view, summary_key, bootstrap_key in styles:
                metrics = recompute(model[summary_key])
                assert (metrics['scans'], metrics['subjects']) == (scans, subjects)
                cost = 8 if view == 'flip8' else 1
                matches = [row for row in relevant if row['checkpoint_sha256'] == model['checkpoint_sha256'] and row['locator_forwards'] == cost]
                assert len(matches) == 1
                row = matches[0]
                assert row['report_sha256'] == p.sha(raw_path)
                assert row['cohort']['scans'] == scans and row['cohort']['subjects'] == subjects
                assert row['resolution'] == 128 and row['threshold'] == 0
                for key in ('scan_dice', 'subject_dice'):
                    assert abs(row[key] - metrics[key]) < 1e-12
                assert row['point_hits'] == metrics['point_hits'] and row['point_denominator'] == scans
                interval = model[bootstrap_key]
                assert row['scan_dice_95ci'] == interval['scan_dsc_mean_95ci']
                assert row['subject_dice_95ci'] == interval['subject_dsc_mean_95ci']
                panels[f'{model_key}_{view}'] = {**metrics, 'leaderboard_rank': row['rank'],
                                                'stored_absolute_ci_transcription_exact': True,
                                                'locator_forwards': cost}
        assert len(panels) == len(relevant)
        results[metadata_key] = {'raw_path': str(raw_path.relative_to(p.ROOT)),
                                 'raw_sha256': p.sha(raw_path),
                                 'receipt_path': str(receipt_path.relative_to(p.ROOT)),
                                 'receipt_sha256': p.sha(receipt_path),
                                 'archived_cache_indices_with_exact_sha': [str(f.relative_to(p.ROOT)) for f in index_paths],
                                 'cache_index_sha256': index_sha,
                                 'metrics': panels}
    p.write_new(out, {'created_utc': datetime.now(timezone.utc).isoformat(), 'status': 'passed',
                     'panels': results, 'leaderboard_sha256_at_audit': p.sha(leaderboard_path),
                     'script_sha256': p.sha(Path(__file__)),
                     'access': {'saved_external_reports_only': True, 'new_model_forwards': 0,
                                'image_label_or_checkpoint_arrays': False},
                     'limits': ['No current rehash of source NIfTI or large image/label caches.',
                                'UCSD raw lacks intersections; there only saved per-case Dice means can be recomputed.',
                                'Absolute CI transcription checked; bootstrap not rerun in this audit.',
                                'No pooled cohort metric or new performance evidence; Fable UCSD56 is separate.']})
    print(json.dumps({'status': 'passed', 'panels': len(results),
                      'leaderboard_rows_checked': sum(len(v['metrics']) for v in results.values())}))


if __name__ == '__main__':
    main()
