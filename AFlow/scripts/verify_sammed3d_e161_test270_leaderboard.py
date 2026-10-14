"""Additional matched interactive table checks, preserving historical validator bytes."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.verify_aflow_brats_leaderboard import validate_leaderboard,validate_ranked_rows,sha256_file,ROOT
j=validate_leaderboard(ROOT/'AFLOW_BRATS_LEADERBOARD.json',ROOT/'AFLOW_BRATS_LEADERBOARD.md')
key='test270_sammed3d_gtcentered_best5_rank_by_scan';rows=j[key];validate_ranked_rows(key,rows)
md=(ROOT/'AFLOW_BRATS_LEADERBOARD.md').read_text().split('<!-- SAM_GRAY_E050_TEST270 -->')[1].split('<!-- END_SAM_GRAY_E050_TEST270 -->')[0]
assert len(rows)==3
for r in rows:
 assert r['cohort']['scans']==270 and r['cohort']['subjects']==119 and r['point_hit'] is None
 assert sha256_file(ROOT/r['report_path'])==r['report_sha256']
 assert f"{r['scan_dice']:.9f}" in md and f"{r['subject_dice']:.9f}" in md
 assert r['reused'] and not r['test_tuned'] and not r['external']
 if r.get('checkpoint_path'):assert sha256_file(ROOT/r['checkpoint_path'])==r['checkpoint_sha256']
positions=[md.index({21:'旧基础模型，第21轮',50:'灰度模型，第50轮',161:'灰度模型，第161轮（事后诊断）'}[r['checkpoint_epoch']]) for r in rows]
assert positions==sorted(positions)
print(json.dumps({'status':'passed','matched_interactive_test_rows':3}))
