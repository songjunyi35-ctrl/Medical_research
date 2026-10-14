"""Additional checks for the independent target-tuned leaderboard group."""
import sys,json,hashlib,re
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.verify_aflow_brats_leaderboard import validate_ranked_rows
ROOT=Path(__file__).resolve().parents[1]

def main():
 d=json.loads((ROOT/'AFLOW_BRATS_LEADERBOARD.json').read_text());rows=d.get('test270_tuned_native_rank_by_scan',[])
 if rows:
  validate_ranked_rows('test270_tuned_native_rank_by_scan',rows)
  md=(ROOT/'AFLOW_BRATS_LEADERBOARD.md').read_text().split('<!-- TEST270_TUNED_NATIVE -->')[1].split('<!-- END_TEST270_TUNED_NATIVE -->')[0]
  lines=[s for s in md.splitlines() if re.match(r'^\|\s*\d+\s*\|',s)];assert len(lines)==len(rows)
  for r,line in zip(rows,lines):
   assert r['test_tuned'] and r['cohort']=={'scans':270,'subjects':119}
   assert r['forward_equivalents']<=8+1e-10
   assert r['point_denominator']==270 and abs(r['point_hits']/270-r['point_hit'])<1e-12
   path=ROOT/r['report_path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==r['report_sha256']
   report=json.loads(path.read_text());assert len(report['per_case'])==270 and len({x['subject_id'] for x in report['per_case']})==119
   assert r['method'] in line and f"{r['scan_dice']:.9f}" in line
 print(json.dumps({'status':'passed','test_tuned_rows':len(rows)}))

if __name__=='__main__':main()
