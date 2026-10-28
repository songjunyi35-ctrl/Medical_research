"""Audit current-study native leaderboard evidence and published Markdown scope."""
import argparse,hashlib,json,re
from pathlib import Path
import numpy as np
from scripts.verify_aflow_brats_leaderboard import validate_ranked_rows


def main(root):
    root=Path(root);study=root/'archive/research_records/2026-10-28_segresnet_auto_inference12h'
    board=json.loads((root/'AFLOW_BRATS_LEADERBOARD.json').read_text());rows=board['test270_tuned_native_rank_by_scan'];validate_ranked_rows('native270',rows)
    own=[r for r in rows if r['method'].startswith('segresnet_w32_e171_auto12h_')];cases=0
    for row in own:
        for kind in ['report','result','selected_recipe']:
            path=root/row[kind+'_path'];assert path.is_file(),path
            assert hashlib.sha256(path.read_bytes()).hexdigest()==row[kind+'_sha256'],path
        report=json.loads((root/row['report_path']).read_text());values=report['per_case'];assert len(values)==270
        assert len({v['case_id'] for v in values})==270 and len({v['subject_id'] for v in values})==119
        for v in values:
            den=2*v['tp']+v['fp']+v['fn'];score=float(np.float32(2*v['tp'])/np.float32(den)) if den else 1.
            assert score==v['locator_dsc'];cases+=1
        scan=float(np.mean([v['locator_dsc'] for v in values],dtype=np.float64));subjects=sorted({v['subject_id'] for v in values})
        subject=float(np.mean([np.mean([v['locator_dsc'] for v in values if v['subject_id']==s],dtype=np.float64) for s in subjects]))
        assert abs(scan-row['scan_dice'])<1e-12 and abs(subject-row['subject_dice'])<1e-12
        assert sum(v['automatic_point_hits_gt'] for v in values)==row['point_hits']
    broken=[];checked=0;documents=list(study.rglob('*.md'))
    pattern=r'<!-- AUTO_INFERENCE12H_20261028 -->.*?<!-- END_AUTO_INFERENCE12H_20261028 -->'
    for name in ['AGENTS.md','AFLOW_BRATS_HANDOFF_CURRENT.md','AFLOW_BRATS_METHOD_CATALOG.md','AFLOW_BRATS_RESULTS.md','AFLOW_BRATS_WEEKLY_REPORT.md']:
        p=root/name;documents.append(p)
    for doc in documents:
        text=doc.read_text()
        if doc.parent==root:text=re.search(pattern,text,re.S).group()
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',text):
            if target.startswith(('https://','http://','#','mailto:')):continue
            path=target.split('#',1)[0];checked+=1
            if not (doc.parent/path).exists():broken.append(dict(document=str(doc),target=target))
    assert not broken,broken
    result=dict(state='passed',native_rows=len(rows),own_audited_rows=len(own),per_case_tp_fp_fn_checks=cases,own_markdown_relative_links_checked=checked,new_broken_own_links=0,scope='native ranking and own source/JSON metric evidence; no model evaluation or full legacy-link claim')
    print(json.dumps(result));return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path.cwd());a=p.parse_args();main(a.root)
