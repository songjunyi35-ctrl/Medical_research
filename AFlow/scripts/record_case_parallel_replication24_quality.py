"""Capture validator outputs and parse actual JUnit results after timed runs."""
from datetime import datetime,timezone
import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from scripts import structured_background_protocol as p

RECORD=p.ROOT/'archive/research_records/2026-09-11_case_parallel_replication24'


def main():
    result=json.loads((RECORD/'RESULT.json').read_text())
    verification=json.loads((RECORD/'VERIFICATION.json').read_text())
    assert p.sha(RECORD/'RESULT.json')==verification['result_sha256']
    suites=list(ET.parse(RECORD/'REGRESSION.xml').getroot().iter('testsuite'))
    tests=sum(int(s.get('tests','0')) for s in suites)
    failures=sum(int(s.get('failures','0')) for s in suites)
    errors=sum(int(s.get('errors','0')) for s in suites)
    skipped=sum(int(s.get('skipped','0')) for s in suites)
    assert tests>=986 and failures==errors==0
    validators={}
    for script in ('verify_aflow_brats_doc_layout.py','verify_aflow_brats_method_catalog.py','verify_aflow_brats_leaderboard.py'):
        r=subprocess.run([sys.executable,str(p.ROOT/'scripts'/script)],cwd=p.ROOT,capture_output=True,text=True)
        assert r.returncode==0,r.stdout+r.stderr
        validators[script]={'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
    links=[]
    for file in RECORD.glob('*.md'):
        base=p.ROOT if file.name=='CARD.md' else RECORD
        for target in re.findall(r'\]\(([^)]+)\)',file.read_text()):
            if target.startswith(('http://','https://','#')):continue
            destination=(base/target.strip('<>').split('#',1)[0]).resolve()
            assert destination.exists(),f'broken link {file.name}: {target}'
            links.append({'file':file.name,'target':target})
    p.write_new(RECORD/'QUALITY.json',{'status':'passed','created_utc':datetime.now(timezone.utc).isoformat(),
        'regression':{'tests':tests,'failures':failures,'errors':errors,'skipped':skipped,
                      'junit_sha256':p.sha(RECORD/'REGRESSION.xml')},
        'validators':validators,'archive_links_checked':links,
        'result_sha256':p.sha(RECORD/'RESULT.json'),'verification_sha256':p.sha(RECORD/'VERIFICATION.json'),
        'engineering_gate_passed':result['all_gates_passed']})
    print(json.dumps({'quality':'passed','regression_tests':tests,'archive_links':len(links)},indent=2))


if __name__=='__main__':main()
