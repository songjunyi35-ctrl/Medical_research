"""Read-only verification of exact moved bytes and honest regression status."""
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'archive/legacy_root_artifacts_20260914'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    receipt=json.loads((DEST/'MIGRATION_RECEIPT.json').read_text())
    for row in receipt['moves']:
        assert not (ROOT/row['old']).exists() and sha(ROOT/row['new'])==row['sha256']
    old=receipt['historical_handoff']
    assert hashlib.sha256(gzip.decompress((ROOT/old['exact_backup']).read_bytes())).hexdigest()==old['old_sha256']
    for path,digest in receipt['protected_files_unchanged'].items():assert sha(ROOT/path)==digest
    suites=list(ET.parse(DEST/'REGRESSION.xml').getroot().iter('testsuite'))
    tests=sum(int(s.get('tests','0')) for s in suites)
    failed=[{'name':case.get('name'),'classname':case.get('classname'),'message':case.find('failure').get('message')}
            for s in suites for case in s.findall('testcase') if case.find('failure') is not None]
    assert tests==991 and len(failed)==1 and failed[0]['name']=='test_dependency_manifest_and_wheel_are_exact'
    assert 'pydicom-3.0.1-py3-none-any.whl' in failed[0]['message']
    validators={}
    for script in ['verify_aflow_brats_doc_layout.py','verify_aflow_brats_method_catalog.py',
                   'verify_aflow_brats_leaderboard.py','verify_aflow_root_artifacts.py']:
        run=subprocess.run([sys.executable,str(ROOT/'scripts'/script)],cwd=ROOT,capture_output=True,text=True)
        assert run.returncode==0,run.stdout+run.stderr
        validators[script]=json.loads(run.stdout)
    result={'migration_status':'passed','full_regression_status':'one_unrelated_missing_temporary_dependency',
            'tests':tests,'passed':tests-len(failed),'failures':failed,'validators':validators,
            'exact_bytes_preserved':True,'historical_code_and_leaderboards_unchanged':True,
            'migration_receipt_sha256':sha(DEST/'MIGRATION_RECEIPT.json'),'regression_xml_sha256':sha(DEST/'REGRESSION.xml'),
            'final_handoff_sha256':sha(ROOT/'AFLOW_BRATS_HANDOFF_CURRENT.md'),
            'quality_note_sha256':sha(DEST/'QUALITY.md'),'new_experiments_or_dependency_installation':False}
    with (DEST/'QUALITY.json').open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps({'migration_status':'passed','regression_passed':990,'regression_failed':1,'validators':4},indent=2))


if __name__=='__main__':main()
