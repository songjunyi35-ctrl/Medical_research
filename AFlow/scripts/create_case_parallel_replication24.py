"""Auditable mechanical derivation; never edits the historical benchmark."""
import ast
import difflib
import json
from scripts import structured_background_protocol as p

RECORD=p.ROOT/'archive/research_records/2026-09-11_case_parallel_replication24'


def main():
    old='scripts/benchmark_case_parallel_preprocessing.py'
    new='scripts/benchmark_case_parallel_replication24.py'
    prior=json.loads((p.ROOT/'archive/research_records/2026-09-11_case_parallel_preprocessing/PREACCESS.json').read_text())
    p.checked(p.ROOT/old,prior['files'][old])
    source=(p.ROOT/old).read_text();code=source
    replacements=[
        ("2026-09-11_case_parallel_preprocessing'","2026-09-11_case_parallel_replication24'"),
        ('fit=p.rows(p.FIT)[:8]',"all_fit=p.rows(p.FIT); excluded={r['subject_id'] for r in all_fit[:8]}; fit=[r for r in all_fit if r['subject_id'] not in excluded][:24]\n    assert len(fit)==24 and len({r['subject_id'] for r in fit})==12 and not excluded & {r['subject_id'] for r in fit}"),
        ("'scripts/benchmark_case_parallel_preprocessing.py'","'scripts/benchmark_case_parallel_replication24.py','scripts/create_case_parallel_replication24.py'"),
        ("'expected_forwards':280","'expected_forwards':792,'excluded_prior_subjects':sorted(excluded),'selected_subjects':12"),
        ('deadline=started+1200','deadline=started+2400'),
        ("'cases_per_second':8/outputs_done","'cases_per_second':24/outputs_done"),
        ('assert len(rows)==8','assert len(rows)==24'),
        ('assert calls[0]==280 and len(runs)==4 and len(baseline)==8','assert calls[0]==792 and len(runs)==4 and len(baseline)==24'),
        ("'input_exact':32,'output_exact':32","'input_exact':96,'output_exact':96")]
    for before,after in replacements:
        assert code.count(before)==1,(before,code.count(before))
        code=code.replace(before,after)
    ast.parse(code)
    with (p.ROOT/new).open('x') as f:f.write(code)
    with (RECORD/'DERIVATION.patch').open('x') as f:f.write(''.join(difflib.unified_diff(source.splitlines(True),code.splitlines(True),fromfile=old,tofile=new)))
    p.write_new(RECORD/'DERIVATION.json',{'source':old,'source_sha256':p.sha(p.ROOT/old),
        'derived':new,'derived_sha256':p.sha(p.ROOT/new),'replacements':replacements,
        'generator_sha256':p.sha(p.ROOT/'scripts/create_case_parallel_replication24.py'),
        'source_unchanged':p.sha(p.ROOT/old)==prior['files'][old]})
    print('Created '+new+'; historical source unchanged')


if __name__=='__main__':main()
