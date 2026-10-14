"""Frozen nested analysis; never performs model inference or reads image labels."""
import json
from datetime import datetime,timezone
import statistics
from sammed3d.automatic_workflow_search import CONTROLLERS,scoped_rows,search_controller,select_workflow,views_for,reward
from scripts.evaluate_structured_background_aux import summarize,compare
from scripts import structured_background_protocol as p

RECORD=p.ROOT/'archive/research_records/2026-09-11_automatic_workflow_search'


def main():
    if (RECORD/'ANALYSIS.json').exists(): raise FileExistsError('single frozen analysis')
    raw=json.loads((RECORD/'RAW_DEVELOPMENT.json').read_text())
    assert raw['status']=='completed' and raw['case_forwards']==2880 and raw['historical_identity_exact']
    pre=json.loads((RECORD/'DEV_PREACCESS.json').read_text())
    assert p.sha(RECORD/'DEV_PREACCESS.json')==raw['preaccess_sha256']
    for name,digest in pre['files'].items(): p.checked(p.ROOT/name,digest)
    receipts=[];results={};gates={}
    groups=pre['patient_groups'];order=pre['case_ids']
    assert len(groups)==3 and all(len(g)==26 for g in groups)
    assert len(set(sum(groups,[])))==78
    p.write_new(RECORD/'ANALYSIS_PREACCESS.json',{'created_utc':datetime.now(timezone.utc).isoformat(),
        'raw_sha256':p.sha(RECORD/'RAW_DEVELOPMENT.json'),'dev_preaccess_sha256':p.sha(RECORD/'DEV_PREACCESS.json'),
        'analyzer_sha256':p.sha(p.ROOT/'scripts/analyze_automatic_workflow_search.py'),
        'primary_controller':'exhaustive','test_access':False})
    for ordinal,seed in enumerate(p.SEEDS):
        info=raw['results'][str(seed)];p.checked(info['path'],info['sha256'])
        data=json.loads(open(info['path']).read());table={int(k):v for k,v in data['rows'].items()}
        assert set(table)==set(range(128))
        for rows in table.values(): assert [r['case_id'] for r in rows]==order
        collected={name:[] for name in CONTROLLERS};costs={name:[] for name in CONTROLLERS}
        for fold in range(3):
            search_subjects=groups[(fold+1)%3];selector_subjects=groups[(fold+2)%3];assessment_subjects=groups[fold]
            assert not (set(search_subjects)&set(selector_subjects) or set(search_subjects)&set(assessment_subjects) or set(selector_subjects)&set(assessment_subjects))
            search=scoped_rows(table,search_subjects)
            best_reward=max(reward(v,k) for k,v in search.items())
            for name in CONTROLLERS:
                outcome=search_controller(name,search,2026091132+ordinal*100+fold)
                pool=outcome['selector_pool']
                selector=scoped_rows({k:table[k] for k in pool},selector_subjects)
                choice=select_workflow(pool,selector)
                receipt={'seed':seed,'fold':fold,'controller':name,
                         'search_subjects':search_subjects,'selector_subjects':selector_subjects,
                         'assessment_subjects':assessment_subjects,'search':outcome,'selection':choice,
                         'best_search_reward_regret':best_reward-max(outcome['search_rewards'].values()),
                         'assessment_used_for_selection':False}
                # Receipt persisted before selected assessment rows enter any metric.
                p.write_new(RECORD/f'SELECTION_{seed}_{fold}_{name}.json',receipt)
                receipts.append(receipt)
                selected=scoped_rows({choice['selected']:table[choice['selected']]},assessment_subjects)[choice['selected']]
                collected[name].extend(selected)
                costs[name].extend([len(choice['views'])]*len(selected))
        base={str(k):summarize(table[k],2026091141+ordinal) for k in (0,127,64,11)}
        policies={}
        for name in CONTROLLERS:
            by_id={r['case_id']:r for r in collected[name]}
            assert len(by_id)==180
            summary=summarize([by_id[k] for k in order],2026091141+ordinal)
            contrast=compare(base['127'],summary,2026091141+ordinal)
            gate={'mean_forwards_at_most4':statistics.fmean(costs[name])<=4,
                  'scan_loss_at_most003':contrast['scan_delta']>=-.003,
                  'subject_loss_at_most003':contrast['subject_delta']>=-.003,
                  'scan_ci_floor':contrast['scan_cluster_95ci'][0]>=-.005,
                  'subject_ci_floor':contrast['subject_95ci'][0]>=-.005,
                  'point_floor':summary['point_hits']>=max(base['0']['point_hits'],base['127']['point_hits'])-1}
            policies[name]={'summary':summary,'versus_full8':contrast,
                            'versus_identity':compare(base['0'],summary,2026091141+ordinal),
                            'mean_forwards':statistics.fmean(costs[name]),'gates':gate,'passed':all(gate.values())}
        for name in CONTROLLERS:
            if name!='exhaustive': policies[name]['versus_exhaustive']=compare(policies['exhaustive']['summary'],policies[name]['summary'],2026091141+ordinal)
        results[str(seed)]={'fixed':base,'policies':policies}
        gates[str(seed)]=policies['exhaustive']['gates']
    passed=all(all(g.values()) for g in gates.values())
    body={'created_utc':datetime.now(timezone.utc).isoformat(),'status':'completed',
          'results':results,'selection_receipts':receipts,'primary_gates':gates,
          'primary_passed':passed,'raw_sha256':p.sha(RECORD/'RAW_DEVELOPMENT.json'),
          'analysis_preaccess_sha256':p.sha(RECORD/'ANALYSIS_PREACCESS.json'),
          'scope':'conditional nested reused development180/78; not pristine or protected outer',
          'test_access':False,'decision':'eligible_fit_online_verification_only' if passed else 'close_fixed_protocol_accuracy_cost_gate_failed'}
    p.write_new(RECORD/'ANALYSIS.json',body)
    print(json.dumps({'primary_passed':passed,'gates':gates,
          'metrics':{s:{n:{'scan':v['summary']['scan_dice'],'subject':v['summary']['subject_dice'],
                          'point':v['summary']['point_hits'],'mean_forwards':v['mean_forwards']}
                       for n,v in r['policies'].items()} for s,r in results.items()}},indent=2))


if __name__=='__main__': main()
