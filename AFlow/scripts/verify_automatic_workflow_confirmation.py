"""Independent saved-logit/mask/point/count/CI verification of the fixed test panel."""
from datetime import datetime,timezone
import json
from pathlib import Path
import numpy as np
from sammed3d.packed_prediction_masks import read_packed_case
from scripts.verify_resumed_test270_panel import verify_case,verify_summary,verify_contrast
from scripts import weak_elastic_test270_helpers as t
from scripts import structured_background_protocol as p

RECORD=p.ROOT/'archive/research_records/2026-09-11_automatic_workflow_confirmation'


def main():
    if (RECORD/'VERIFICATION.json').exists():raise FileExistsError('panel already verified')
    raw=json.loads((RECORD/'RAW_TEST270.json').read_text())
    pre=json.loads((RECORD/'TEST_PREACCESS.json').read_text())
    assert raw['status']=='completed' and raw['case_forwards']==pre['expected_case_forwards']
    assert raw['test_preaccess_sha256']==p.sha(RECORD/'TEST_PREACCESS.json')
    assert raw['online_exact_cases']==540 and raw['test_access'] and not raw['test_tuned']
    for name,digest in pre['files'].items():p.checked(p.ROOT/name,digest)
    for report in pre['models']:p.checked(report['checkpoint'],report['sha256'])
    p.checked(t.TEST,t.TEST_SHA);p.checked(t.CACHE/'index.json',t.INDEX_SHA);p.checked(t.CACHE/'labels.u8',t.LABELS_SHA)
    p.checked(RECORD/'MASK_INDEX.json',raw['mask_index_sha256'])
    masks=json.loads((RECORD/'MASK_INDEX.json').read_text())
    expected=p.rows(t.TEST);ids=[r['case_id'] for r in expected]
    assert masks['case_ids']==ids and set(masks['files'])==set(raw['results'])==set(raw['logits'])
    labels=np.memmap(t.CACHE/'labels.u8',mode='r',dtype=np.uint8,shape=(270,128,128,128))
    checks={};gates={}
    for ordinal,report in enumerate(pre['models']):
        seed=report['seed'];bs=pre['bootstrap_seeds'][ordinal]
        for kind in ('identity','pruned','full8'):
            name=f'seed{seed}_{kind}';summary=raw['results'][name]
            assert [r['case_id'] for r in summary['per_case']]==ids
            info=raw['logits'][name];p.checked(info['path'],info['sha256'])
            assert info['shape']==[270,128,128,128] and info['dtype']=='float32'
            p.checked(masks['files'][name]['path'],masks['files'][name]['sha256'])
            logits=np.memmap(info['path'],mode='r',dtype=np.float32,shape=(270,128,128,128))
            for i,row in enumerate(summary['per_case']):
                assert row['subject_id']==expected[i]['subject_id']
                values=np.asarray(logits[i]);assert np.isfinite(values).all()
                pred=read_packed_case(masks,name,i);assert np.array_equal(pred,values>0)
                assert int(np.argmax(values))==row['point_flat_index']
                verify_case(row,pred,np.array(labels[i],dtype=bool))
            verify_summary(summary,bs)
            checks[name]={'cases':270,'mask_logit_exact':True,'argmax_exact':True,'count_and_ci_verified':True,
                          'logit_sha256':info['sha256'],'mask_sha256':masks['files'][name]['sha256']}
            print('Verified '+name,flush=True)
        a,b,c=(raw['results'][f'seed{seed}_{kind}'] for kind in ('identity','pruned','full8'))
        contrasts=raw['comparisons'][str(seed)]
        verify_contrast(contrasts['pruned_vs_full8'],c,b,bs)
        verify_contrast(contrasts['pruned_vs_identity'],a,b,bs)
        d=contrasts['pruned_vs_full8']
        gates[str(seed)]={'fixed_forwards_at_most4':len(report['medoid']['views'])<=4,
            'scan_loss_at_most003':d['scan_delta']>=-.003,'subject_loss_at_most003':d['subject_delta']>=-.003,
            'scan_ci_floor':d['scan_cluster_95ci'][0]>=-.005,'subject_ci_floor':d['subject_95ci'][0]>=-.005,
            'point_floor':b['point_hits']>=max(a['point_hits'],c['point_hits'])-1}
    assert gates==raw['gates'] and all(all(g.values()) for g in gates.values())==raw['both_models_passed']
    p.write_new(RECORD/'VERIFICATION.json',{'status':'passed','created_utc':datetime.now(timezone.utc).isoformat(),
        'raw_sha256':p.sha(RECORD/'RAW_TEST270.json'),'preaccess_sha256':p.sha(RECORD/'TEST_PREACCESS.json'),
        'mask_index_sha256':p.sha(RECORD/'MASK_INDEX.json'),'checks':checks,'gates_recomputed':gates,
        'models':pre['models'],'new_model_forwards':0,'test_labels_reused_for_audit':True,
        'verifier_sha256':p.sha(Path(__file__))})
    print('All six test outputs independently verified',flush=True)


if __name__=='__main__':main()
