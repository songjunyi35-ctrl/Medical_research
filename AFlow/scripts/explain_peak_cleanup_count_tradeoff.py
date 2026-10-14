"""Posthoc saved-count explanation only: no new inference or rule selection."""
from datetime import datetime,timezone
import json
from scripts import peak_cleanup_protocol as p


def main():
    path=p.RECORD/'RAW_DEVELOPMENT.json';raw=json.loads(path.read_text())
    results={}
    for seed,values in raw['results'].items():
        tp=fp=0;checked=0;positive=negative=zero=0
        for a,b,c in zip(values['base']['per_case'],values['clean']['per_case'],values['component_diagnostics'],strict=True):
            true_removed=c['removed_true_positive_voxels'];false_removed=c['removed_false_positive_voxels']
            # Exact integer condition: I*(t+f) > (P+T)*t iff removal improves Dice.
            margin=a['intersection']*(true_removed+false_removed)-(a['predicted_voxels']+a['target_voxels'])*true_removed
            observed=b['dice']-a['dice']
            assert (margin>0)==(observed>0) and (margin<0)==(observed<0)
            positive+=int(margin>0);negative+=int(margin<0);zero+=int(margin==0)
            tp+=true_removed;fp+=false_removed;checked+=1
        results[seed]={'cases_exact_integer_sign_verified':checked,'improved':positive,'worse':negative,'tied':zero,'removed_true_voxels':tp,'removed_false_voxels':fp,'pooled_removed_voxel_precision_descriptive_not_mean_dice_predictor':tp/(tp+fp)}
    p.write_new(p.RECORD/'COUNT_TRADEOFF.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'raw_sha256':p.sha(path),'script_sha256':p.sha(__file__),'results':results,'new_forwards':0,'threshold_or_rule_changed':False,'status':'posthoc_algebraic_explanation_not_new_hypothesis_test'})
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
