"""Isolated wrapper: same training implementation, only sampling weights changed."""
import json
from pathlib import Path
import sys
from scripts import subject_balanced_protocol as p
from scripts import train_structured_background_aux as training
from scripts.build_t1ce_wt_native_volume_artifact import native_sampling_weights
from sammed3d.subject_balanced_sampling import normalize_subject_weights


def balanced_weights(case_ids,native_volumes):
    mapping={r['case_id']:r['subject_id'] for r in p.rows(p.FIT)}
    return normalize_subject_weights(case_ids,native_sampling_weights(case_ids,native_volumes),mapping)


def main():
    p.check_freeze()
    if '--arm' not in sys.argv or sys.argv[sys.argv.index('--arm')+1]!='subject_balanced':
        raise ValueError('arm must be subject_balanced: auxiliary coefficient stays zero')
    seed=int(sys.argv[sys.argv.index('--seed')+1])
    if seed not in p.SEEDS:raise ValueError('seed')
    p.write_new(p.RECORD/f'WRAPPER_ENTRY_{seed}.json',{'seed':seed,'arm':'subject_balanced','only_training_override':'native_sampling_weights -> per-subject normalized native weights','auxiliary_coefficient':0,'source_training_sha256':p.sha(Path(training.__file__)),'wrapper_sha256':p.sha(Path(__file__)),'no_checkpoint_warm_start':True})
    # These assignments affect this fresh Python process only. Source files stay frozen.
    training.p=p
    training.native_sampling_weights=balanced_weights
    training.main()


if __name__=='__main__':main()
