"""Read-only combined status for training and all current inference queues."""
import datetime
import json
import os
import time

from .governance import RECORD, atomic_json, guard
from .select_final_continuation import IA, IB, A, B, TRAIN, NA, NB, SC, EDGE, APPEARANCE


def small(path, keys=None):
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    return value if keys is None else {key: value[key] for key in keys if key in value}


def main():
    while True:
        guard(1154121)
        fields = ('at', 'state', 'stage', 'phase', 'completed', 'total', 'new_epoch',
                  'iteration', 'new_updates', 'best_new_epoch', 'best_scan', 'actual_new_epochs',
                  'actual_new_updates', 'checkpoint_sha256', 'metrics', 'recipe_id', 'partner_model_id',
                  'new_case_forwards', 'new_network_forwards', 'new_model_forwards', 'maximum', 'input_group')
        queues = {}
        for name, folder in [('training', TRAIN), ('innovation_a', IA), ('innovation_b', IB),
                             ('a_transfer', IA / 'final_snapshot'), ('b_transfer', IB / 'final_snapshot'),
                             ('healthy_input_a', NA), ('image_evidence_b', NB), ('soft_context_a', SC),
                             ('guided_input_b', EDGE), ('appearance_b', APPEARANCE),
                             ('a_final', A / 'final270'), ('b_final', B / 'final270')]:
            state = {}
            for filename in ('STATUS.json', 'HEARTBEAT.json', 'RESULT.json', 'DONE.json',
                             'WATCH_STATUS.json', 'INNOVATION_WATCH_STATUS.json', 'SKIPPED.json', 'NO_NEW_MODEL.json'):
                current = small(folder / filename, fields)
                if current is not None:
                    state[filename] = current
            queues[name] = state
        final = small(RECORD / 'PIPELINE_STATUS.json')
        complete = final is not None and final.get('state') == 'complete'
        failures = [str(path) for folder in (IA, IB, IA / 'final_snapshot', IB / 'final_snapshot')
                    for path in folder.glob('*FAILURE*.json')]
        failures += [str(path) for path in (A / 'final270').glob('INNOVATION_FAILURE*.json')]
        atomic_json(RECORD / 'INNOVATION_STAGE_STATUS.json', dict(
            at_beijing=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
            state='complete' if complete else 'original_training_and_registered_inference_queues',
            observer_pid=os.getpid(), queues=queues, new_innovation_failures=failures,
            final_test_selection_frozen=(RECORD / 'FINAL_TEST_SELECTION.json').exists(),
            final_result=final, no_training_or_model_operation_by_observer=True,
            report='INFERENCE_INNOVATION108_RESULT.md',
            note='Actual new forwards and completed studies are listed; CPU analysis/audits may leave GPUs idle; no artificial load'))
        if complete:
            return
        time.sleep(60)


if __name__ == '__main__':
    main()
