"""Register a finite image-intensity factorial after geometry/space gates."""
import copy
from scripts.segresnet_auto_inference12h_20261028.common import *


def main():
    gate=read(R/'INTENSITY_SYNTHETIC_GATE.json');budget=read(R/'INTENSITY_BUDGET_GATE.json')
    assert gate['state']==budget['state']=='passed'
    specs=[c['spec'] for c in gate['checks']];assert len(specs)==28
    producer_template=read(R/'queue_v2/r1_0041_resize_near.json')
    analysis_template=read(R/'analysis_queue_aux_v1/a2_0103_phase256_fusions.json')
    calibration_template=read(R/'cpu_queue_v2/a2_1003_phase256_calibration.json')
    registered=[]
    for i,spec in enumerate(specs):
        lane=0 if i%2==0 else 1;queue='queue_v2' if lane==0 else 'queue_aux_v1'
        phase='r1_0050_polyphase512_flip7' if lane==0 else 'a2_0000_polyphase512'
        number=60+i//2 if lane==0 else 80+i//2
        gid=f'r1_{number:04}_intensity{i:02}';aid=f'g1_{i:04}_intensity{i:02}_fusions';cid=f'g1_{1000+i:04}_intensity{i:02}_calibration'
        p=copy.deepcopy(producer_template);p.update(id=gid,spec=spec,depends_on=[phase,'r1_0000_core_mirror8'],analysis_dependencies=[],seed=20261028)
        p['source_sha256'].update({str(Path(__file__).relative_to(ROOT)):sha(Path(__file__)),str((ROOT/'scripts/segresnet_auto_inference12h_20261028/intensity_gate_v1.py').relative_to(ROOT)):sha(ROOT/'scripts/segresnet_auto_inference12h_20261028/intensity_gate_v1.py')})
        if lane==1:p['source_sha256']['scripts/segresnet_auto_inference12h_20261028/gpu_aux_v1.py']=sha(ROOT/'scripts/segresnet_auto_inference12h_20261028/gpu_aux_v1.py')
        a=copy.deepcopy(analysis_template);a.update(id=aid,handler='phase_fusions_fast_v1',group_source=gid,source_queue=queue,depends_on=[gid,'r1_0000_core_mirror8'])
        a.pop('control_amendment',None);a['source_sha256']['scripts/segresnet_auto_inference12h_20261028/phase_fusions_fast_v1.py']=sha(ROOT/'scripts/segresnet_auto_inference12h_20261028/phase_fusions_fast_v1.py')
        a['source_sha256']['scripts/segresnet_auto_inference12h_20261028/guard_cached_v1.py']=sha(ROOT/'scripts/segresnet_auto_inference12h_20261028/guard_cached_v1.py')
        a['control_amendment']='same65 fusion formulas as phase_fusions_v1 byte-identical after cached-guard import/name reversal; no mathematical change'
        c=copy.deepcopy(calibration_template);c.update(id=cid,source_analysis=aid,analysis_dependencies=[aid])
        # Descriptors are available before SOURCE is read by the group evaluator.
        source=R/'jobs'/gid/'SOURCE.json';assert not source.exists();write(source,p)
        for q,j in [('cpu_queue_v2',c),('analysis_queue_v2',a),(queue,p)]:
            path=R/q/(j['id']+'.json');assert not path.exists();write(path,j)
        registered.append(dict(render=gid,analysis=aid,calibration=cid,physical_gpu=lane,spec=spec))
    write(R/'INTENSITY_REGISTRATION.json',dict(at=now(),state='registered_pending_phase_dependency',groups=registered,network_case_forwards_expected=28*8*270,
        threshold_configurations=28*65*39,calibration_configurations=28*8*6375,neural_checkpoint_sha256=CHECKPOINT_SHA,synthetic_gate_sha256=sha(R/'INTENSITY_SYNTHETIC_GATE.json'),budget_gate_sha256=sha(R/'INTENSITY_BUDGET_GATE.json'),test_tuned=True,training=False))
    print(dict(groups=28,per_gpu=14,network_case_forwards_expected=60480,starts='after existing512 parents, without interrupting'))

if __name__=='__main__':main()
