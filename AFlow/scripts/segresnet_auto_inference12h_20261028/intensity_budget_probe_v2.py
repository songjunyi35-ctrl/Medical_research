from scripts.segresnet_auto_inference12h_20261028.common import *
from scripts.segresnet_auto_inference12h_20261028.guard_v2 import stable_size

def main():
    one=sum(int(np.prod(g['extent'])) for g in read(CACHE/'index.json')['geometry'].values())*4
    current=stable_size(TMP)+stable_size(CACHE)+read(R/'MAIN_SCRATCH_BUDGET.json')['bytes']
    extra=28*(one+8*270*65536*2*4+8*270*6375*6)+8*2**30
    result=dict(at=now(),current_scratch_bytes=current,worst_case_extra_bytes=extra,worst_case_total_bytes=current+extra,max_bytes=300*2**30,all28_outputs_retained_assumption=True,additional_phase_headroom_gib=8,persistent_bytes=stable_size(R),persistent_free_bytes=shutil.disk_usage(ROOT).free,source_sha256=sha(Path(__file__)))
    result['state']='passed' if current+extra<300*2**30 and result['persistent_bytes']+256*2**20<2*2**30 and result['persistent_free_bytes']>20*2**30 else 'insufficient_worst_case_headroom'
    write(R/'INTENSITY_BUDGET_GATE_POSTPUBLICATION.json',result);print(result)
if __name__=='__main__':
    import numpy as np
    main()
