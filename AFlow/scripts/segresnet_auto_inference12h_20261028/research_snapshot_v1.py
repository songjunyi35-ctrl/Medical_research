"""Report recorded compute and completed grids without extrapolating gaps."""
import json
from scripts.segresnet_auto_inference12h_20261028.common import R,read,write,now


def main():
    analyses=[]
    for path in sorted((R/'analyses').glob('*/DONE.json')):
        d=read(path)
        if d.get('state')!='complete':continue
        n=d.get('configurations')
        if n is not None:analyses.append(dict(id=path.parent.name,configurations=int(n),best=d.get('best'),source=str(path)))
    jobs=[];parents=set()
    for path in sorted((R/'jobs').glob('*/DONE.json')):
        d=read(path)
        if d.get('parent'):parents.add(d['parent'])
        jobs.append((path.parent.name,d))
    forwards=[]
    for name,d in jobs:
        if name in parents:continue  # Stage increments already include shared prefixes exactly once.
        n=d.get('new_case_forwards_this_stage',d.get('new_case_forwards'))
        if n is not None:forwards.append(dict(job=name,recorded_case_forwards=int(n),snapshot_parent=d.get('parent')))
    verifications=[]
    for path in sorted((R/'live_verification').glob('*/DONE.json')):
        d=read(path);verifications.append(dict(candidate=path.parent.name,model_case_forwards=d['model_case_forwards'],seconds=d['seconds'],mask_xor=d['mask_different_voxels']))
    samples=[]
    for line in (R/'UTILIZATION_V2.jsonl').read_text().splitlines():
        try:
            d=json.loads(line);g={}
            if d.get('exit_code')!=0:continue
            for row in d['gpu'].splitlines():
                cols=[x.strip() for x in row.split(',')];g[int(cols[0])]=float(cols[1])
            if set(g)=={0,1}:samples.append((float(d['unix']),g))
        except (ValueError,KeyError):continue
    samples.sort();coverage=0.;weighted={i:0. for i in range(2)};busy={i:0. for i in range(2)};high={i:0. for i in range(2)};gaps=0.
    for (ta,a),(tb,b) in zip(samples,samples[1:]):
        dt=min(max(tb-ta,0),60.);coverage+=dt;gaps+=max(tb-ta-60.,0)
        for i in range(2):weighted[i]+=a[i]*dt;busy[i]+=(a[i]>0)*dt;high[i]+=(a[i]>=90)*dt
    result=dict(at=now(),state='interim_observed_snapshot',completed_configuration_tuples=sum(x['configurations'] for x in analyses),
       configurations_are='scored global parameter tuples, not independent methods or unique masks',analyses=analyses,
       completed_renderer_case_forwards_lower_bound=sum(x['recorded_case_forwards'] for x in forwards),completed_renderers=forwards,
       deployment_verifications=verifications,compute_limitations='Incomplete live phase work, failed/uncommitted batches and unavailable old-worker work excluded from this lower bound. Synthetic forwards excluded. Shared phase prefixes counted by stage increments, never summed prefix sizes.',
       utilization=dict(source='UTILIZATION_V2.jsonl on worker1153287 only',sample_count=len(samples),coverage_seconds=coverage,unfilled_gaps_seconds=gaps,
          first_unix=samples[0][0] if samples else None,last_unix=samples[-1][0] if samples else None,
          time_weighted_mean_percent={str(i):weighted[i]/coverage if coverage else None for i in range(2)},
          busy_fraction={str(i):busy[i]/coverage if coverage else None for i in range(2)},
          at_least90_fraction={str(i):high[i]/coverage if coverage else None for i in range(2)},
          rule='left-sample utilization weighted to next sample, capped at60 seconds; no extrapolation across missing workers or outside sampled range'),
       engineering_recovery=read(R/'RECOVERY_COUNT.json'),training=False,test_tuned=True)
    write(R/'COMPUTE_SNAPSHOT.json',result)
    print({k:result[k] for k in ['at','completed_configuration_tuples','completed_renderer_case_forwards_lower_bound','utilization']})

if __name__=='__main__':main()
