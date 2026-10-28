"""Lightweight rolling index of existing records; never reads image/mask payloads."""
import json,time
from scripts.segresnet_auto_inference12h_20261028.common import R,read,write,now


def main():
 resource=read(R/'RESOURCE_CURRENT.json');token=resource['token'];deadline=read(R/'AUTHORIZATION.json')['deadline_unix']
 while time.time()<deadline:
  if read(R/'RESOURCE_CURRENT.json')['token']!=token:return
  completed=[];active=[]
  for p in (R/'analyses').glob('*/DONE.json'):
   value=read(p)
   if 'configurations' in value:completed.append(dict(id=p.parent.name,configurations=value['configurations'],best_scan=value.get('best',{}).get('scan_dice')))
  for queue in ['analysis_queue_v2','analysis_queue_aux_v1','cpu_queue_v2','cpu_aux_queue_v1']:
   for path in (R/queue).glob('*.json'):
    job=read(path);folder=R/'analyses'/job['id']
    if (folder/'DONE.json').exists() or not (folder/'PLAN.json').exists():continue
    active.append(dict(id=job['id'],queue=queue,completed_cases=len(list(folder.glob('case[0-9][0-9][0-9].json'))),state='active_or_waiting_final_summary'))
  states={}
  for name in ['PIPELINE_STATUS','ANALYSIS_AUX_STATUS','ANALYSIS_STATUS','CPU_STATUS','AUX_STATUS','DENSE_CRF_STATUS','BRAIN_DOMAIN_STATUS','ROLLING_CLEANUP_STATUS']:
   path=R/(name+'.json')
   if path.exists():states[name]=read(path)
  write(R/'RESEARCH_STATUS.json',dict(at=now(),state='active_dual_gpu_research',worker=resource['worker'],resource_token=token,deadline_beijing=read(R/'AUTHORIZATION.json')['end_beijing'],remaining_seconds=max(0,deadline-time.time()),incumbent=read(R/'INCUMBENT.json'),completed_configuration_tuples=sum(x['configurations'] for x in completed),completed_analyses=completed,active=active,states=states,engineering_recoveries=read(R/'RECOVERY_COUNT.json')['used'],training=False,test_tuned=True,research_continues=True))
  time.sleep(30)
if __name__=='__main__':main()
