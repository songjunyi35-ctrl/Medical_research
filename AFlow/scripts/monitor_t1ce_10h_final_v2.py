"""Preserve the single-worker visibility incident in final ten-hour publication."""
import sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import scripts.monitor_t1ce_10h_final
from scripts import complete_t1ce_10h as original
from scripts.t1ce_10h_common import *
publish=original.publish;evidence=original.evidence

def publish_with_availability(final=False):
 publish(final);p=R/'RESULT.json';d=read(p);incident=read(R/'WORKER_VISIBILITY_INCIDENT_001.json')
 d['worker_visibility_incidents']=[incident]
 d['worker_storage_observation_scope']={'dual':'latest worker heartbeat; independently observed available','single':'last confirmed heartbeat before endpoint loss, not a current storage observation'}
 d['ten_hour_scope']='overall research supervision window; single-worker heartbeat ceased after19:06 UTC and its active platform handle disappeared; do not claim every worker stayed online for10h'
 d['single_cache_redundancy_verification']=read(R/'SINGLE_CACHE_REDUNDANCY_VERIFIED.json');write(p,d)
 md=R/'RESULT.md';s=md.read_text();a='<!-- WORKER_AVAILABILITY -->';b='<!-- END_WORKER_AVAILABILITY -->'
 body=a+'\n## 单卡节点可见性事件\n\n单卡1105022最后心跳为19:06:25 UTC，19:08起SSH连接失败；复查平台活动清单已无该worker，仅剩1104883。不能把连接失败本身冒充PID退出证明，也不能声称单卡全程在线。B训练在13:52已停止，C未启动；全部持久模型和报告核验完整。B的270份临时预测与剩余双卡上的ref_pretrained179逐文件SHA完全一致，已现场核验，无须重跑模型。\n\n主节点与双卡监督继续覆盖原定23:04:52 UTC截止，资源监控持续记录缺失端点；没有启动替代GPU、恢复训练或新增评估。10小时指整体科研监督窗口，单卡可见性确有中断；其磁盘数字仅代表最后可观测时刻。\n\n[节点事件](WORKER_VISIBILITY_INCIDENT_001.json) · [平台原始清单](PLATFORM_INVENTORY_AFTER_SINGLE_LOSS.txt) · [缓存冗余核验](SINGLE_CACHE_REDUNDANCY_VERIFIED.json) · [监控修订](MONITORING_AMENDMENT_004.json)\n'+b
 if a in s:s=s[:s.index(a)]+body+s[s.index(b)+len(b):]
 else:s+='\n\n'+body+'\n'
 md.write_text(s)

def evidence_with_availability():
 evidence();p=R/'EVIDENCE_SHA256.json';d=read(p)
 for name in ['WORKER_VISIBILITY_INCIDENT_001.json','PLATFORM_INVENTORY_AFTER_SINGLE_LOSS.txt','SINGLE_CACHE_REDUNDANCY_VERIFIED.json','MONITORING_AMENDMENT_004.json','OUTPUT_CONTRASTS.json','INDEPENDENT_METRIC_AUDIT_PREACCESS.json','COMPLETION_READINESS_AUDIT.json']:
  q=R/name
  if q.exists():d['files'][str(q.relative_to(ROOT))]=sha(q)
 write(p,d)
original.publish=publish_with_availability;original.evidence=evidence_with_availability

if __name__=='__main__':
 try:
  publish_with_availability();original.access_summary();original.checks();evidence_with_availability();original.main()
 except BaseException:write(R/f'FINAL_MONITOR_V2_FAILURE_{int(time.time())}.json',{'utc':now(),'traceback':traceback.format_exc()});raise
