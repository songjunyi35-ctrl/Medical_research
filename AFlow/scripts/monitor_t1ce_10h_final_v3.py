"""Final window continues from master after both completed GPU workers disappear."""
import sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import scripts.monitor_t1ce_10h_final_v2
from scripts import complete_t1ce_10h as original
from scripts.t1ce_10h_common import *
publish=original.publish;evidence=original.evidence

def publish_with_both_incidents(final=False):
 publish(final);p=R/'RESULT.json';d=read(p)
 d['worker_visibility_incidents']=[read(R/'WORKER_VISIBILITY_INCIDENT_001.json'),read(R/'WORKER_VISIBILITY_INCIDENT_002.json')]
 d['worker_storage_observation_scope']={'dual':'last confirmed heartbeat21:06:12 UTC; endpoint subsequently unavailable, not current storage','single':'last confirmed heartbeat19:06:25 UTC; endpoint subsequently unavailable, not current storage'}
 d['ten_hour_scope']='overall master-managed research supervision window; GPU worker visibility ended after completed computations, at19:08(single) and21:06(dual); not10h uninterrupted GPU-worker uptime or10h training'
 d['persistent_artifacts_after_both_workers_unavailable']=read(R/'PERSISTENT_ARTIFACTS_AFTER_DUAL_LOSS.json');write(p,d)
 md=R/'RESULT.md';s=md.read_text()
 s=s.replace('主节点与双卡监督继续覆盖原定23:04:52 UTC截止，资源监控持续记录缺失端点；','单卡事件后主节点与双卡曾继续监督；随后双卡也失去可见性，见下节。主节点仍持续记录端点状态至原截止；')
 a='<!-- DUAL_WORKER_AVAILABILITY -->';b='<!-- END_DUAL_WORKER_AVAILABILITY -->'
 body=a+'\n## 双卡节点可见性事件与剩余监督\n\n双卡1104883最后心跳为21:06:12 UTC，21:06:43起SSH连接失败；重复查询的平台活动worker清单为空。其远程PID当前不可直接观察，不能仅凭SSH错误宣称具体进程退出状态。A训练15:17已完成，最终面板、部署检查和拟合诊断也已完成；3个部署权重、21份270原始报告及3份固定数据清单在节点不可访问后再次核验完整、SHA一致。\n\n两节点临时盘此后不可访问；不声称临时预测或影像缓存永久可用。已有缓存SHA、源码、模型和报告保存在持久目录，没有因为节点离线重新构建或推理。余下监督由主节点持续监测持久化终态、端点状态和磁盘直到23:04:52 UTC；没有启动替代GPU或恢复研究。10小时是整体科研监督窗口，不是每个worker全程在线或10小时满载训练。\n\n[双卡事件](WORKER_VISIBILITY_INCIDENT_002.json) · [空活动清单](PLATFORM_INVENTORY_AFTER_DUAL_LOSS.txt) · [持久产物再核验](PERSISTENT_ARTIFACTS_AFTER_DUAL_LOSS.json)\n'+b
 if a in s:s=s[:s.index(a)]+body+s[s.index(b)+len(b):]
 else:s+='\n\n'+body+'\n'
 md.write_text(s)

def evidence_with_both_incidents():
 evidence();p=R/'EVIDENCE_SHA256.json';d=read(p)
 for name in ['WORKER_VISIBILITY_INCIDENT_002.json','PLATFORM_INVENTORY_AFTER_DUAL_LOSS.txt','PERSISTENT_ARTIFACTS_AFTER_DUAL_LOSS.json','MONITORING_AMENDMENT_005.json']:
  q=R/name
  if q.exists():d['files'][str(q.relative_to(ROOT))]=sha(q)
 write(p,d)
original.publish=publish_with_both_incidents;original.evidence=evidence_with_both_incidents

if __name__=='__main__':
 try:
  publish_with_both_incidents();original.access_summary();original.checks();evidence_with_both_incidents();original.main()
 except BaseException:write(R/f'FINAL_MONITOR_V3_FAILURE_{int(time.time())}.json',{'utc':now(),'traceback':traceback.format_exc()});raise
