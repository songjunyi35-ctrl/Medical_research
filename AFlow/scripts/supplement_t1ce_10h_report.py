"""Bind completed diagnostics, numeric scope and source attribution to final reports."""
import sys,re
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.t1ce_10h_common import *

def supplement():
 p=R/'RESULT.json';d=read(p);fit=read(R/'TRAIN_FIT_DIAGNOSTIC.json');bench=read(R/'DEPLOYMENT_BENCHMARK.json');reload=read(R/'DEPLOYMENT_CHECK.json')
 d['source_attribution']='overall winner uses existing native173 and combination166 at threshold0.4; new A training is the best single-model8-patch result, not a component of the overall winner'
 d['clean_fit_diagnostic']={k:v for k,v in fit.items() if k not in ['configs']}
 d['deployment_benchmark']=bench;d['deployment_reload']=reload
 d['training_gpu_hours_wall_estimate']=sum(s.get('elapsed',0)*(2 if c=='A' else 1)/3600 for c,s in d['training'].items())
 d['disk_observation_scope']='disk field above describes master; worker scratch shown separately'
 d['worker_storage']={s:read(R/f'HEARTBEAT_{s}.json')['resources'] for s in ['dual','single']}
 d['candidate_C_access_scope']={'target_diagnostic_used_for_start_decision':True,'candidate_model_test_forward':False}
 d['current_elapsed_wall_seconds']=time.time()-timestamp(P['start_utc'])
 refs=['TRAIN_FIT_DIAGNOSTIC.json','DEPLOYMENT_BENCHMARK.json','DEPLOYMENT_CHECK.json','FINAL_PANEL_CHECKS.json','EXPORT_REPAIR_RECEIPT.json','ACCESS_BOUNDARY_CLARIFICATION.json','EVALUATION_CONTRACT.json','NUMERIC_COMPARABILITY.json','CHECKPOINT_IDENTITIES.json']
 d['supplementary_evidence_sha256']={name:sha(R/name) for name in refs};write(p,d)
 text=(R/'RESULT.md').read_text();a='<!-- COMPLETION_SUPPLEMENT -->';b='<!-- END_COMPLETION_SUPPLEMENT -->'
 tr=fit['train']['selected_workflow']['scan_dice'];va=fit['validation']['selected_workflow']['scan_dice'];ratio=bench['results']['overall'][0]['measured_gpu_time_ratio_vs128'];single_ratio=bench['results']['best_single'][0]['measured_gpu_time_ratio_vs128']
 paragraph=f'''{a}
## 收益来源与诊断

总体0.775891来自**已有native173＋组合166的等权logits融合及统一概率阈值0.4**，不是新训练A带来的总体收益。A额外训练200轮后仍选新增第90轮，最佳单模型0.772830，需要8块前向；总体方案2次native前向约5.332个128³等价量。[输出方法完整解释](output_panel/RESULT.md)。

无增强、同训练/验证原生栅格诊断：最终流程训练972/444 Scan {tr:.6f}、验证108/50 Scan {va:.6f}；纯193为0.809540/0.742672，native173为0.810327/0.746099。训练病例本身也存在明显剩余误差；这些数据不能单独证明具体过拟合原因、标签病因或T1ce性能上限。诊断在最终配置固定后执行，没有回流选择，也没有新270访问。[诊断](TRAIN_FIT_DIAGNOSTIC.md)。

暖启动合成GPU基准：总体batch1约{bench['results']['overall'][0]['seconds_per_scan']:.3f}秒/例、相对128³约{ratio:.2f}倍；最佳单模型约{bench['results']['best_single'][0]['seconds_per_scan']:.3f}秒/例、约{single_ratio:.2f}倍。仅测模型/融合/阈值，排除N4、磁盘、启动和权重加载；不是临床端到端耗时。[测量](DEPLOYMENT_BENCHMARK.json)。训练进程墙钟折算约{d['training_gpu_hours_wall_estimate']:.3f} GPU小时，不把10小时监督窗口写成10小时满负载训练。

混合精度下batch1部署与batch2评分不是逐位相同：预设4例重载检查最大掩膜不一致比例{max(x['mask_disagreement_fraction'] for x in reload['cases']):.8f}；榜上成绩对应冻结batch2。与历史参考的微小数值差保留于[数值记录](NUMERIC_COMPARABILITY.json)，旧分数不改写。

C虽然没有模型前向，其是否启动使用了270错误诊断；不能把C的“无候选测试”误读为全流程未看270。早期缓存字节哈希访问与随后native构建访问的边界见[范围补记](ACCESS_BOUNDARY_CLARIFICATION.json)，未倒填日期。部署打包字段错误已经修复，原12项报告和选择完全保留，没有重新评分；见[工程修订003](ENGINEERING_AMENDMENT_003.json)。

[总体部署清单](../../../workspace/BraTS/segresnet_t1ce_wt/test270_10h_20260921/deployment/manifest.json) · [最佳单模型清单](../../../workspace/BraTS/segresnet_t1ce_wt/test270_10h_20260921/deployment/best_single_manifest.json) · [训练曲线](LIVE_CURVES.png)
{b}'''
 if a in text:text=text[:text.index(a)]+paragraph+text[text.index(b)+len(b):]
 else:text+='\n\n'+paragraph+'\n'
 (R/'RESULT.md').write_text(text)
 return d

if __name__=='__main__':supplement()
