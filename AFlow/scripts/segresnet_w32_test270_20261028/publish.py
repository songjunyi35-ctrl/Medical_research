"""Raw-only statistics, independent saved-mask audit, and leaderboard publication."""
import copy,fcntl,re,subprocess,shutil
import numpy as np
from scripts.segresnet_w32_test270_20261028.common import *
from scripts.train_t1ce_wt_capg3d import summarize_cases

def references():
 paths={
  'multiscale166':ROOT/'archive/research_records/2026-09-29_segresnet_multiscale_e166_test270/RAW_TEST270.json',
  'native173':ROOT/'archive/research_records/2026-09-20_segresnet_native_res_e173_test270/RAW_TEST270.json',
  'patch90':ROOT/'archive/research_records/2026-09-21_t1ce_test270_10h/reports/panel_02.json',
  'segresnet_fusion':ROOT/'archive/research_records/2026-09-21_t1ce_test270_10h/reports/panel_10.json',
  'nnunet_single525':ROOT/'archive/research_records/2026-10-13_nnunet_resenc_reference/reports/nnunet_resenc_20261013_single.json'}
 refs={}
 for k,p in paths.items():
  d=read(p)
  if k=='native173':cases=d['candidate']['summary']['per_case']
  elif k=='multiscale166':cases=d['summary']['per_case']
  else:cases=d['per_case']
  validate_rows(cases);refs[k]={'path':str(p.relative_to(ROOT)),'sha256':sha(p),'per_case':cases}
 return refs

def block(file,text):
 p=ROOT/file;s=p.read_text();a='<!-- WIDTH32_TEST270_20261028 -->';b='<!-- END_WIDTH32_TEST270_20261028 -->';t=a+'\n'+text+'\n'+b
 if a in s:s=s[:s.index(a)]+t+s[s.index(b)+len(b):]
 else:
  i=s.index('\n');s=s[:i+1]+'\n'+t+'\n'+s[i+1:]
 p.write_text(s)
def progress(state,description):
 rel=str(R.relative_to(ROOT))
 block('AGENTS.md',f'## 2026-10-28 最新用户授权：32通道171轮一次270测试\n\n用户明确“去测评一下吧，2张A10040G，速度拉到最快”，覆盖本候选此前无自动test边界，仅已选w32第171轮SHA02a60b2d…一次270/119评测，不恢复训练、不新增16模型测试。{description} worker1152470双A10040GB，固定单T1ce、native1mm/160×208×168去填充、阈值0.5、bf16、主输出每例一次，两卡偶/奇槽位各135例，batch只按合成硬件吞吐选。每例N4一次同时生成两栅格，四载荷SHA须匹配历史。MODEL_PREACCESS独占保护，失败不自动重做前向，raw完整只恢复统计/出版。旧验证门失败保留，test-tuned内部复用探索，无本次test调参或新outer/confirmation。持久新增<=2GiB/剩余>=20GiB，大缓存只worker/tmp，4小时预算。当前真实状态与结果在{rel}/。')
 block('AFLOW_BRATS_HANDOFF_CURRENT.md',f'**2026-10-28固定w32第171轮270测试：** {description} 新用户授权仅该模型；原训练验证门未过，单T1ce/同原生栅格/阈值0.5，不重新选epoch或融合。无新outer/confirmation。[方案]({rel}/IDEA.md) · [状态]({rel}/PIPELINE_STATUS.json)。')
 block('AFLOW_BRATS_METHOD_CATALOG.md',f'## 2026-10-28：32通道171轮固定270评测｜{state}\n\n**一句话：** 固定验证选出的宽模型，让两卡分担病例测实际效果。\n**出发点：** 108验证0.753276，未过原明确门，用户希望检查270表现。\n**具体做法：** 同历史原生缓存、单T1ce、概率0.5、bf16一次整脑输出，270例各一次。\n**固定与改动：** 仅新增该固定模型测试；参考报告复用，不按测试结果选择参数。\n**结果：** {description}\n**成本与决定：** 双卡合成选batch，每例N4一次；缓存和纯推理耗时分开报告。复用内部探索，不自动续训/重测。[核对材料]({rel}/IDEA.md)。')
 block('AFLOW_BRATS_RESULTS.md',f'## 2026-10-28：用户授权32通道171轮270评测\n\n问题：宽模型验证点估计略高但不确定。做法：唯一171模型、原270/119、同原生栅格、双卡分病例推理。{description} 原验证门未过事实保留，当前不按测试调参，无新outer/confirmation。[记录]({rel}/IDEA.md)。')
 block('AFLOW_BRATS_WEEKLY_REPORT.md',f'## 2026-10-28：宽模型固定测试\n\n{description} 仅108验证选定的32通道171轮模型，270扫描/119患者内部复用探索；固定阈值/预处理，无本次测试调参，不称独立泛化。原验证门未过。[材料]({rel}/IDEA.md)。')
def checks():
 results={}
 for name in ['verify_aflow_brats_doc_layout.py','verify_aflow_brats_method_catalog.py','verify_aflow_brats_leaderboard.py','verify_aflow_root_artifacts.py']:
  p=subprocess.run([str(PY),str(ROOT/'scripts'/name)],cwd=ROOT,env=env(),text=True,capture_output=True);results[name]={'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
 write(R/'DOC_CHECKS.json',{'at':now(),'checks':results})
 if any(v['returncode'] for v in results.values()):raise RuntimeError('documentation verification failed')

def merge_raw():
 ranks=[read(R/f'RAW_RANK_{i}.json') for i in range(2)];cases={d['case_id']:d for rank in ranks for d in rank['per_case']}
 assert len(cases)==270 and sum(len(x['per_case']) for x in ranks)==270
 ordered=[cases[d['case_id']] for d in rows('test')];validate_rows(ordered)
 raw={'at':now(),'status':'complete_candidate_inference','method':ID,'epoch':EPOCH,'checkpoint_sha256':CHECKPOINT_SHA,'summary':summarize_cases(ordered),'predictions':[p for rank in ranks for p in rank['predictions']],'test_access':True,'test_tuned_development':True,'current_test_retuning':False,'new_outer_confirmation_access':False,'grid':'native1mm crop; padding excluded','runtime':{'inference_seconds_parallel_max':max(v['seconds'] for v in ranks),'sum_rank_seconds':sum(v['seconds'] for v in ranks),'rank_runtime':[{k:v[k] for k in ['rank','seconds','batch_forward_calls','peak_allocated_gib','peak_reserved_gib']} for v in ranks],'evaluation_gpus':2,'batch_per_gpu':read(R/'BATCH_DECISION.json')['batch'],'model_forwards_per_scan':1,'input_voxel_equivalents_128cubed':float(np.prod(CANVAS)/128**3),'not_FLOPs_ratio':True,'timing_includes':'tensor loading, forward, metrics, mask compression and journal persistence; excludes cache construction and model load'}}
 write(R/'RAW_TEST270.json',raw);return raw

def audit_masks(raw):
 idx=read(CACHE/'index.json');labels=np.memmap(CACHE/'labels.u8',dtype=np.uint8,mode='r',shape=(270,*CANVAS));lookup={p['case_id']:p for p in raw['predictions']}
 for d in raw['summary']['per_case']:
  p=lookup[d['case_id']];assert sha(ROOT/p['path'])==p['sha256'];g=idx['geometry'][d['case_id']]
  with np.load(ROOT/p['path']) as saved:
   shape=tuple(int(v) for v in saved['shape']);assert list(shape)==g['extent'];pred=np.unpackbits(saved['mask_bits'])[:int(np.prod(shape))].reshape(shape)>0;point=tuple(int(v) for v in saved['point'])
  sl=tuple(slice(o,o+e) for o,e in zip(g['canvas_offset'],g['extent']));target=np.asarray(labels[(d['source_slot'],*sl)])>0
  tp=int((pred&target).sum());fp=int((pred&~target).sum());fn=int((~pred&target).sum())
  assert (tp,fp,fn)==(d['tp'],d['fp'],d['fn']) and bool(target[point])==d['automatic_point_hits_gt']
  value=float(np.float32(2*tp)/np.float32(2*tp+fp+fn)) if 2*tp+fp+fn else 1.
  assert value==d['locator_dsc'] and list(point)==d['automatic_point']
 return {'at':now(),'scans':270,'patient_model_forward':False,'saved_masks_sha_and_TP_FP_FN_point_verified':True}

def board_update(result):
 rel=str(R.relative_to(ROOT));key='test270_tuned_native_rank_by_scan'
 with (R/'board.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);path=ROOT/'AFLOW_BRATS_LEADERBOARD.json';board=read(path);before=copy.deepcopy(board)
  row={'method':ID,'scan_dice':result['scan_dice'],'subject_dice':result['subject_dice'],'point_hit':result['point_hit'],'point_hits':result['point_hits'],'point_denominator':270,'cohort':{'scans':270,'subjects':119},'checkpoint_epoch':EPOCH,'checkpoint_path':rel+'/checkpoint_epoch171.pth','checkpoint_sha256':CHECKPOINT_SHA,'grid':'native1mm crop;160x208x168 canvas;padding excluded','score_grid':'native','threshold':.5,'threshold_space':'sigmoid_probability','threshold_probability':.5,'raw_logit_threshold':0.,'threshold_raw_logit':0.,'comparison_operator':'>','tta':False,'postprocessing':False,'locator_forwards':1,'ensemble_size':1,'forward_equivalents':float(np.prod(CANVAS)/128**3),'forward_cost_note':'input-voxel equivalence only, width32 costs more than width16; not FLOPs','report_path':rel+'/RAW_TEST270.json','report_sha256':sha(R/'RAW_TEST270.json'),'result_path':rel+'/RESULT.json','result_sha256':sha(R/'RESULT.json'),'evidence_status':'test-tuned development; fixed validation-selected one-shot; reused internal exploratory; no current test retuning','test_tuned':True,'current_test_retuning':False,'validation_gate_passed':False,'scan_ci95':result['scan_ci95'],'subject_ci95':result['subject_ci95'],'role':'candidate','logit_formula':'single main-output whole-brain model','input_modality':'T1ce','input_mode':'native','uses_GT_for_localization':False,'model_precision':'bfloat16','fusion_precision':'float32','parameters':PARAMETERS,'paired_comparisons':result['comparisons'],'training_scope':{'train_scans':972,'train_subjects':444,'validation_scans':108,'validation_subjects':50,'epochs':200,'optimizer_updates':24400,'initialization':'random','public_pretraining_scans':0},'new_outer_confirmation_access':False}
  ranked=[v for v in board[key] if v['method']!=ID]+[row];ranked.sort(key=lambda v:(-v['scan_dice'],v['forward_equivalents'],v['method']))
  for i,v in enumerate(ranked,1):v['rank']=i
  for old in before[key]:
   new=next(v for v in ranked if v['method']==old['method']);assert {k:v for k,v in old.items() if k!='rank'}=={k:v for k,v in new.items() if k!='rank'}
  board[key]=ranked;board['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
  board['latest_segresnet_w32_e171_test270_20261028']={'candidate':ID,'fixed_epoch':171,'validation_gate_passed':False,'current_test_retuning':False,'rank':row['rank']}
  for k,v in before.items():
   if k not in (key,'updated_utc','latest_segresnet_w32_e171_test270_20261028'):assert board[k]==v
  write(path,board)
  md=ROOT/'AFLOW_BRATS_LEADERBOARD.md';s=md.read_text()
  table='\n'.join(f"| {v['rank']} | {v['method']} | {v['scan_dice']:.9f} | {v['subject_dice']:.9f} | {v['point_hits']}/270 | {v['forward_equivalents']:.3f}× | [报告]({v['report_path']}) |" for v in ranked)
  body='<!-- TEST270_TUNED_NATIVE -->\n## 270例定向开发：原始栅格，test-tuned，独立分榜\n\n本榜为复用内部探索，非独立泛化。新32通道171仅按108验证选定，本次固定阈值0.5且不按测试调参。统一原始1mm裁剪评分、排除填充；不同数值/推理模式见各报告。前向等价量只按输入体素折算，不代表不同宽度/架构的FLOPs或实测耗时。\n\n| 排名 | 方法/快照 | Scan Dice | Subject Dice | Point | 128³前向等价量 | 证据 |\n|---:|---|---:|---:|---:|---:|---|\n'+table+'\n<!-- END_TEST270_TUNED_NATIVE -->'
  assert s.count('<!-- TEST270_TUNED_NATIVE -->')==1
  s=re.sub(r'<!-- TEST270_TUNED_NATIVE -->.*?<!-- END_TEST270_TUNED_NATIVE -->',lambda _:body,s,flags=re.S)
  s=re.sub(r'更新时间：[^\n]+','更新时间：2026-10-28 UTC',s,count=1);md.write_text(s)
  write(R/'LEADERBOARD_UPDATE.json',{'at':now(),'new_method':ID,'new_rank':row['rank'],'all_historical_rows_preserved':True,'all_other_rankings_unchanged':True})

def main():
 verify();raw=read(R/'RAW_TEST270.json') if (R/'RAW_TEST270.json').exists() else merge_raw();cases=raw['summary']['per_case'];validate_rows(cases)
 audit=audit_masks(raw);write(R/'INDEPENDENT_MASK_AUDIT.json',audit)
 refs=references();metrics=summary_ci(cases);metrics['point_hits']=sum(v['automatic_point_hits_gt'] for v in cases)
 contrasts={k:paired(cases,v['per_case']) for k,v in refs.items()}
 for v in contrasts.values():v['bootstrap']='10000 patient-cluster paired percentile seed20260918;95% primary descriptive;97.5% sensitivity, not correction for all comparisons'
 result={'at':now(),'status':'complete_test270','method':ID,'epoch':EPOCH,'checkpoint_sha256':CHECKPOINT_SHA,**metrics,'comparisons':contrasts,'reference_summaries':{k:summary_ci(v['per_case']) for k,v in refs.items()},'reference_reports':{k:{s:v[s] for s in ['path','sha256']} for k,v in refs.items()},'runtime':raw['runtime'],'cache':read(R/'CACHE_READY.json'),'grid':raw['grid'],'validation_gate_passed':False,'test_access':True,'test_tuned_development':True,'current_test_retuning':False,'not_pristine_blind':True,'not_external':True,'new_outer_confirmation_access':False,'numeric_comparability_note':'same native payload and score;bf16 deterministic, batch selected synthetically on2xA10040GB; historical batch/hardware precision differences retained','evidence':{str(p.relative_to(ROOT)):sha(p) for p in [CHECKPOINT,R/'RAW_TEST270.json',R/'MODEL_PREACCESS.json',R/'CACHE_READY.json',R/'BATCH_DECISION.json',R/'EXECUTION_FREEZE.json',R/'INDEPENDENT_MASK_AUDIT.json']}}
 write(R/'RESULT.json',result)
 docs=['AGENTS.md','AFLOW_BRATS_HANDOFF_CURRENT.md','AFLOW_BRATS_METHOD_CATALOG.md','AFLOW_BRATS_RESULTS.md','AFLOW_BRATS_WEEKLY_REPORT.md','AFLOW_BRATS_LEADERBOARD.md','AFLOW_BRATS_LEADERBOARD.json']
 backup=R/'docs_before_publication';backup.mkdir(exist_ok=True)
 for f in docs:
  if not (backup/f).exists():shutil.copy2(ROOT/f,backup/f)
 board_update(result)
 lines=['# 32通道SegResNet第171轮固定270评测','','270扫描/119患者，原始1mm裁剪栅格去填充。',f"Scan Dice **{metrics['scan_dice']:.9f}**，95%CI{metrics['scan_ci95']}；Subject **{metrics['subject_dice']:.9f}**，95%CI{metrics['subject_ci95']}；Point **{metrics['point_hits']}/270**。",'','| 历史参考 | 参考Scan | 本次−参考 | 配对95%区间 | Point差 |','|---|---:|---:|---|---:|']
 for k,v in contrasts.items():lines.append(f"| {k} | {result['reference_summaries'][k]['scan_dice']:.9f} | {v['scan_delta']:+.9f} | {v['scan_delta_ci95']} | {v['point_hits_delta']:+d} |")
 primary=contrasts['multiscale166'];verdict='复用队列配对区间支持探索性正差，但不构成独立泛化确认' if primary['scan_delta_ci95'][0]>0 else ('复用队列配对区间为负，本次固定模型较该参考差' if primary['scan_delta_ci95'][1]<0 else '配对区间跨0，没有确认稳定增益或等效')
 lines+=['',f'主要同一次整脑参考为历史多尺度166；{verdict}。其他参考的推理成本/初始化/预算不同，只作描述性比较。',f"双卡各135例，batch{raw['runtime']['batch_per_gpu']}/卡，包含读入、推理、评分、预测压缩及journal的并行较慢rank耗时{raw['runtime']['inference_seconds_parallel_max']:.2f}秒；缓存另计{result['cache']['seconds']:.2f}秒。每例一次整脑，无TTA/融合/后处理，阈值0.5固定。",'单T1ce、原972/108/270与患者隔离；新模型由108选171，未通过原验证收益门，现为用户另行授权固定探索性测试，不改写旧门。历史测试已复用，全部仍test-tuned开发；本次无测试参数搜索或新outer/confirmation，不称独立盲测。','当前候选270前向各例一次完成，保存预测掩膜逐SHA及TP/FP/FN/Point独立重算通过，无参考模型额外前向。缓存四个载荷SHA均与历史完全一致；每例只算一次N4，同时导出两栅格。', '原生框外历史4个阳性体素沿用旧评分，不以标签扩框。batch/硬件数值差不能通过重复参考前向消除；输入体素等价量不是不同宽度的FLOPs比。','所有正负结果保留，结束后不自动重测、换epoch/阈值/融合或续训。','[逐例原始结果](RAW_TEST270.json) · [机器结果](RESULT.json) · [模型](checkpoint_epoch171.pth) · [掩膜复核](INDEPENDENT_MASK_AUDIT.json)']
 (R/'RESULT.md').write_text('\n'.join(lines)+'\n')
 desc=f"固定171轮270评测完成：Scan{metrics['scan_dice']:.6f}、Subject{metrics['subject_dice']:.6f}、Point{metrics['point_hits']}/270；对多尺度166差{primary['scan_delta']:+.6f}，95%CI{primary['scan_delta_ci95']}。{verdict}。[结果]({R.relative_to(ROOT)}/RESULT.md)。"
 progress('complete',desc)
 write(TRAIN/'TEST_ACCESS_ADDENDUM_20261028.json',{'at':now(),'new_stage_report':str((R/'RESULT.json').relative_to(ROOT)),'model_epoch':171,'checkpoint_sha256':CHECKPOINT_SHA,'test_access_completed':True,'test_scans':270,'test_subjects':119,'original_training_false_access_flags_describe_training_stage':True,'original_training_result_unchanged':True,'new_outer_confirmation_access':False})
 checks()
 # Audit both leaderboard serialization and basic summaries independently.
 scan=float(np.mean([v['locator_dsc'] for v in cases]));subject=float(np.mean([np.mean([v['locator_dsc'] for v in cases if v['subject_id']==sid]) for sid in sorted({v['subject_id'] for v in cases})]))
 assert abs(scan-metrics['scan_dice'])<1e-12 and abs(subject-metrics['subject_dice'])<1e-12
 board=read(ROOT/'AFLOW_BRATS_LEADERBOARD.json');ranked=board['test270_tuned_native_rank_by_scan'];assert [v['rank'] for v in ranked]==list(range(1,len(ranked)+1))
 md=(ROOT/'AFLOW_BRATS_LEADERBOARD.md').read_text()
 for v in ranked:
  line=f"| {v['rank']} | {v['method']} | {v['scan_dice']:.9f} | {v['subject_dice']:.9f} | {v['point_hits']}/270 | {v['forward_equivalents']:.3f}× | [报告]({v['report_path']}) |"
  assert md.count(line)==1 and sha(ROOT/v['report_path'])==v['report_sha256']
 write(R/'COMPLETION_AUDIT.json',{'at':now(),'status':'passed','result_sha256':sha(R/'RESULT.json'),'raw_sha256':sha(R/'RAW_TEST270.json'),'checkpoint_sha256':CHECKPOINT_SHA,'saved_masks_all270_independently_verified':True,'both_leaderboards_checked':True,'checks_passed':True,'new_outer_confirmation_access':False})
 status('complete',running=False,scan_dice=metrics['scan_dice'],subject_dice=metrics['subject_dice'],point_hits=metrics['point_hits'],test_access_completed=True,new_outer_confirmation_access=False)
if __name__=='__main__':main()
