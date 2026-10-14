"""Back up exact prior documents, then publish the mentor-facing reading layer."""
from datetime import datetime,timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'archive/reviews/2026-09-14_mentor_facing_cleanup'
DOCS=['README.md','AFLOW_BRATS_WEEKLY_REPORT.md','AFLOW_BRATS_RESULTS.md',
      'AFLOW_BRATS_METHOD_CATALOG.md','AFLOW_BRATS_HANDOFF_CURRENT.md']
HISTORICAL='AFLOW_BRATS_HANDOFF_20260813_SINGLE_REPO.md'
SNAPSHOTS={'README.md':'README_DETAIL.md','AFLOW_BRATS_WEEKLY_REPORT.md':'WEEKLY_REPORT_DETAIL.md',
           'AFLOW_BRATS_RESULTS.md':'RESULTS_DETAIL.md','AFLOW_BRATS_METHOD_CATALOG.md':'METHOD_CATALOG_DETAIL.md',
           'AFLOW_BRATS_HANDOFF_CURRENT.md':'HANDOFF_DETAIL.md',HISTORICAL:'HISTORICAL_HANDOFF_DETAIL.md'}


def sha(data):return hashlib.sha256(data).hexdigest()


def link(path,label):return f'[{label}]({os.path.relpath(ROOT/path,DEST)})'


def relocate_links(text):
    def replace(match):
        target=match.group(2)
        if target.startswith(('http://','https://','#')):return match.group(0)
        bits=target.split('#',1);path=bits[0]
        if not path:return match.group(0)
        suffix=('#'+bits[1]) if len(bits)>1 else ''
        moved=os.path.relpath((ROOT/path).resolve(),DEST)
        return match.group(1)+moved+suffix+')'
    return re.sub(r'(\[[^\]]+\]\()([^)]+)\)',replace,text)


def main():
    if (DEST/'PUBLICATION_RECEIPT.json').exists():raise FileExistsError('reading layer already published')
    originals={name:(ROOT/name).read_bytes() for name in DOCS+[HISTORICAL,'AGENTS.md']}
    board_sha={name:sha((ROOT/name).read_bytes()) for name in ['AFLOW_BRATS_LEADERBOARD.md','AFLOW_BRATS_LEADERBOARD.json']}
    backups=DEST/'originals';backups.mkdir(exist_ok=False)
    receipts=[]
    for name,content in originals.items():
        packed=gzip.compress(content,mtime=0)
        with (backups/(name+'.gz')).open('xb') as f:f.write(packed)
        assert gzip.decompress(packed)==content
        receipts.append({'path':name,'before_sha256':sha(content),'compressed_backup':str((backups/(name+'.gz')).relative_to(ROOT))})
        if name in SNAPSHOTS:
            note=f'# 清理前技术资料：{name}\n\n> 2026-09-14整理前的完整技术快照。所有“当前最好”“运行中”“下一步”均属当时状态，不可据此恢复实验；当前阅读入口请看{link("AFLOW_BRATS_WEEKLY_REPORT.md","导师汇报")}和{link("AFLOW_BRATS_RESULTS.md","日期流水账")}。原字节另有压缩备份；这里仅调整相对链接以便归档后阅读。\n\n'
            with (DEST/SNAPSHOTS[name]).open('x') as f:f.write(note+relocate_links(content.decode()))
    cards=json.loads((DEST/'METHOD_SUMMARIES.json').read_text())
    evidence=['# 导师汇报与日期流水账：原始证据索引','',
        '本页是资料层。正文不再展示完整路径、哈希和运行收据；需要核对数字时再从这里进入。原始报告、失败记录、代码、数据和权重未删除。','',
        '## 8月13日后的原流水账没有丢失','',
        '- [8月13日命名长交接的完整快照](HISTORICAL_HANDOFF_DETAIL.md)：实际追加至8月18日。',
        '- [旧结果账本全文](RESULTS_DETAIL.md)、[旧周报全文](WEEKLY_REPORT_DETAIL.md)：标题日期较早，但正文也有8月18日记录。',
        '- '+link('archive/handoffs_20260819/README.md','8月19日六份续篇及顺序说明')+'。',
        '- 后续实验仍按日期分别保存；下面给出主要过程证据。',
        '- [清理前方法全文](METHOD_CATALOG_DETAIL.md)、[清理前当前交接全文](HANDOFF_DETAIL.md)：所有未压入主汇报的支线、来源审计和旧阶段状态保留。','',
        '## 关键时间线证据','']
    timeline=[
        ('8月13日：师兄路线复现','archive/handoffs_pre_20260813/AFLOW_BRATS_RESULTS.md'),
        ('8月14日：旧修正器搜索与剪枝','archive/research_documents/2026-08-17_root_cleanup/INFERENCE_PRUNING_RESULT_20260814.md'),
        ('8月17日：自动定位测试','archive/research_documents/2026-08-17_root_cleanup/PLAIN_LOCATOR_TEST270_RESULT_20260817.md'),
        ('8月18日：长训练','archive/research_records/2026-08-18_plain_long_schedule/RESULT.md'),
        ('8月19日：模型权重平滑','archive/research_records/2026-08-19_plain_ema/EMA168_TEST270_RESULT.md'),
        ('8月19日：输出融合','archive/research_records/2026-08-19_plain_logit_ensemble/RESULT.md'),
        ('8月19日：蒸馏压缩','archive/research_records/2026-08-19_plain_voxel_kd/RESULT.md'),
        ('8月24日：完整972例从头训练及后续探索测试','archive/research_records/2026-08-24_voxel_kd_fulltrajectory_972/RESULT.md'),
        ('8月25日：多种子融合','archive/research_records/2026-08-25_full972_multiseed_ensemble_distill/RESULT.md'),
        ('8月26日：更宽网络','archive/research_records/2026-08-26_full972_width24_fulltrajectory/RESULT.md'),
        ('8月31日：优化器支线','archive/research_records/2026-08-31_gcsam_full972_stability/RESULT.md'),
        ('9月3日：优化器固定轮次复制','archive/research_records/2026-09-03_gcsam_fixed184_replication/RESULT.md'),
        ('9月4日：八镜像','archive/research_records/2026-09-04_direct_locator_flip8_tta_stability/RESULT.md'),
        ('9月4日：镜像一致训练','archive/research_records/2026-09-04_flip_equivariance_consistency_stability/RESULT.md'),
        ('9月5日：取消额外N4且教师隔离','archive/research_records/2026-09-05_no_n4_clean_teacher_replication/RESULT.md'),
        ('9月5日：自动前景拟合N4','archive/research_records/2026-09-05_n4_otsu_clean_replication/RESULT.md'),
        ('9月7日：患者均衡采样','archive/research_records/2026-09-07_subject_balanced_sampling/RESULT.md'),
        ('9月7日：切除腔辅助监督','archive/research_records/2026-09-07_structured_background_aux/RESULT.md'),
        ('9月8日：空间往返保真','archive/research_records/2026-09-08_fit_spatial_roundtrip_v2/RESULT.md'),
        ('9月10日：第一轮集中数据处理','archive/research_records/2026-09-10_eight_hour_data_processing/RESULT.md'),
        ('9月10日：第二轮集中数据处理','archive/research_records/2026-09-10_resumed_eight_hour/RESULT.md'),
        ('9月11日：自动工作流搜索','archive/research_records/2026-09-11_automatic_workflow_search/RESULT.md'),
        ('9月11日：固定内部确认失败','archive/research_records/2026-09-11_automatic_workflow_confirmation/RESULT.md'),
        ('9月11日：搜索随机性','archive/research_records/2026-09-11_workflow_search_seed_sensitivity/RESULT.md'),
        ('9月11日：单病例总耗时','archive/research_records/2026-09-11_workflow_file_to_output_timing/RESULT.md'),
        ('9月11日：批量调度不同患者复制','archive/research_records/2026-09-11_case_parallel_replication24/RESULT.md')]
    for label,path in timeline:
        assert (ROOT/path).exists(),path
        evidence.append('- '+link(path,label))
    evidence.extend(['','## 32张方法卡的对应原始材料',''])
    for i,c in enumerate(cards,1):
        assert (ROOT/c['source']).exists(),c['source']
        evidence.extend([f'## M{i:02d}','',c['title']+'：'+link(c['source'],'查看原始记录')+'。',''])
    evidence.extend(['## 当前数字、模型与数据角色','',
        '- '+link('AFLOW_BRATS_LEADERBOARD.md','当前各队列排行榜及置信区间')+'。其原始报告和checkpoint链接保持原样。',
        '- '+link('archive/research_records/2026-09-07_reused_development_roles/AUTHORIZATION.md','复用开发数据授权')+'；'+link('archive/research_records/2026-09-07_reused_development_roles/SPLIT.json','固定患者划分')+'。',
        '- '+link('archive/research_records/2026-09-04_direct_locator_flip8_tta_stability/RESULT.md','总体领先模型的报告和权重')+'。',
        '- '+link('archive/research_records/2026-08-31_gcsam_full972_stability/RESULT.md','单次前向Scan领先检查点及区间')+'。',
        '- '+link('archive/reviews/2026-09-05_30h_chain_audit/TEACHER_INNER_EXPOSURE.json','历史教师患者接触核查')+'。',
        '- 外部比较以当前排行榜及其更正为准；旧跨队列V5“UCSD56确定为负”等措辞已经被后续数值精度审计弱化，不直接复制为最新结论。','',
        '## 保存与整理说明','',
        '[清理前压缩原件](originals/)可逐字节恢复；[整理收据](PUBLICATION_RECEIPT.json)记录前后哈希与校验。详细快照中的历史进程ID和命令只作追溯，不是恢复任务的授权。',
        '此前生成的9月14日全量ZIP是本次清理前快照，保持原样，不冒充本次精简版。'])
    with (DEST/'EVIDENCE.md').open('x') as f:f.write('\n'.join(evidence)+'\n')
    updates={name:(DEST/'drafts'/name).read_bytes() for name in DOCS}
    agent=originals['AGENTS.md'].decode()
    agent=agent.replace('`AFLOW_BRATS_RESULTS.md` (historical compatibility)','`AFLOW_BRATS_RESULTS.md` (active chronological research log)')
    agent=agent.replace('`AFLOW_BRATS_WEEKLY_REPORT.md` (historical compatibility)','`AFLOW_BRATS_WEEKLY_REPORT.md` (mentor-facing research report)')
    agent+='''

## 导师可读文档维护（2026-09-14用户要求）

根文档分工固定，不再将同一张长卡片复制到多个文件：

- `AFLOW_BRATS_WEEKLY_REPORT.md`是导师汇报：研究起点、核心优化、最终Dice、完成度与局限；重要结论改变时更新，通常控制在150行内。
- `AFLOW_BRATS_RESULTS.md`是从2026-08-13起的连续日期流水账。每次工作追加/更新对应日期，用“问题→做法→结果→决定”说明；明确仅文档或工程工作是否没有新精度结果。不得再让日期入口停在旧阶段。
- `AFLOW_BRATS_METHOD_CATALOG.md`保留通俗方法卡及最终状态：保留六个必要字段，短句说明直觉、唯一改动、固定项、结果、成本和下一步，链接标签用“核对材料”等可读名称。不直接显示长目录、哈希、PID、启动命令或多份收据。逐项细节可进入资料层，但负结果必须留有索引。
- `AFLOW_BRATS_HANDOFF_CURRENT.md`只写当前状态、数据角色、真实待办及禁止恢复的已关闭分支，通常控制在100行内；历史“运行中”不得累积在正文。
- `AFLOW_BRATS_LEADERBOARD.md/.json`保持精确分队列排名和证据约束，不为精简汇报删除旧排名。

新方法的七问、预注册、过程日志、完整参数、路径、哈希和raw结果仍放对应实验归档；正式结果需同步日期流水账和方法卡。禁止继续调用旧`publish_research_card.py`把完整技术卡自动前插到handoff和方法目录；它保留作历史重现，不再作为新文档发布入口。

历史完整根文档与来源索引见`archive/reviews/2026-09-14_mentor_facing_cleanup/EVIDENCE.md`。当前汇报不应把历史“当时纪录”写成“当前最好”；内部、开发、交互、外部及不同栅格必须明确区分。文档清理不改变数据访问历史或赋予新测试许可。
'''
    updates['AGENTS.md']=agent.encode()
    badge='> 历史长交接（原文件名为8月13日，正文实际追加至8月18日）。本页保留原文追溯，不是当前执行入口；汇报看[导师报告](AFLOW_BRATS_WEEKLY_REPORT.md)，连续日期看[科研流水账](AFLOW_BRATS_RESULTS.md)，继续工作看[当前接续状态](AFLOW_BRATS_HANDOFF_CURRENT.md)。下文旧“唯一权威/当前”均是历史时点。\n\n'
    updates[HISTORICAL]=badge.encode()+originals[HISTORICAL]
    for name,before in originals.items():assert (ROOT/name).read_bytes()==before,'concurrent document edit'
    for name,after in updates.items():(ROOT/name).write_bytes(after)
    for item in receipts:
        after=(ROOT/item['path']).read_bytes()
        item.update({'after_sha256':sha(after),'before_lines':len(originals[item['path']].splitlines()),'after_lines':len(after.splitlines())})
    for name,digest in board_sha.items():assert sha((ROOT/name).read_bytes())==digest,'leaderboard unexpectedly changed'
    checks={}
    for name in ('verify_aflow_brats_doc_layout.py','verify_aflow_brats_method_catalog.py','verify_aflow_brats_leaderboard.py'):
        result=subprocess.run([sys.executable,str(ROOT/'scripts'/name)],cwd=ROOT,capture_output=True,text=True)
        checks[name]={'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr}
    receipt={'created_utc':datetime.now(timezone.utc).isoformat(),'changes':receipts,'validators':checks,
             'leaderboard_hashes_unchanged':board_sha,'original_research_artifacts_modified':False,
             'new_model_or_data_access':False,'subagent_tasks':['timeline','methods','claims'],
             'scope':'rewrite human-facing roots; retain exact compressed originals and readable technical snapshots',
             'publisher_sha256':sha(Path(__file__).read_bytes())}
    with (DEST/'PUBLICATION_RECEIPT.json').open('x') as f:json.dump(receipt,f,ensure_ascii=False,indent=2)
    assert all(r['returncode']==0 for r in checks.values()),checks
    print(json.dumps({'status':'published_and_verified','changes':[{k:v for k,v in r.items() if k in ('path','before_lines','after_lines')} for r in receipts]},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
