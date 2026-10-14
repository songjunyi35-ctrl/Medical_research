"""Move four exact legacy artifacts; retain a short historical handoff path."""
from datetime import datetime,timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'archive/legacy_root_artifacts_20260914'
EXPECTED={
 'FIXED1_DEPLOYMENT_CANDIDATE_20260814.json':'e162f1bc337fb83c19163a6723a6d62f136c2518caaeb7def2d04eda6651a6d4',
 'MCTS_DEPLOYMENT_CANDIDATE_20260813.json':'89ae9e10992f9989ca99a86d96bb5b153eb91050b45096940ea2a871297a986b',
 'HANDOFF_CLEANUP_20260813.json':'49042f9cbf8bb6eb3970e0f73072f5094742576d1566409ea98f629d74f05400',
 'CODE_ARCHIVE_CANDIDATES_20260813.sha256':'786dc90a37756924911beb4cb32b43296d4848b2116bf62a47abe611003c21ee'}
HISTORICAL='AFLOW_BRATS_HANDOFF_20260813_SINGLE_REPO.md'
SCRIPTS=['scripts/evaluate_t1ce_wt_adaptive_router.py','scripts/summarize_t1ce_wt_fixed1_benchmark.py',
         'scripts/search_t1ce_wt_inference_workflow_mcts.py']


def sha(data):return hashlib.sha256(data).hexdigest()


def main():
    if (DEST/'MIGRATION_RECEIPT.json').exists():raise FileExistsError('already migrated')
    originals={name:(ROOT/name).read_bytes() for name in EXPECTED}
    for name,digest in EXPECTED.items():
        assert sha(originals[name])==digest,name
        assert not (ROOT/name).is_symlink() and not (DEST/name).exists()
    protected={name:sha((ROOT/name).read_bytes()) for name in SCRIPTS+['AFLOW_BRATS_LEADERBOARD.json','AFLOW_BRATS_LEADERBOARD.md']}
    history=(ROOT/HISTORICAL).read_bytes()
    packed=gzip.compress(history,mtime=0)
    with (DEST/(HISTORICAL+'.original.gz')).open('xb') as f:f.write(packed)
    def relocate(match):
        target=match.group(2)
        if target.startswith(('http://','https://','#')):return match.group(0)
        bits=target.split('#',1)
        suffix='#'+bits[1] if len(bits)>1 else ''
        return match.group(1)+os.path.relpath((ROOT/bits[0]).resolve(),DEST)+suffix+')'
    readable='# 已归档的历史长交接\n\n> 这是整理前全文，正文中的“当前/唯一权威”均是历史表述，不提供恢复实验的授权。当前工作看[当前交接](../../AFLOW_BRATS_HANDOFF_CURRENT.md)。本可读副本仅调整相对链接；原始字节与哈希另存。\n\n'
    readable+=re.sub(r'(\[[^\]]+\]\()([^)]+)\)',relocate,history.decode())
    with (DEST/HISTORICAL).open('x') as f:f.write(readable)
    stub='''# 历史交接兼容入口

这份2026-08-13命名、实际追加至8月18日的长交接已经归档，不再作为当前执行入口。

- [查看历史完整交接](archive/legacy_root_artifacts_20260914/AFLOW_BRATS_HANDOFF_20260813_SINGLE_REPO.md)
- [继续当前工作](AFLOW_BRATS_HANDOFF_CURRENT.md)
- [给导师汇报](AFLOW_BRATS_WEEKLY_REPORT.md) · [按日期回顾研究](AFLOW_BRATS_RESULTS.md)

此根路径只为兼容旧引用而保留；旧命令、候选和“当时最好”不代表当前状态。原文及SHA恢复方式见[迁移说明](archive/legacy_root_artifacts_20260914/README.md)。
'''
    # Prepare readable entry updates; preserve exact prior bytes for recovery.
    texts={name:(ROOT/name).read_bytes() for name in ['README.md','AGENTS.md','AFLOW_BRATS_HANDOFF_CURRENT.md','AFLOW_BRATS_RESULTS.md']}
    updated={name:body.decode() for name,body in texts.items()}
    updated['README.md']+='''

## 根目录其他文件是什么？

平时只需看上面的导师汇报、日期流水账和方法记录。`AGENTS.md`及当前交接供Agent接续；排行榜MD用于精确核对，唯一保留的JSON是它的机器数据；方法模板只在新实验前使用。旧交接现为短跳转页，不再是一份重复的当前交接。

三份旧JSON与旧哈希清单已移入[历史技术材料](archive/legacy_root_artifacts_20260914/README.md)。不需要你逐个打开，也没有删除实验数据或权重。
'''
    updated['AGENTS.md']+='''

## 根目录技术材料约束（2026-09-14）

根目录JSON仅允许`AFLOW_BRATS_LEADERBOARD.json`，其他实验候选、清理收据、哈希清单必须放入相应archive目录；不得新建根目录临时JSON或SHA文件。历史根交接仅保留不超过20行的兼容导航页，全文放archive。

旧候选路径迁移见`archive/legacy_root_artifacts_20260914/README.md`。冻结脚本原文不改；旧默认根路径已失效，历史复现需显式提供归档候选和预注册参数，不得自动恢复关闭的实验。

文档整理或新增技术材料后，除原三个校验外，运行`PYTHONDONTWRITEBYTECODE=1 ../sammed3d-venv/bin/python scripts/verify_aflow_root_artifacts.py`，防止根目录重新堆积历史JSON和长交接。
'''
    updated['AFLOW_BRATS_HANDOFF_CURRENT.md']+='\n旧候选JSON、旧清理收据与哈希清单已统一[移入历史资料](archive/legacy_root_artifacts_20260914/README.md)；根历史交接仅作短导航。旧脚本若需复现必须显式指定迁移路径，整理本身不授予恢复实验的权限。\n'
    needle='**结果：** 这是文档整理，没有新训练、新精度评测或新榜首。'
    assert updated['AFLOW_BRATS_RESULTS.md'].count(needle)==1
    updated['AFLOW_BRATS_RESULTS.md']=updated['AFLOW_BRATS_RESULTS.md'].replace(needle,needle+' 同日进一步将三份历史JSON和旧哈希清单移档，根历史长交接改为短导航，保留迁移映射与恢复备份。')
    for name,body in texts.items():
        with (DEST/(name+'.before_root_cleanup.gz')).open('xb') as f:f.write(gzip.compress(body,mtime=0))
    assert (ROOT/HISTORICAL).read_bytes()==history
    for name,body in texts.items():assert (ROOT/name).read_bytes()==body
    for name,body in originals.items():assert (ROOT/name).read_bytes()==body
    moves=[]
    for name,digest in EXPECTED.items():
        (ROOT/name).rename(DEST/name)
        assert sha((DEST/name).read_bytes())==digest and not (ROOT/name).exists()
        moves.append({'old':name,'new':str((DEST/name).relative_to(ROOT)),'sha256':digest,'byte_exact':True})
    (ROOT/HISTORICAL).write_text(stub)
    for name,text in updated.items():(ROOT/name).write_text(text)
    for name,digest in protected.items():assert sha((ROOT/name).read_bytes())==digest,name
    assert gzip.decompress((DEST/(HISTORICAL+'.original.gz')).read_bytes())==history
    receipt={'created_utc':datetime.now(timezone.utc).isoformat(),'moves':moves,
        'historical_handoff':{'old_sha256':sha(history),'readable_archive_sha256':sha((DEST/HISTORICAL).read_bytes()),
            'exact_backup':str((DEST/(HISTORICAL+'.original.gz')).relative_to(ROOT)),
            'root_stub_sha256':sha(stub.encode()),'old_lines':len(history.splitlines()),'new_lines':len(stub.splitlines())},
        'protected_files_unchanged':protected,'root_document_changes':{name:{'before_sha256':sha(texts[name]),'after_sha256':sha(text.encode())} for name,text in updated.items()},
        'compatibility':'old script default candidate paths no longer valid; explicit CLI overrides documented; no root aliases and no historical code changes',
        'data_models_or_reports_deleted':False,'new_experiment_access':False,
        'subagents':['root_audit','dependency_audit','handoff_audit'],'migrator_sha256':sha(Path(__file__).read_bytes())}
    with (DEST/'MIGRATION_RECEIPT.json').open('x') as f:json.dump(receipt,f,ensure_ascii=False,indent=2)
    print(json.dumps({'moved_files':len(moves),'root_json_remaining':[p.name for p in ROOT.glob('*.json')],
                      'historical_handoff_lines':len(stub.splitlines()),'old_scripts_unchanged':True,'recoverable':True},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
