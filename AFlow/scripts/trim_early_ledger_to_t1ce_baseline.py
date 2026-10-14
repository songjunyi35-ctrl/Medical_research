"""User-authorized, recoverable removal of pre-baseline ledger prose only."""
from datetime import datetime,timezone
import gzip
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'archive/reviews/2026-09-14_ledger_baseline_cleanup'
LEDGER=ROOT/'archive/handoffs_pre_20260813/AFLOW_BRATS_RESULTS.md'
ARCHIVE_README=LEDGER.parent/'README.md'
HANDOFF=ROOT/'AFLOW_BRATS_HANDOFF_CURRENT.md'
MARKER='## 2026-08-13：官方预处理 vanilla SAM-Med3D baseline 复现完成\n'


def digest(data):return hashlib.sha256(data).hexdigest()


def main():
    if DEST.exists():raise FileExistsError('cleanup already recorded; do not repeat')
    old=LEDGER.read_bytes();text=old.decode()
    assert text.startswith('# AFlow x SAM-Med3D BraTS 实验结果记录\n')
    assert text.count(MARKER)==1
    offset=text.index(MARKER);retained=text[offset:]
    assert 'vanilla SAM-Med3D T1ce→WT baseline 复现' in retained
    header='''# BraTS T1ce/WT 科研流水账

正式阅读起点：2026-08-13，单通道T1ce→WT，师兄路线对应的vanilla SAM-Med3D基础模型复现完成。

按用户2026-09-14要求，已从本账本正文移除这个起点之前的旧工作流搜索、四模态方向及未对齐的早期试跑。以下复现记录原文保留，不改实验数字、交互协议或既有测试访问事实；文档裁剪不代表测试集重新成为盲测。

本早期账本的留存记录截至2026-08-13。此后的工作记录继续见：

- [8月13日起的长交接](../../AFLOW_BRATS_HANDOFF_20260813_SINGLE_REPO.md)
- [方法尝试与成败记录](../../AFLOW_BRATS_METHOD_CATALOG.md)
- [当前交接](../../AFLOW_BRATS_HANDOFF_CURRENT.md)

本文件是历史复现记录，不是当前实验执行授权。清理详情及压缩回退备份见[清理收据](../reviews/2026-09-14_ledger_baseline_cleanup/RECEIPT.json)。

---

'''
    new=(header+retained).encode()
    readme_old=ARCHIVE_README.read_bytes();handoff_old=HANDOFF.read_bytes()
    readme_note='''> 2026-09-14变更：本目录的`AFLOW_BRATS_RESULTS.md`已按用户要求裁剪，以2026-08-13单通道T1ce师兄路线baseline复现完成为正文起点；其他归档文件未改。下表仍是历史移动前SHA，不代表裁剪后账本的当前SHA。[变更、旧全文压缩备份与新SHA](../reviews/2026-09-14_ledger_baseline_cleanup/RECEIPT.json)。

'''
    handoff_note='''<!-- ledger-baseline-cleanup-20260914:start -->
## 流水账阅读起点已清理

按用户2026-09-14要求，[早期科研流水账](archive/handoffs_pre_20260813/AFLOW_BRATS_RESULTS.md)现从2026-08-13“单通道T1ce→WT、师兄路线基础模型复现完成”开始；此前旧搜索/四模态/未对齐试跑已从该正文删除，复现原文及此后各文件不改。仅文档清理，不改代码、数据、模型、指标、负结果档案或历史测试访问状态；不因此重新声称盲测。压缩回退备份和哈希见[清理收据](archive/reviews/2026-09-14_ledger_baseline_cleanup/RECEIPT.json)。此前全量文档包属于清理前快照，保持原样。
<!-- ledger-baseline-cleanup-20260914:end -->

'''
    changes=[(LEDGER,old,new),(ARCHIVE_README,readme_old,readme_note.encode()+readme_old),
             (HANDOFF,handoff_old,handoff_note.encode()+handoff_old)]
    DEST.mkdir(parents=True)
    backups=[]
    for path,before,after in changes:
        backup=DEST/(path.name+'.before_cleanup.gz')
        packed=gzip.compress(before,mtime=0)
        with backup.open('xb') as f:f.write(packed)
        assert gzip.decompress(backup.read_bytes())==before
        backups.append({'path':str(path.relative_to(ROOT)),'backup_path':str(backup.relative_to(ROOT)),
                        'before_sha256':digest(before),'after_sha256':digest(after),
                        'backup_sha256':digest(packed),'before_lines':len(before.splitlines()),'after_lines':len(after.splitlines())})
    receipt={'created_utc':datetime.now(timezone.utc).isoformat(),'operation':'remove pre-baseline prose from one early ledger',
        'authorized_by':'user requested T1ce-only brother-paper reproduction as starting point and removal of earlier ledger content',
        'cutoff_marker':MARKER.strip(),'retained_original_start_line':text[:offset].count('\n')+1,
        'removed_original_lines':text[:offset].count('\n'),'retained_original_lines':len(retained.splitlines()),
        'retained_original_sha256':digest(retained.encode()),'changes':backups,
        'scope_exclusions':['other historical ledgers and weekly reports','method catalog','experiment raw reports','code and checkpoints','image/label data and caches','existing pre-cleanup ZIP snapshots'],
        'recovery':'gzip-decompress the listed backup to recover the exact original bytes; do not overwrite subsequent edits blindly',
        'script_sha256':digest(Path(__file__).read_bytes())}
    with (DEST/'RECEIPT.json').open('x') as f:json.dump(receipt,f,ensure_ascii=False,indent=2)
    for path,before,after in changes:assert path.read_bytes()==before,'concurrent file edit'
    for path,before,after in changes:path.write_bytes(after)
    current=LEDGER.read_text()
    assert current[current.index(MARKER):]==retained
    assert '## 2026-07-20' not in current and '## 历史 work #1' not in current
    for path,before,after in changes:assert path.read_bytes()==after
    print(json.dumps({'ledger':str(LEDGER),'removed_lines':receipt['removed_original_lines'],
                      'retained_original_lines':receipt['retained_original_lines'],'new_lines':len(new.splitlines()),
                      'retained_content_exact':True,'recoverable':True},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
