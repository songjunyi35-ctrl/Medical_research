"""Publish truthful startup only after a real full epoch and strict reload."""
from __future__ import annotations

import datetime
import json
import subprocess
import sys
import time

from .governance import ROOT, RECORD, atomic_json, guard, sha

TRAIN = ROOT/'archive/research_records/2026-10-29_race_v2_training'


def main():
    while not (TRAIN/'STARTUP_VERIFIED.json').exists():
        guard(1154259, training=True)
        time.sleep(30)
    startup = json.loads((TRAIN/'STARTUP_VERIFIED.json').read_text())
    budget = json.loads((TRAIN/'BUDGET_DECISION.json').read_text())
    assert budget['status'] == 'passed' and budget['new_updates'] == budget['new_epochs']*122
    for split in ('train', 'validation'):
        assert json.loads((TRAIN/f'PAYLOAD_AUDIT_{split}.json').read_text())['status'] == 'passed'
    latest = json.loads((TRAIN/'STATUS.json').read_text())
    assert latest['new_epoch'] >= 1 and latest['new_updates'] == latest['new_epoch']*122
    replacements = {
        'AFLOW_BRATS_HANDOFF_CURRENT.md': [
            ('训练数据准备中', '正式微调进行中'),
            ('训练从e164新建优化器，972独立载荷正在重建；先80新增轮建议，所有数据／数值／预算门通过才开训。',
             f"训练972/108实际载荷、几何和数值／预算门已通过；从e164新建AdamW正式微调，冻结新增{budget['new_epochs']}轮／{budget['new_updates']}更新，首完整epoch、108验证及CPU严格重载通过。首轮分数仅为早期结果。[启动证据](archive/research_records/2026-10-29_race_v2_training/STARTUP_VERIFIED.json)。")],
        'AFLOW_BRATS_METHOD_CATALOG.md': [
            ('108首批完成、训练准备', '108首批完成、正式微调'),
            ('训练数据准备中，无新270分数。', '正式微调已通过首完整epoch／108验证／CPU重载，无新270分数。'),
            ('两卡训练数据门后拟80新增轮', f"两卡已冻结新增{budget['new_epochs']}轮并启动")],
        'AGENTS.md': [('当前处于真实缓存重建与合成检查，无新270成绩。',
                       '正式新优化器微调已启动，数据／数值／预算及首完整epoch／108验证／CPU重载通过，无新270成绩。')],
        'AFLOW_BRATS_LEADERBOARD.md': [('训练数据准备中，无新270成绩。', '正式微调已启动，无新270成绩。')],
    }
    for name, changes in replacements.items():
        path = ROOT/name; text = path.read_text()
        for old, new in changes:
            assert old in text, (name, old)
            text = text.replace(old, new, 1)
        path.write_text(text)
    board_path = ROOT/'AFLOW_BRATS_LEADERBOARD.json'
    board = json.loads(board_path.read_text())
    board['active_research_stage_20261029'].update(status='formal_training_running_initial108_complete',
        startup_evidence_path=str((TRAIN/'STARTUP_VERIFIED.json').relative_to(ROOT)),
        startup_evidence_sha256=sha(TRAIN/'STARTUP_VERIFIED.json'))
    board['updated_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    atomic_json(board_path, board)
    atomic_json(RECORD/'STARTUP_PUBLICATION.json', dict(state='formal_training_started',
        startup_evidence_sha256=sha(TRAIN/'STARTUP_VERIFIED.json'), budget=budget,
        no_new270_scoring=True, source_startup=startup))
    for script in ('verify_aflow_brats_doc_layout.py', 'verify_aflow_root_artifacts.py',
                   'verify_aflow_brats_method_catalog.py', 'verify_aflow_brats_leaderboard.py'):
        subprocess.run([sys.executable, str(ROOT/'scripts'/script)], cwd=ROOT, check=True)
    subprocess.run([sys.executable, '-m', 'scripts.race_v2_stage_20261029.publish_snapshot_v2', '--message',
                    'Verify real RACE-v2 finetuning startup with full epoch and strict checkpoint reload'], cwd=ROOT, check=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        atomic_json(RECORD/'STARTUP_PUBLICATION_FAILURE.json', {'at_unix': time.time(), 'error': repr(error),
                    'training_not_restarted': True, 'no_automatic_neural_retry': True})
        raise
