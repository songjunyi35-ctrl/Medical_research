"""Derive continuation entrypoints without mutating the frozen 50-epoch run."""
import ast
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / 'archive/research_records/2026-09-29_segresnet_native_multiscale_resume50'
OLD_R = ROOT / 'archive/research_records/2026-09-24_segresnet_native_multiscale_supervision'
OLD_ARCHIVE = '2026-09-24_segresnet_native_multiscale_supervision'
NEW_ARCHIVE = '2026-09-29_segresnet_native_multiscale_resume50'
OLD_OUT = 'native_multiscale_20260924'
NEW_OUT = 'native_multiscale_resume50_20260929'
RESUME_SHA = '93a969ad592849e2d057111c0111837726cb44b9bd126668aa6567faa6dba8a6'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def exactly(text, old, new, count=1):
    if text.count(old) != count:
        raise ValueError(f'expected {count} source matches, found {text.count(old)}: {old[:100]}')
    return text.replace(old, new)


def base(name):
    path = ROOT / 'scripts' / name
    freeze = json.loads((OLD_R / 'EXECUTION_FREEZE.json').read_text())
    if sha(path) != freeze['source_sha256']['scripts/' + name]:
        raise ValueError('frozen source identity changed: ' + name)
    return path.read_text().replace(OLD_ARCHIVE, NEW_ARCHIVE).replace(OLD_OUT, NEW_OUT)


def save(name, source):
    ast.parse(source)
    path = ROOT / 'scripts' / name
    if path.exists():
        raise FileExistsError('derived entrypoint already exists: ' + name)
    path.write_text(source)
    return str(path.relative_to(ROOT)), sha(path)


def main():
    written = {}
    train = base('train_native_multiscale_20260924.py')
    train = exactly(train, '/tmp/aflow_multiscale_native_20260924_v2', '/tmp/aflow_multiscale_resume50_20260929')
    train = exactly(train,
        "    stop_reason = ('epoch budget completed' if start > EPOCHS else\n                   'epoch50 same-budget gain below0.003' if start > 50 and best < .71971019177194 else None)",
        "    stop_reason = 'epoch budget completed' if start > EPOCHS else None")
    train = exactly(train,
        "            if epoch == 50 and best < .71971019177194:\n                stop_reason = 'epoch50 same-budget gain below0.003'\n", '')
    train = exactly(train, "    resume = previous.exists()\n", "    resume = previous.exists()\n    if not resume:\n        raise RuntimeError('continuation requires complete latest epoch50-or-later state')\n")
    train = exactly(train, "        OUT.mkdir(parents=True, exist_ok=True)\n",
        "        OUT.mkdir(parents=True, exist_ok=True)\n        if not (R / 'RESUME_PREACCESS.json').exists():\n            write(R / 'RESUME_PREACCESS.json', {'utc': now(), 'worker': 1114913, 'start_epoch': start, 'source_resume_sha256': sha(previous), 'loaded_optimizer_updates': (start - 1) * 122, 'old_best_epoch_carried': best_epoch, 'old_best_scan_carried': best, 'test_pixels_access': False, 'freeze_sha256': sha(R / 'EXECUTION_FREEZE.json')})\n")
    a, b = train.index('def smoke():'), train.index('def train():')
    smoke = train[a:b]
    smoke = exactly(smoke, "    system.backbone.load_state_dict(source(), strict=True)\n",
        "    resume_snapshot = torch.load(OUT / 'latest.pth', map_location='cpu', weights_only=False)\n    if resume_snapshot['epoch'] != 50 or sha(OUT / 'latest.pth') != '" + RESUME_SHA + "':\n        raise RuntimeError('synthetic resume smoke requires the frozen epoch50 state')\n    system.load_state_dict(resume_snapshot['system_state_dict'], strict=True)\n")
    smoke = exactly(smoke,
        "    opt = torch.optim.AdamW(ddp.parameters(), lr=2e-4, weight_decay=1e-5, fused=True)\n",
        "    opt = torch.optim.AdamW(ddp.parameters(), lr=2e-4, weight_decay=1e-5, fused=True)\n    opt.load_state_dict(resume_snapshot['optimizer_state_dict'])\n")
    smoke = exactly(smoke, "'worker': 1109780", "'worker': 1114913")
    smoke = exactly(smoke, "'model_source_sha256': PRETRAIN_SHA", "'model_source_sha256': '" + RESUME_SHA + "', 'resume_epoch': 50, 'optimizer_updates_before_synthetic_steps': 6100")
    train = train[:a] + smoke + train[b:]
    train = train.replace("'worker': 1109780", "'worker': 1114913")
    p, h = save('train_native_multiscale_resume50_20260929.py', train); written[p] = h

    supervisor = base('supervise_native_multiscale_20260924.py')
    supervisor = supervisor.replace('train_native_multiscale_20260924.py', 'train_native_multiscale_resume50_20260929.py').replace('finalize_native_multiscale_20260924.py', 'finalize_native_multiscale_resume50_20260929.py')
    supervisor = supervisor.replace('MAX_WALL_SECONDS = 14 * 3600', 'MAX_WALL_SECONDS = 8 * 3600').replace('14h wall budget exceeded', '8h continuation wall budget exceeded')
    supervisor = supervisor.replace("'worker': 1109780", "'worker': 1114913")
    supervisor = exactly(supervisor, "'resume_from_latest': attempt > 1", "'resume_from_latest': True")
    p, h = save('supervise_native_multiscale_resume50_20260929.py', supervisor); written[p] = h

    final = base('finalize_native_multiscale_20260924.py')
    final = final.replace('MULTISCALE_20260924', 'MULTISCALE_RESUME_20260929')
    final = exactly(final, "stop['last_complete_epoch'] not in (50, 200)", "stop['last_complete_epoch'] != 200")
    final = exactly(final, "    write(R / 'FINAL_RESULT.json', output)\n",
        "    output.update(resumed_after_epoch=50, original_completed_updates=6100, new_stage_updates=output['completed_updates']-6100, resume_checkpoint_sha256='" + RESUME_SHA + "', adaptive_budget_amendment=True, original_epoch50_gate_failed=True, current_worker=1114913)\n    write(R / 'FINAL_RESULT.json', output)\n")
    final = final.replace('## 2026-09-24 原生SegResNet多尺度辅助监督：用户授权新一组训练', '## 2026-09-29 多尺度辅助监督50轮续训到200轮：用户最新授权')
    final = final.replace('2026-09-24多尺度辅助监督终态', '2026-09-29多尺度辅助监督续训终态').replace('2026-09-24新组终态', '2026-09-29续训终态')
    final = exactly(final,
        '状态：{stage}，共{result[\'completed_updates\']}次更新；本组只训练原972扫描/444患者，以固定108/50验证选择。',
        '状态：{stage}，累计{result[\'completed_updates\']}次更新，其中原50轮6100更新、新增150轮18300更新；从完整第50轮模型/辅助头/AdamW状态继续，原停止记录保留。用户看过50轮验证后追加预算，此次为自适应预算修订；本组只训练原972扫描/444患者，以固定108/50验证选择。')
    start = final.index('缓存在worker1109780的/tmp独立重建')
    stop = final.index('\n\n所报显存峰值', start)
    final = final[:start] + ('旧worker1109780已不在活动清单，本阶段在worker1114913两张A10080GB上续训；每卡4/global8及原学习率不变，硬件迁移可能改变浮点轨迹，不承诺逐位连续。新worker/tmp原生训练/验证缓存与旧载荷SHA一致；原50轮主干、辅助头和AdamW步计数6100全部恢复，原最佳49轮参与完整1–200轮选择。旧50轮规则未通过的事实与原报告、快照完整保留。') + final[stop:]
    final = final.replace('## 原始分辨率SegResNet多尺度辅助监督｜{stage}', '## 原始分辨率SegResNet多尺度辅助监督续训｜{stage}')
    final = final.replace('**一句话：** 主模型交完整肿瘤图，训练时让两个中间尺度也练习识别肿瘤。 **出发点：** 270影像整体特征与原972相近，旧组合166同格验证Scan0.751179，训练病例也有未拟合错误。', '**一句话：** 从第50轮完整状态继续到总200轮，观察后半程是否有收益。 **出发点：** 前50最佳较旧同预算+0.002048，原0.003门未过；用户看过结果后追加预算。')
    p, h = save('finalize_native_multiscale_resume50_20260929.py', final); written[p] = h
    receipt = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'derived_sources_sha256': written,
               'old_execution_freeze_sha256': sha(OLD_R / 'EXECUTION_FREEZE.json'),
               'change': 'separate continuation paths; remove only epoch50 budget stop; keep numerical training formula and total200 schedule; restore epoch50 full optimizer; adaptive-budget/hardware provenance in publication',
               'frozen_original_sources_modified': False}
    (R / 'SOURCE_DERIVATION.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == '__main__':
    main()
