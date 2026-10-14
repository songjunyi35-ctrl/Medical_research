"""Frozen fit-image-only benchmark. Never opens a label array."""
import hashlib
import json
from pathlib import Path
import statistics
import time
from datetime import datetime, timezone
import numpy as np
import torch
from sammed3d.automatic_fixed_workflow import AXES, WORKFLOWS, restored_atom, online_logits, fuse_atoms
from sammed3d.t1ce_wt_capg3d import AutomaticPromptLocator
from scripts.evaluate_t1ce_wt_flip8_tta_replication_inner import flip8_logits
from scripts import structured_background_protocol as p

RECORD = p.ROOT / 'archive/research_records/2026-09-11_automatic_workflow_engineering'
CHECKPOINT_SHAS = ('fdad97d9abc736638bcd9911fc82ddca1a434fd0c9a46c3e9306d608276b3d6e',
                  '5b014fb256c272b828e01322b7985f10d8ac5c53f19fa475761c25537f697e43')
IMAGE_SHA = 'f772520ca0d99fca0edc60aa04a7c73e0d7818eea9fa0174a29038f8334644b9'
FILES = ['sammed3d/automatic_fixed_workflow.py', 'tests/test_automatic_fixed_workflow.py',
         'scripts/benchmark_automatic_fixed_workflow.py', 'scripts/structured_background_protocol.py',
         'sammed3d/t1ce_wt_capg3d.py', 'scripts/evaluate_t1ce_wt_flip8_tta_replication_inner.py',
         str((RECORD/'IDEA.md').relative_to(p.ROOT))]


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    if (RECORD/'PREACCESS.json').exists():
        raise FileExistsError('Single engineering run already started')
    p.checked(p.FIT, p.FIT_SHA)
    fit = p.rows(p.FIT)
    assert len(fit) == 684 and len({r['subject_id'] for r in fit}) == 313
    selected = fit[:24]
    p.checked(p.CACHE/'index.json', p.INDEX_SHA)
    p.checked(p.CACHE/'images.f32', IMAGE_SHA)
    index = json.loads((p.CACHE/'index.json').read_text())
    reports = []
    for seed, digest in zip(p.SEEDS, CHECKPOINT_SHAS):
        ck = p.WORK/f'seed{seed}_control/fixed_epoch168.pth'
        p.checked(ck, digest)
        train_path = ck.parent/'TRAIN_DONE.json'
        train = json.loads(train_path.read_text())
        assert train['fit_sha256'] == p.FIT_SHA and train['fixed_epochs'] == 168
        assert not train['teacher_checkpoints']
        reports.append({'seed': seed, 'checkpoint': str(ck), 'sha256': digest,
                        'training_report_sha256': p.sha(train_path)})
    p.write_new(RECORD/'PREACCESS.json', {
        'created_utc': now(), 'files': {name: p.sha(p.ROOT/name) for name in FILES},
        'models': reports, 'fit_sha256': p.FIT_SHA, 'index_sha256': p.INDEX_SHA,
        'images_sha256': IMAGE_SHA, 'cases': [r['case_id'] for r in selected],
        'views': WORKFLOWS, 'expected_case_forwards': 1654, 'forwards_at_write': 0,
        'accuracy_access': False, 'labels_opened': False})
    # No BrainCropDataset: it maps labels alongside images. Image memmap only.
    cases = index['case_ids']
    slots = {case: i for i, case in enumerate(cases)}
    images = np.memmap(p.CACHE/'images.f32', dtype=np.float32, mode='r',
                       shape=(len(cases), 128, 128, 128))
    tensors = [torch.from_numpy(np.array(images[slots[r['case_id']]], copy=True))[None, None]
               for r in selected]
    assert all(torch.isfinite(x).all() for x in tensors)
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    torch.set_num_threads(1)
    started = time.monotonic()
    rows, parity = [], []
    calls = [0]

    def check_deadline():
        if time.monotonic() - started > 900:
            raise TimeoutError('Frozen 15-minute GPU phase limit')

    def count_hook(module, args, output):
        calls[0] += len(args[0])

    try:
        for report in reports:
            model = AutomaticPromptLocator(center_aware=False).to(device).eval()
            payload = torch.load(report['checkpoint'], map_location='cpu', weights_only=False)
            model.load_state_dict(payload['model_state_dict'], strict=True)
            handle = model.register_forward_hook(count_hook)
            torch.cuda.reset_peak_memory_stats(device)
            seed_start_calls = calls[0]
            with torch.inference_mode():
                for ci, cpu in enumerate(tensors[:2]):
                    check_deadline()
                    image = cpu.to(device)
                    before = image.clone()
                    atoms = [restored_atom(model, image, v) for v in range(8)]
                    for name, views in WORKFLOWS.items():
                        n = calls[0]
                        actual = online_logits(model, image, views)
                        expected = fuse_atoms(atoms, views)
                        assert calls[0] - n == len(views)
                        assert torch.equal(actual, expected), f'atom parity {name}'
                        assert torch.isfinite(actual).all()
                        parity.append({'seed': report['seed'], 'case_id': selected[ci]['case_id'],
                                       'workflow': name, 'max_abs_error': 0, 'calls': calls[0]-n})
                    old_identity, old_full = flip8_logits(model, image)
                    assert torch.equal(old_identity, atoms[0])
                    assert torch.equal(old_full, fuse_atoms(atoms, range(8)))
                    assert torch.equal(before, image)
                    del atoms, actual, expected, old_identity, old_full, before, image

                def measure(cpu, name):
                    check_deadline()
                    views = WORKFLOWS[name]
                    n = calls[0]
                    torch.cuda.synchronize()
                    begin = time.perf_counter()
                    image = cpu.to(device)
                    logits = online_logits(model, image, views)
                    mask = (logits > 0).cpu()
                    point = int(logits.flatten().argmax().cpu())
                    torch.cuda.synchronize()
                    seconds = time.perf_counter()-begin
                    assert calls[0]-n == len(views)
                    assert mask.shape == (1, 1, 128, 128, 128) and 0 <= point < 128**3
                    return seconds, calls[0]-n

                for name in WORKFLOWS:
                    for cpu in tensors[:3]:
                        measure(cpu, name)
                names = list(WORKFLOWS)
                for repeat in range(2):
                    for ci, cpu in enumerate(tensors):
                        order = names if (ci+repeat) % 2 == 0 else list(reversed(names))
                        for name in order:
                            seconds, used = measure(cpu, name)
                            rows.append({'seed': report['seed'], 'case_id': selected[ci]['case_id'],
                                         'repeat': repeat, 'workflow': name,
                                         'seconds': seconds, 'case_forwards': used})
            assert calls[0]-seed_start_calls == 827
            report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated(device)
            handle.remove()
            del model, payload
            torch.cuda.empty_cache()
        assert calls[0] == 1654 and len(rows) == 384 and len(parity) == 16
        summary = {}
        for report in reports:
            seed = report['seed']
            by_key = {(r['case_id'], r['repeat'], r['workflow']): r['seconds'] for r in rows if r['seed']==seed}
            stats = {}
            for name in WORKFLOWS:
                durations = [r['seconds'] for r in rows if r['seed']==seed and r['workflow']==name]
                ratios = [by_key[(case['case_id'], rep, 'full8')]/by_key[(case['case_id'], rep, name)]
                          for case in selected for rep in range(2)]
                stats[name] = {'mean_seconds': statistics.fmean(durations),
                               'median_seconds': statistics.median(durations),
                               'paired_median_speedup_vs_full8': statistics.median(ratios)}
            summary[str(seed)] = stats
        gates = {seed: {'identity_2x': s['identity']['paired_median_speedup_vs_full8'] >= 2,
                        'four_1_2x': s['single_axis4']['paired_median_speedup_vs_full8'] >= 1.2}
                 for seed, s in summary.items()}
        raw = {'status': 'completed', 'created_utc': now(), 'models': reports,
               'hardware': torch.cuda.get_device_name(device), 'torch_version': torch.__version__,
               'rows': rows, 'parity': parity, 'summary': summary, 'gates': gates,
               'all_gates_passed': all(all(g.values()) for g in gates.values()),
               'case_forwards': calls[0], 'gpu_phase_seconds': time.monotonic()-started,
               'preaccess_sha256': p.sha(RECORD/'PREACCESS.json'),
               'access': {'fit_images': True, 'labels': False, 'development': False,
                          'test': False, 'outer': False, 'confirmation': False, 'external': False},
               'scope': 'image tensor H2D through mask/point D2H; excludes image file loading, preprocessing and model loading',
               'accuracy_claim': None, 'accuracy_promotion': False}
        p.write_new(RECORD/'RAW_BENCHMARK.json', raw)
        print(json.dumps({k:v for k,v in raw.items() if k not in ('rows','parity')}, indent=2))
    except BaseException as error:
        p.write_new(RECORD/'FAILURE.json', {'created_utc': now(), 'error': repr(error),
                    'completed_case_forwards': calls[0], 'partial_rows': rows, 'parity': parity,
                    'labels_opened': False, 'preaccess_sha256': p.sha(RECORD/'PREACCESS.json')})
        raise


if __name__ == '__main__':
    main()
