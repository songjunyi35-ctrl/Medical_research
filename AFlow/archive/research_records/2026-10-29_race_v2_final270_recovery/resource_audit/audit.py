"""Independent read-only resource audit after original training terminal.

Writes only this new recovery resource_audit directory. No patient inference,
no test labels, no training/code/checkpoint/cache mutation on either node.
"""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path('/mlx_devbox/users/songjunyi.666/playground/AFlow')
OUT = Path(__file__).resolve().parent
TRAIN = ROOT / 'archive/research_records/2026-10-29_race_v2_training'
REMOTE_PY = ROOT.parent / 'sammed3d-venv/bin/python'
SSH = ['ssh', '-p', '10275', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
       '-o', 'UserKnownHostsFile=/tmp/aflow_race_v2_20261029_known_hosts',
       'tiger@fdbd:dccd:cde2:1086:109d:c0df:800c:b0e2']

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def save(name, data):
    path = OUT / name
    assert not path.exists(), 'never overwrite audit receipt: ' + str(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)

REMOTE_SCRIPT = r'''
import datetime,hashlib,json,os,platform,shutil,socket,subprocess,sys,time
from pathlib import Path
ROOT=Path('/mlx_devbox/users/songjunyi.666/playground/AFlow')
TRAIN=ROOT/'archive/research_records/2026-10-29_race_v2_training'
TMP=Path('/tmp/aflow_race_v2_20261029')
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def command(args):return subprocess.check_output(args,text=True).strip()
begun=time.monotonic()
result=dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),worker='1154259',
 hostname=socket.gethostname(),audit_pid=os.getpid(),python_executable=sys.executable,
 old_training_pids={str(p):os.path.exists('/proc/'+str(p)) for p in [2941,4603,4604,4608,4609]},
 nvidia_gpu_before=command(['nvidia-smi','--query-gpu=index,uuid,name,driver_version,memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits']),
 compute_apps_before=command(['nvidia-smi','--query-compute-apps=pid,process_name,gpu_uuid,used_gpu_memory','--format=csv,noheader,nounits']),
 root_disk=shutil.disk_usage(ROOT)._asdict(),tmp_disk=shutil.disk_usage('/tmp')._asdict(),
 caches={},checkpoints={},test_labels_accessed=False,model_forwards=0,gpu_tensor_allocations=0,
 lost_worker_reason='unknown; not inferred from resource disappearance')
assert not any(result['old_training_pids'].values())
assert not result['compute_apps_before'], 'GPU compute process present at audit start'
assert result['root_disk']['free']>=20*2**30
for split,count in [('train',972),('validation',108)]:
 folder=TMP/'native'/split
 expected=read(TRAIN/f'DATA_{split}.json');index=read(folder/'index.json')
 assert index['schema']=='aflow.race.native' and index['test_access'] is False
 assert len(index['case_ids'])==count and index['canvas']==[160,208,160]
 assert sha(folder/'index.json')==expected['index_sha256']
 payloads={}
 for name,size in expected['payload_bytes'].items():
  path=folder/name;actual_size=path.stat().st_size;digest=sha(path)
  assert actual_size==size and digest==expected['digests'][name],(split,name)
  payloads[name]=dict(path=str(path),bytes=actual_size,sha256=digest,historical_sha_match=True)
 result['caches'][split]=dict(count=count,index_sha256=expected['index_sha256'],payloads=payloads,
  actual_payload_bytes_hashed=True,canvas=index['canvas'],test_access=False)
backup=read(TRAIN/'RECOVERY_BACKUP.json')
for name in ('best','latest'):
 expected=backup['latest_by_name'][name]
 for kind,path in [('run',TMP/'run'/f'{name}.pth'),('pinned_backup',Path(expected['worker_snapshot']))]:
  digest=sha(path);assert digest==expected['sha256']
  result['checkpoints'][name+'_'+kind]=dict(path=str(path),bytes=path.stat().st_size,sha256=digest,
   expected_new_epoch=expected['new_epoch'],expected_new_updates=expected['new_updates'],unchanged=True)
import torch,monai,nibabel,SimpleITK
assert torch.cuda.is_available() and torch.cuda.device_count()==2
result['environment']=dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,
 monai=monai.__version__,nibabel=nibabel.__version__,SimpleITK=SimpleITK.Version_VersionString(),
 cuda_available=True,cuda_device_count=torch.cuda.device_count(),
 device_names=[torch.cuda.get_device_name(i) for i in range(2)],
 gpu_tensor_allocations=0,model_forwards=0)
result['root_disk_after']=shutil.disk_usage(ROOT)._asdict()
result['seconds']=time.monotonic()-begun;result['status']='passed'
print(json.dumps(result,ensure_ascii=False))
'''

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                       OPENBLAS_NUM_THREADS='1', MPLCONFIGDIR='/tmp/aflow_resource_audit_mpl')
    remote_command = shlex.join(['env', 'PYTHONDONTWRITEBYTECODE=1', 'OMP_NUM_THREADS=2',
                                'MKL_NUM_THREADS=2', 'OPENBLAS_NUM_THREADS=1',
                                'MPLCONFIGDIR=/tmp/aflow_resource_audit_mpl',
                                str(REMOTE_PY), '-c', REMOTE_SCRIPT])
    completed = subprocess.run(SSH + [remote_command], text=True, capture_output=True,
                               check=True, timeout=240, env=environment)
    worker = json.loads(completed.stdout)
    save('WORKER_RESOURCE.json', worker)
    if completed.stderr:
        save('WORKER_STDERR.json', {'stderr': completed.stderr})
    sys.path.insert(0, str(ROOT))
    from scripts.race_v2_20261029.common import verify_sources
    frozen = verify_sources()
    assert len(frozen['source_sha256']) == 25
    import torch
    from scripts.race_v2_20261029.model import RaceBenefitSeg
    torch.set_num_threads(2)
    done, handoff, result = (read(TRAIN / name) for name in ('DONE.json','HANDOFF.json','RESULT.json'))
    assert done['state'] == result['state'] == 'complete'
    assert done['new_epoch'] == result['actual_new_epochs'] == handoff['total_stage_new_epochs'] == 80
    assert done['new_updates'] == result['actual_new_updates'] == handoff['total_stage_new_updates'] == 9760
    assert handoff['actual_new_epochs'] == 0 and handoff['actual_new_updates'] == 0
    assert result['best_new_epoch'] == 0 and not result['promoted']
    model = RaceBenefitSeg(**handoff['construction'])
    portable = {}
    for name in ('best','latest'):
        expected = result['portable'][name]
        path = Path(expected['path'])
        assert sha(path) == expected['sha256']
        saved = torch.load(path, map_location='cpu', weights_only=False)
        model.load_state_dict(saved['model_state_dict'], strict=True)
        assert all(bool(v.isfinite().all()) for v in model.state_dict().values())
        assert saved['new_epoch'] == expected['new_epoch'] and saved['new_updates'] == expected['new_updates']
        portable[name] = dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path),
                              new_epoch=saved['new_epoch'], new_updates=saved['new_updates'], strict_cpu_reload=True)
        del saved
    assert sha(handoff['checkpoint_path']) == handoff['checkpoint_sha256'] == portable['best']['sha256']
    backups = read(TRAIN/'RECOVERY_BACKUP.json')['latest_by_name']
    backup_receipts = {}
    for name in ('best','latest'):
        receipt = backups[name]
        assert sha(receipt['local_path']) == receipt['sha256']
        saved = torch.load(receipt['local_path'], map_location='cpu', weights_only=False)
        model.load_state_dict(saved['model_state_dict'], strict=True)
        assert saved['new_epoch'] == receipt['new_epoch'] and saved['new_updates'] == receipt['new_updates']
        assert 'optimizer_state_dict' in saved
        assert all(int(v['step']) == saved['new_updates'] for v in saved['optimizer_state_dict']['state'].values())
        backup_receipts[name] = dict(path=receipt['local_path'], sha256=sha(receipt['local_path']),
                                    new_epoch=saved['new_epoch'],new_updates=saved['new_updates'],
                                    strict_cpu_reload=True,optimizer_checked=True)
        del saved
    local = dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(), status='passed',
                 execution_freeze_sha256=sha(TRAIN/'EXECUTION_FREEZE.json'),frozen_source_count=25,
                 frozen_sources_exact=True,recipe_budget_idea_AST_verified=True,
                 portable=portable,head_tmp_recovery_backup=backup_receipts,
                 original_training=dict(state='complete',new_epochs=80,new_updates=9760,best_new_epoch=0),
                 original_training_files_changed=False,test_labels_accessed=False,model_forwards=0)
    save('LOCAL_TRAINING_FREEZE.json', local)
    save('SUMMARY.json', dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='passed',
         worker='1154259',worker_receipt_sha256=sha(OUT/'WORKER_RESOURCE.json'),
         local_receipt_sha256=sha(OUT/'LOCAL_TRAINING_FREEZE.json'),audit_source_sha256=sha(__file__),
         lost_worker_reason='unknown',training_or_inference_started=False,
         resource_allocation_owner='root; this audit does not allocate GPUs'))
    print(json.dumps(read(OUT/'SUMMARY.json'),ensure_ascii=False))

if __name__ == '__main__':
    main()
