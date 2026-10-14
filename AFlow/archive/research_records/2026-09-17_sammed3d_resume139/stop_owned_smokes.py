import os,signal,time,json
from pathlib import Path
root=Path('/mlx_devbox/users/songjunyi.666/playground/AFlow');r=root/'archive/research_records/2026-09-17_sammed3d_resume139'
assert json.loads((r/'AMP_TEST.json').read_text())['status']=='passed'
assert json.loads((r/'DDP_SMOKE.json').read_text())['status']=='passed'
owned={494:'scripts/check_sammed3d_amp_recovery.py',495:'scripts/check_sammed3d_amp_recovery.py',628:'scripts/train_sammed3d_single972_resume139.py\x00smoke',629:'scripts/train_sammed3d_single972_resume139.py\x00smoke'}
for pid,needle in owned.items():
 p=Path(f'/proc/{pid}/cmdline')
 if p.exists() and needle.encode() in p.read_bytes():os.kill(pid,signal.SIGTERM)
time.sleep(3)
for pid,needle in owned.items():
 p=Path(f'/proc/{pid}/cmdline')
 if p.exists() and needle.encode() in p.read_bytes():os.kill(pid,signal.SIGKILL)
(r/'SMOKE_CLEANUP.json').write_text(json.dumps({'reason':'completed smoke processes retained GPU after gate receipt; only exact owned PIDs terminated','pids':list(owned)}))
