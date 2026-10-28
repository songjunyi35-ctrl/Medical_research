"""Same disk/deadline limits; tolerate atomic status-file rename during sizing."""
from scripts.segresnet_auto_inference12h_20261028.common import R,ROOT,TMP,read,time,shutil


def stable_size(folder):
    total=0
    for p in folder.rglob('*'):
        try:
            if p.is_file():total+=p.stat().st_size
        except FileNotFoundError:
            # Concurrent atomic JSON publication can remove the temporary name.
            # The final name is counted on the next guard, rather than failing a job.
            continue
    return total


def guard():
    assert time.time()<read(R/'AUTHORIZATION.json')['deadline_unix'],'12h window ended'
    assert shutil.disk_usage(ROOT).free>=20*2**30,'persistent reserve below20GiB'
    assert stable_size(R)<2*2**30,'new persistent above2GiB'
    if TMP.exists():assert shutil.disk_usage(TMP).free>=70*2**30,'scratch reserve below70GiB'
