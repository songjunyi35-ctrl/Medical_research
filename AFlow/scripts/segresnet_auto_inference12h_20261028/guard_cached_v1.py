"""Same caps; avoid a network filesystem inventory for every tiny CPU task."""
import time,shutil
from scripts.segresnet_auto_inference12h_20261028.common import ROOT,R,TMP,read
from scripts.segresnet_auto_inference12h_20261028.guard_v2 import stable_size
_last_inventory=0.
_last_disk=0.
_size=0


def guard(force=False):
    global _last_inventory,_last_disk,_size
    stamp=time.monotonic();assert time.time()<read(R/'AUTHORIZATION.json')['deadline_unix'],'12h window ended'
    if force or stamp-_last_disk>=.5:
        assert shutil.disk_usage(ROOT).free>=20*2**30,'persistent reserve below20GiB'
        if TMP.exists():assert shutil.disk_usage(TMP).free>=70*2**30,'scratch reserve below70GiB'
        _last_disk=stamp
    # Registered plaintext-output projection stays well below2GiB. Large CSV
    # writers additionally force this check at every bounded200k-row shard.
    if force or stamp-_last_inventory>=5 or _size>=int(1.5*2**30):
        _size=stable_size(R);assert _size<2*2**30,'new persistent above2GiB';_last_inventory=stamp
