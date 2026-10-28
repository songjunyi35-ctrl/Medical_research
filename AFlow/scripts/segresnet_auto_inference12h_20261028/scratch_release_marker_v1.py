"""Permit a planned long renderer only after enough authorized cleanup completed."""
from scripts.segresnet_auto_inference12h_20261028.common import *
from scripts.segresnet_auto_inference12h_20261028.guard_v2 import guard

def main():
    resource=read(R/'RESOURCE_CURRENT.json');token=resource['token']
    while time.time()<read(R/'AUTHORIZATION.json')['deadline_unix']:
        guard();assert read(R/'RESOURCE_CURRENT.json')['token']==token
        records=[read(p) for p in (R/'cleanup_records').glob('*/DONE.json')]
        freed=sum(x.get('bytes_freed',0) for x in records)
        if freed>=60*2**30:
            write(R/'jobs/scratch_release_060/DONE.json',dict(at=now(),state='complete',role='storage reservation gate, not a model experiment',bytes_freed=freed,resource_token=token));return
        time.sleep(30)
if __name__=='__main__':main()
