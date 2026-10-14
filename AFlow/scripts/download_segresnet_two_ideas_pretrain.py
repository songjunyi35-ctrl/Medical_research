"""Download the frozen 334 OpenNeuro ds007045 T1ce/label pairs (versioned S3 URLs) to worker scratch with SHA-256 verification.

CC0 public data; institution-disjoint from the BraTS2024 seven sites; NOT an evaluated cohort of this project
(its external panel was closed before any model forward on 2026-09-06), so it can be used for pretraining.
"""
import sys,os,json,time,hashlib,traceback,urllib.request,concurrent.futures as cf
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.segresnet_two_ideas_common import *


def fetch(item,dest):
    dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists() and dest.stat().st_size==item['bytes'] and sha(dest)==item['sha256']:return 'cached'
    for attempt in range(4):
        try:
            with urllib.request.urlopen(item['url'],timeout=120) as r,open(dest,'wb') as f:
                while True:
                    b=r.read(1<<20)
                    if not b:break
                    f.write(b)
            if dest.stat().st_size!=item['bytes']:raise IOError(f'size {dest.stat().st_size}!={item["bytes"]}')
            if sha(dest)!=item['sha256']:raise IOError('sha256 mismatch')
            return 'downloaded'
        except Exception as exc:
            err=exc;time.sleep(2+3*attempt)
    raise RuntimeError(f'{item["path"]}: {err!r}')


def main():
    verify();inv,cases=selected_cases()
    if (R/'DOWNLOAD_START.json').exists():raise FileExistsError('download already attempted; inspect receipt before retry')
    import shutil
    if shutil.disk_usage('/tmp').free < 8*(1<<30):raise RuntimeError('insufficient scratch disk')
    RAW_DIR.mkdir(parents=True,exist_ok=True)
    write(R/'DOWNLOAD_START.json',{'utc':now(),'dataset':inv['dataset'],'snapshot':inv['snapshot_id'],'license':inv['license'],'cases':334,'bytes_expected':sum(c['image']['bytes']+c['label']['bytes'] for c in cases),'destination':str(RAW_DIR),'role':'pretraining source only; never an evaluated cohort of this project','test_access':False})
    tick=time.monotonic();jobs=[]
    for c in cases:
        jobs.append((c['image'],RAW_DIR/c['subject_id']/'t1ce.nii.gz'));jobs.append((c['label'],RAW_DIR/c['subject_id']/'seg.nii.gz'))
    counts={'downloaded':0,'cached':0}
    with cf.ThreadPoolExecutor(16) as ex:
        for k,status in enumerate(ex.map(lambda j:fetch(*j),jobs),1):
            counts[status]+=1
            if k%100==0 or k==len(jobs):print(json.dumps({'utc':now(),'done':k,'of':len(jobs),**counts,'seconds':time.monotonic()-tick}),flush=True)
    manifest=[{'subject_id':c['subject_id'],'case_id':cache_case_id(c['subject_id']),'site_code':c['site_code'],'image_path':str(RAW_DIR/c['subject_id']/'t1ce.nii.gz'),'label_path':str(RAW_DIR/c['subject_id']/'seg.nii.gz'),'image_sha256':c['image']['sha256'],'label_sha256':c['label']['sha256'],'image_url':c['image']['url'],'label_url':c['label']['url']} for c in cases]
    (R/'PRETRAIN_MANIFEST.jsonl').write_text(''.join(json.dumps(m)+'\n' for m in manifest))
    write(R/'DOWNLOAD_RECEIPT.json',{'utc':now(),'status':'passed','files':len(jobs),**counts,'bytes':sum(j[0]['bytes'] for j in jobs),'all_sha256_verified':True,'seconds':time.monotonic()-tick,'manifest':str((R/'PRETRAIN_MANIFEST.jsonl').relative_to(ROOT)),'manifest_sha256':sha(R/'PRETRAIN_MANIFEST.jsonl'),'site_counts':{s:sum(1 for c in cases if c['site_code']==s) for s in sorted({c['site_code'] for c in cases})},'test_access':False})
    print(json.dumps(counts),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as exc:write(R/'FAILURE_download.json',{'utc':now(),'error':repr(exc),'traceback':traceback.format_exc()});raise
