"""Metadata-only cross-worker status; excludes obsolete sequential-queue failure state."""
import sys,os,json,time,datetime,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'archive/research_records/2026-09-21_segresnet_two_experiments'
OUT=ROOT/'workspace/BraTS/segresnet_t1ce_wt/two_ideas_20260921'
RECS={'semantic_regions':ROOT/'archive/research_records/2026-09-21_segresnet_semantic_regions',
      'finetune_pretrained':ROOT/'archive/research_records/2026-09-21_segresnet_external_pretrain'}


def read(p):return json.loads(p.read_text()) if p.exists() else None
def write(p,x):
    temp=p.with_suffix('.tmp');temp.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');os.replace(temp,p)


def snapshot():
    semantic=read(OUT/'semantic_regions/STATUS.json');pretrain=read(OUT/'pretrain/STATUS.json');fine=read(OUT/'finetune_pretrained/STATUS.json')
    bdone=(RECS['semantic_regions']/'DONE.json').exists();adone=(RECS['finetune_pretrained']/'DONE.json').exists()
    bfailed=not bdone and (RECS['semantic_regions']/'PIPELINE_FAILURE.json').exists()
    afailed=not adone and ((R/'SINGLE_PIPELINE_FAILURE.json').exists() or (R/'MULTIWORKER_dual_FAILURE.json').exists())
    result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'authoritative cross-worker view for hardware amendment003',
        'status':'complete' if adone and bdone else ('ended_with_failure' if (adone or afailed) and (bdone or bfailed) else 'running'),
        'semantic':{'worker':1104883,'training':semantic,'test_and_archive_complete':bdone,'failed':bfailed},
        'external_candidate':{'pretrain_worker':1105022,'pretrain':pretrain,'finetune_worker':1104883,'finetune':fine,
                              'finetune_queue':read(R/'DUAL_FINETUNE_QUEUE_STATUS.json'),'test_and_archive_complete':adone,'failed':afailed},
        'test_tuning_used':False,'historical_pipeline_failure_is_not_cross_worker_status':True}
    write(R/'LIVE_STATUS.json',result)
    return result['status']!='running'


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='launch':
        p=R/'LIVE_STATUS_WATCHER.json'
        if p.exists():raise RuntimeError('watcher already launched')
        with (R/'live_status.log').open('x') as log:
            proc=subprocess.Popen([sys.executable,str(Path(__file__))],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                                  stdin=subprocess.DEVNULL,start_new_session=True)
        write(p,{'pid':proc.pid,'worker':1104883});print(json.dumps({'pid':proc.pid}))
    else:
        while not snapshot():time.sleep(20)
