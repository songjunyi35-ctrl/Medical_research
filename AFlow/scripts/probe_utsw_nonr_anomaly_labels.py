"""Dual-download/read only 35 anomalous cases' non-r manual labels."""
import base64,hashlib,json,os
from datetime import datetime,timezone
from pathlib import Path
import subprocess
import nibabel as nib
import numpy as np
from scripts import download_utsw_no_n4_paired_external as shared
from scripts import utsw_frozen_baseline_external_protocol as p

IDS=('BT0021','BT0041','BT0048','BT0050','BT0055','BT0061','BT0063','BT0081','BT0103','BT0130','BT0150','BT0155','BT0156','BT0157','BT0158','BT0166','BT0181','BT0182','BT0183','BT0189','BT0194','BT0199','BT0220','BT0224','BT0227','BT0251','BT0252','BT0270','BT0289','BT0307','BT0318','BT0336','BT0653','BT0747','BT0840')
ROOT=Path('/tmp/aflow-utsw-nonr-anomaly-labels-20260906');TOTAL=556450
def fsha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def expected():
    inv=json.loads(p.source_contract.INVENTORY.read_text())['files'];by={x['path']:x for x in inv};out={}
    for sid in IDS:
        path=f'/UTSW-Glioma/{sid}/tumorseg_manual_correction.nii.gz';out[path]=int(by[path]['size'])
    text=''.join(f'{k}\t{out[k]}\n' for k in sorted(out))
    if len(out)!=35 or sum(out.values())!=TOTAL or hashlib.sha256(text.encode()).hexdigest()!='d83a940bc0e8a142be08da56c1f6ea110bf95f311786b4b80584e147340897e3':raise ValueError('frozen paths')
    return out
def transfer(name,paths):
    session,headers=shared.authenticate();reply=session.post(shared.BASE+'/api/v5/packages/1265/transfer_spec/download',headers=headers,
        params={'type':'received','transfer_type':'connect'},json={'paths':[{'path':x,'type':'file'} for x in sorted(paths)]},timeout=120)
    if reply.status_code!=201:
        raise RuntimeError('spec')
    spec=reply.json()
    sources=[x['source'] for x in spec['paths']]
    suffix='/UTSW-Glioma/'
    if len(sources)!=35 or {x.split(suffix,1)[1] for x in sources}!={x.split(suffix,1)[1] for x in paths}:raise ValueError('spec paths')
    prefixes={x.split(suffix,1)[0]+'/UTSW-Glioma' for x in sources}
    if len(prefixes)!=1:raise ValueError('source root')
    dest=ROOT/name;dest.mkdir(parents=True);log=ROOT/'logs'/name;log.mkdir(parents=True)
    cmd=[str(shared.ASCP),'--mode=recv','--host='+spec['remote_host'],'--user='+spec['remote_user'],'-P',str(spec['ssh_port']),'-O',str(spec['fasp_port']),'-i',str(shared.KEY),
        '-l','50m','--overwrite=never','--src-base='+prefixes.pop(),'-L',str(log),'-f',str(shared.CONFIG),*sources,str(dest)]
    if 'tags' in spec:cmd[1:1]=['--tags64='+base64.b64encode(json.dumps(spec['tags']).encode()).decode()]
    env=dict(os.environ,ASPERA_SCP_TOKEN=spec['token'],ASPERA_SCP_COOKIE=spec.get('cookie',''));run=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=300)
    if run.returncode!=0:raise RuntimeError('ascp '+str(run.returncode))
    return dest
def main():
    if ROOT.exists():raise FileExistsError(ROOT)
    paths=expected();a=transfer('copy_a',paths);b=transfer('copy_b',paths);files=[];rdiag=json.loads((p.RECORD/'LABEL_DOMAIN_DIAGNOSTIC_AMENDMENT_01.json').read_text());rmap={x['subject_id']:x for x in rdiag['rows']}
    for official,size in sorted(paths.items()):
        rel=official.split('/UTSW-Glioma/',1)[1];pa=a/rel;pb=b/rel
        if not pa.is_file() or not pb.is_file() or pa.stat().st_size!=size or pb.stat().st_size!=size or fsha(pa)!=fsha(pb):raise RuntimeError('dual '+rel)
        nii=nib.load(str(pa),mmap=True);arr=np.asarray(nii.dataobj,dtype=np.float32);values=[float(x) for x in np.unique(arr)]
        sid=rel.split('/',1)[0];files.append({'subject_id':sid,'bytes':size,'sha256':fsha(pa),'nonr_shape':list(nii.shape),'nonr_affine':np.asarray(nii.affine).tolist(),
            'nonr_values':values,'r_values':rmap[sid]['values'],'finite':bool(np.isfinite(arr).all()),'integer_exact':bool(np.array_equal(arr,np.rint(arr)))})
    canonical=all(x['finite'] and x['integer_exact'] and set(x['nonr_values'])<={0.,1.,2.,4.} for x in files)
    report={'schema':'aflow.utsw_nonr_anomaly_label_probe','created_utc':datetime.now(timezone.utc).isoformat(),'status':'supports_r_resampling_artifact' if canonical else 'nonr_domain_also_unresolved',
        'cases':35,'bytes_per_copy':TOTAL,'dual_exact':True,'all_nonr_canonical_fets_domain':canonical,'nonr_union':sorted({v for x in files for v in x['nonr_values']}),
        'r_union':sorted({v for x in files for v in x['r_values']}),'files':files,'image_arrays':False,'resampling':False,'model':False,'metrics':False,'runner_sha256':p.sha(Path(__file__))}
    out=p.RECORD/'NONR_ANOMALY_PROBE.json';p.write_new(out,report);print(json.dumps({k:v for k,v in report.items() if k!='files'},indent=2))
if __name__=='__main__':main()
