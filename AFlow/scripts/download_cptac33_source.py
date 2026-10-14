#!/usr/bin/env python3
"""Dual-download 66 frozen CPTAC33 files and verify official MD5 plus local SHA."""
from __future__ import annotations
import base64,hashlib,json,os,subprocess
from datetime import datetime,timezone
from pathlib import Path
import requests
import scripts.audit_brats_africa_source_20260906 as api
import scripts.audit_cptac_brats2021_source_20260906 as audit
from scripts import cptac33_protocol as p
SDK=Path('/tmp/aflow-rhuh-sdk-FuoemL');ASCP=SDK/'ibm-aspera-transfer-sdk-linux-amd64-1.1.9/bin/ascp';KEY=SDK/'ibm-public-client-key.pem';CONFIG=SDK/'ibm-aspera-transfer-sdk-linux-amd64-1.1.9/etc/aspera.conf'
ASCP_SHA='f59e7d6afae896f320084a5d441caaece48a1303e5afd014cf885a74bb329efe';KEY_SHA='9983bd387eebf27190c754de76a0ae09d5c39597ee224c3974a6bdb8faec8645';CONFIG_SHA='edfcd31fd75b65f0c10b2bf0d06b983ca15f5ffb6adb8eba52be4817df08fadf'
SUFFIX='/RSNA-ASNR-MICCAI-BraTS-2021/BraTS2021_TrainingSet/CPTAC-GBM/'
def digest(path,algorithm):
 h=hashlib.new(algorithm)
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def expected(rows):
 d={}
 for row in rows:
  for k in ('t1ce','seg'):
   path=row[k]['path'];
   if SUFFIX not in path:raise ValueError('path')
   d[path.split(SUFFIX,1)[1]]=int(row[k]['size'])
 if len(d)!=p.FILES or sum(d.values())!=p.BYTES:raise ValueError('expected')
 return d
def transfer(name,rows,sizes,official):
 api.PAGE=audit.PAGE;api.DATAINFO=audit.MAPPING;api.PACKAGE_ID=636;bearer,*_=api.authorize();session=requests.Session();headers={'Authorization':'Bearer '+bearer};requested=sorted(row[k]['path'] for row in rows for k in ('t1ce','seg'))
 reply=session.post(api.BASE+'/api/v5/packages/636/transfer_spec/download',headers=headers,params={'type':'received','transfer_type':'connect'},json={'paths':[{'path':x,'type':'file'} for x in requested]},timeout=90)
 if reply.status_code!=201:raise RuntimeError('spec '+str(reply.status_code))
 spec=reply.json();sources=[x['source'] for x in spec['paths']];relative=[x.split(SUFFIX,1)[1] if SUFFIX in x else '' for x in sources]
 if len(sources)!=66 or set(relative)!=set(sizes):raise ValueError('spec paths')
 prefixes={x.split(SUFFIX,1)[0]+SUFFIX.rstrip('/') for x in sources}
 if len(prefixes)!=1:raise ValueError('root')
 root=p.SOURCE/name;root.mkdir(parents=True,exist_ok=False);logs=p.SOURCE/'logs'/name;logs.mkdir(parents=True,exist_ok=False)
 command=[str(ASCP),'--mode=recv','--host='+spec['remote_host'],'--user='+spec['remote_user'],'-P',str(spec['ssh_port']),'-O',str(spec['fasp_port']),'-i',str(KEY),'-l','200m','--overwrite=never','--src-base='+prefixes.pop(),'-v','-L',str(logs),'-f',str(CONFIG),*sources,str(root)]
 if 'tags' in spec:command[1:1]=['--tags64='+base64.b64encode(json.dumps(spec['tags']).encode()).decode()]
 env=dict(os.environ,ASPERA_SCP_TOKEN=spec['token'],ASPERA_SCP_COOKIE=spec.get('cookie',''));run=subprocess.run(command,env=env,capture_output=True,text=True,timeout=600)
 files=[]
 for rel in sorted(sizes):
  path=root/rel;offpath=(SUFFIX+rel).lstrip('/');item={'relative_path':rel,'path':str(path),'exists':path.is_file(),'expected_bytes':sizes[rel],'official_md5':official[offpath]}
  if path.is_file():item.update(bytes=path.stat().st_size,size_exact=path.stat().st_size==sizes[rel],md5=digest(path,'md5'),sha256=digest(path,'sha256'))
  item['official_md5_exact']=item.get('md5')==item['official_md5'];files.append(item)
 return {'copy':name,'returncode':run.returncode,'message':(run.stdout+run.stderr)[-1200:].replace(spec['token'],'[REDACTED]'),'files':files,'all_exact':run.returncode==0 and all(x.get('size_exact') and x['official_md5_exact'] for x in files)}
def main():
 p.docs();rows=p.rows();p.checked(ASCP,ASCP_SHA);p.checked(KEY,KEY_SHA);p.checked(CONFIG,CONFIG_SHA)
 if not p.SOURCE_PREACCESS.exists():raise FileNotFoundError('preaccess')
 pre=json.load(open(p.SOURCE_PREACCESS));sums=json.load(open(p.SUMS_RECEIPT))
 if pre.get('status')!='frozen_before_patient_payload_and_model' or pre['access']['payload'] is not False or sums.get('status')!='passed':raise ValueError('preaccess/sums')
 if p.SOURCE.exists() or p.SOURCE_RECEIPT.exists():raise FileExistsError('source')
 sizes=expected(rows);official=sums['selected'];p.SOURCE.mkdir();copies=[transfer(x,rows,sizes,official) for x in ('copy_a','copy_b')];maps=[{x['relative_path']:x for x in c['files']} for c in copies];comparisons=[]
 for rel in sorted(sizes):
  a,b=maps[0][rel],maps[1][rel];exact=bool(a.get('official_md5_exact') and b.get('official_md5_exact') and a.get('sha256')==b.get('sha256') and a.get('bytes')==b.get('bytes'));comparisons.append({'relative_path':rel,'bytes':a.get('bytes'),'md5':a.get('md5'),'sha256':a.get('sha256'),'official_and_copy_exact':exact})
 passed=all(c['all_exact'] for c in copies) and all(x['official_and_copy_exact'] for x in comparisons);report={'schema':'aflow.cptac33.source_download','schema_version':1,'created_at_utc':datetime.now(timezone.utc).isoformat(),'status':'passed' if passed else 'failed','files_per_copy':66,'bytes_per_copy':p.BYTES,'official_sums_sha256':p.sha(p.SUMS),'copies':copies,'comparisons':comparisons,'exact_files':sum(x['official_and_copy_exact'] for x in comparisons),'preaccess_sha256':p.sha(p.SOURCE_PREACCESS),'arrays':False,'label_values':False,'model_inference':False,'runner_sha256':p.sha(Path(__file__))};p.write_new(p.SOURCE_RECEIPT,report);print(json.dumps({'status':report['status'],'exact_files':report['exact_files'],'receipt_sha256':p.sha(p.SOURCE_RECEIPT)},indent=2))
 if not passed:raise RuntimeError('source gate')
if __name__=='__main__':main()
