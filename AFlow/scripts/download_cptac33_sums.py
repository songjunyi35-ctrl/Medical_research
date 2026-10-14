#!/usr/bin/env python3
"""Download only package636 checksum manifest and bind the frozen CPTAC33 paths."""
from __future__ import annotations
import base64
from datetime import datetime,timezone
import json,os,re,subprocess
from pathlib import Path
import scripts.audit_brats_africa_source_20260906 as api
import scripts.audit_cptac_brats2021_source_20260906 as audit
from scripts import cptac33_protocol as p
SDK=Path('/tmp/aflow-rhuh-sdk-FuoemL');ASCP=SDK/'ibm-aspera-transfer-sdk-linux-amd64-1.1.9/bin/ascp';KEY=SDK/'ibm-public-client-key.pem';CONFIG=SDK/'ibm-aspera-transfer-sdk-linux-amd64-1.1.9/etc/aspera.conf'
ASCP_SHA='f59e7d6afae896f320084a5d441caaece48a1303e5afd014cf885a74bb329efe';KEY_SHA='9983bd387eebf27190c754de76a0ae09d5c39597ee224c3974a6bdb8faec8645';CONFIG_SHA='edfcd31fd75b65f0c10b2bf0d06b983ca15f5ffb6adb8eba52be4817df08fadf'
def main():
 p.docs();rows=p.rows();p.checked(ASCP,ASCP_SHA);p.checked(KEY,KEY_SHA);p.checked(CONFIG,CONFIG_SHA)
 if p.SUMS.exists() or p.SUMS_RECEIPT.exists():raise FileExistsError('sums')
 api.PAGE=audit.PAGE;api.DATAINFO=audit.MAPPING;api.PACKAGE_ID=636
 bearer,page,mapping,package=api.authorize();session=__import__('requests').Session();headers={'Authorization':'Bearer '+bearer}
 root,pages=api.list_directory('/',bearer);items=[x for x in root if x.get('path')=='/RSNA-ASNR-MICCAI-BraTS-2021.sums']
 if pages!=1 or len(items)!=1 or items[0].get('type')!='file' or int(items[0].get('size') or 0)!=49850968:raise ValueError('sums identity')
 reply=session.post(api.BASE+'/api/v5/packages/636/transfer_spec/download',headers=headers,params={'type':'received','transfer_type':'connect'},json={'paths':[{'path':items[0]['path'],'type':'file'}]},timeout=60);reply.raise_for_status();spec=reply.json();sources=[x['source'] for x in spec['paths']]
 if len(sources)!=1 or not sources[0].endswith('/RSNA-ASNR-MICCAI-BraTS-2021.sums'):raise ValueError('spec')
 p.SUMS.parent.mkdir(parents=True,exist_ok=True);command=[str(ASCP),'--mode=recv','--host='+spec['remote_host'],'--user='+spec['remote_user'],'-P',str(spec['ssh_port']),'-O',str(spec['fasp_port']),'-i',str(KEY),'-l','100m','--overwrite=never','-v','-L',str(SDK),'-f',str(CONFIG),sources[0],str(p.SUMS.parent)]
 if 'tags' in spec:command[1:1]=['--tags64='+base64.b64encode(json.dumps(spec['tags']).encode()).decode()]
 env=dict(os.environ,ASPERA_SCP_TOKEN=spec['token'],ASPERA_SCP_COOKIE=spec.get('cookie',''));run=subprocess.run(command,env=env,capture_output=True,text=True,timeout=300)
 if run.returncode!=0 or not p.SUMS.is_file() or p.SUMS.stat().st_size!=49850968:raise RuntimeError('sums transfer')
 parsed={};lengths=set()
 with p.SUMS.open(errors='strict') as f:
  for line in f:
   line=line.rstrip('\n\r');checksum,path=line.split(maxsplit=1);path=path.lstrip('*').lstrip('./').replace('\\','/');checksum=checksum.lower()
   if not re.fullmatch(r'[0-9a-f]{32}|[0-9a-f]{64}',checksum) or path in parsed:raise ValueError('sums row')
   parsed[path]=checksum;lengths.add(len(checksum))
 expected=[]
 for row in rows:
  expected.extend(row[k]['path'].lstrip('/') for k in ('t1ce','seg'))
 selected={path:parsed.get(path) for path in expected};missing=[x for x,v in selected.items() if v is None];empty=[x for x,v in selected.items() if v=='d41d8cd98f00b204e9800998ecf8427e']
 passed=len(parsed)>400000 and len(selected)==66 and not missing and not empty and len({len(v) for v in selected.values()})==1
 receipt={'schema':'aflow.cptac33.sums_receipt','schema_version':1,'created_at_utc':datetime.now(timezone.utc).isoformat(),'status':'passed' if passed else 'failed','file':{'path':str(p.SUMS.relative_to(p.ROOT)),'bytes':p.SUMS.stat().st_size,'sha256':p.sha(p.SUMS)},'rows':len(parsed),'checksum_lengths':sorted(lengths),'selected_rows':len(selected),'selected_checksum_length':len(next(iter(selected.values()))) if selected and not missing else None,'missing':missing,'empty_file_checksums':empty,'selected':selected,'package_id':636,'cohort_sha256':p.COHORT_SHA,'payload_downloaded':False,'arrays':False,'model_inference':False,'runner_sha256':p.sha(Path(__file__))};p.write_new(p.SUMS_RECEIPT,receipt);print(json.dumps({k:v for k,v in receipt.items() if k!='selected'},indent=2,sort_keys=True))
 if not passed:raise RuntimeError('sums gate')
if __name__=='__main__':main()
