"""Fixed one draw per fit case, p=1 label-only stress, not an accuracy metric."""
from datetime import datetime,timezone
import json
import time
import numpy as np
import torch
from sammed3d.small_rotation import SmallRotation,rotate_pair
from scripts import small_rotation_protocol as p


def main():
    output=p.RECORD/'FIT_ROTATION_STRESS.json'
    if output.exists():raise FileExistsError(output)
    p.source.check_freeze();fit,_=p.check_roles();torch.set_num_threads(2)
    index=json.loads((p.CACHE/'index.json').read_text());mapping={c:i for i,c in enumerate(index['case_ids'])}
    labels=np.memmap(p.CACHE/'labels.u8',mode='r',dtype=np.uint8,shape=(972,128,128,128))
    gen=torch.Generator().manual_seed(2026091050)
    rows=[];start=time.monotonic()
    for row in fit:
        truth=torch.from_numpy(np.array(labels[mapping[row['case_id']]],dtype=np.float32))[None,None]
        angles=(torch.rand(1,3,generator=gen)*2-1)*15
        _,rotated=rotate_pair(truth,truth,angles*torch.pi/180)
        a=int(truth.sum());b=int(rotated.sum());assert a>0
        rows.append({'case_id':row['case_id'],'source_voxels':a,'rotated_voxels':b,'angles_degrees':angles[0].tolist(),'relative_volume_change':(b-a)/a})
    erasures=[r['case_id'] for r in rows if r['rotated_voxels']==0]
    report={'status':'passed' if not erasures else 'failed','created_utc':datetime.now(timezone.utc).isoformat(),'fit_sha256':p.FIT_SHA,'cache_index_sha256':p.sha(p.CACHE/'index.json'),'labels_sha256':p.sha(p.CACHE/'labels.u8'),'code_sha256':p.sha(__file__),'rotation_sha256':p.sha(p.ROOT/'sammed3d/small_rotation.py'),'scans':len(rows),'erasures':erasures,'absolute_relative_volume_change_quantiles':np.quantile([abs(r['relative_volume_change']) for r in rows],[.5,.95,1]).tolist(),'rows':rows,'seconds':time.monotonic()-start,'new_model_forwards':0,'dev_access':False,'test_access':False,'scope':'one deterministic all-selected rotation per fit label, no original shift; not all possible training draws; not model Dice'}
    p.write_new(output,report)
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2));assert not erasures


if __name__=='__main__':main()
