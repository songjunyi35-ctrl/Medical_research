"""Three fixed checkpoints, validation108 only, one common native scoring grid."""
import sys,json,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.segresnet_pretrained_native_common import *
from scripts.train_t1ce_wt_capg3d import BrainCropDataset


def crop_metrics(logits,label):
    d=float(dice_per_case(logits[None],label[None])[0]);flat=int(logits[0].flatten().argmax());h,w=logits.shape[-2:]
    point=[flat//(h*w),(flat%(h*w))//w,flat%w]
    return {'locator_dsc':d,'automatic_point':point,'automatic_point_hits_gt':bool(label[0,*point]>.5)}


def main():
    verify();validation_mode();torch.set_num_threads(4);device=torch.device('cuda:0');torch.cuda.set_device(device)
    if (R/'REFERENCE_PREACCESS.json').exists():raise RuntimeError('reference evaluation already attempted; inspect before rerun')
    audit=json.loads((R/'DATA_AUDIT.json').read_text())
    native=NativeCanvasDataset(VAL_CACHE,expected_split='validation',case_ids=audit['validation_case_ids'])
    grid=BrainCropDataset(CANONICAL['validation'],expected_split='validation',case_ids=audit['validation_case_ids'])
    assert native.case_ids==grid.case_ids and len(native)==108
    for name,(ckpt,digest,_,_) in REFERENCES.items():assert sha(ckpt)==digest,name
    write(R/'REFERENCE_PREACCESS.json',{'utc':now(),'role':'reused fixed validation108/50','freeze_sha256':sha(R/'FREEZE.json'),
          'checkpoints':{k:sha(v[0]) for k,v in REFERENCES.items()},'selection_or_retuning':False,'test_access':False,
          'numeric_mode':'deterministic, benchmark off, matmul TF32 off, bf16 batch2','grid':'native crop, padding excluded'})
    outputs={}
    for name,(ckpt,digest,historical,mode) in REFERENCES.items():
        model=model_for(device);state=torch.load(ckpt,map_location='cpu',weights_only=False);model.load_state_dict(state['model_state_dict'],strict=True);del state
        model.eval();model.requires_grad_(False);rows=[];tick=time.monotonic()
        reported={x['case_id']:x for x in json.loads(historical.read_text())['per_case']}
        with torch.inference_mode():
            for start in range(0,108,2):
                ni=[native[i] for i in range(start,start+2)];items=ni if mode=='native' else [grid[i] for i in range(start,start+2)]
                image=torch.stack([x['image'] for x in items]).to(device)
                with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(image).segmentation_logits.float()
                for k,item in enumerate(items):
                    cid=item['case_id'];assert cid==ni[k]['case_id'];geo=native.geometry[cid]
                    target=ni[k]['label'][(slice(None),)+canvas_slices(geo)].to(device)
                    if mode=='native':
                        cropped=logits[k][(slice(None),)+canvas_slices(geo)];identity=float(dice_per_case(cropped[None],target[None])[0])
                    else:
                        cropped=F.interpolate(logits[k:k+1],size=tuple(geo['extent']),mode='trilinear',align_corners=False)[0]
                        identity=float(dice_per_case(logits[k:k+1],item['label'][None].to(device))[0])
                    row=crop_metrics(cropped,target);row.update(case_id=cid,subject_id=item['subject_id'],
                        own_grid_recomputed_dice=identity,own_grid_historical_dice=reported[cid]['locator_dsc'])
                    rows.append(row)
        error=max(abs(x['own_grid_recomputed_dice']-x['own_grid_historical_dice']) for x in rows)
        if mode=='128' and error>1e-6:raise RuntimeError(f'{name} deterministic128 reproduction error {error}')
        outputs[name]={'checkpoint_sha256':digest,'summary':summary_ci(rows),'per_case':rows,'seconds':time.monotonic()-tick,
                       'own_grid_historical_max_dice_difference':error,'historical_native_scoring_used_different_numeric_mode':mode=='native'}
        del model;torch.cuda.empty_cache()
    write(R/'REFERENCE_NATIVE_VALIDATION.json',{'utc':now(),'status':'passed','grid':'native crop','scans':108,'subjects':50,
          'references':outputs,'test_access':False,'selection':'fixed prior checkpoints; no epoch search'})


if __name__=='__main__':main()
