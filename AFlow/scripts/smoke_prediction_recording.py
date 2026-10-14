"""Old fit8 model only: observer output exactness, no candidate/development readout."""
from datetime import datetime,timezone
import gc
import json
import torch
from sammed3d.prediction_recording import recording_locator
from sammed3d.t1ce_wt_capg3d import AutomaticPromptLocator
from scripts.train_t1ce_wt_capg3d import BrainCropDataset
from scripts import structured_background_protocol as p

REC=p.ROOT/'archive/research_records/2026-09-10_resumed_eight_hour'


class MemoryWriter:
    def __init__(self):self.rows=[]
    def append(self,key,mask,ids):self.rows.append((key,mask.copy(),list(ids)))


@torch.inference_mode()
def main():
    out=REC/'PREDICTION_RECORDING_SMOKE.json'
    if out.exists():raise FileExistsError(out)
    fit,_=p.check_roles();torch.set_num_threads(1);device=torch.device('cuda:0');torch.cuda.set_device(device)
    assert torch.cuda.mem_get_info(device)[0]>10*1024**3
    ids=[r['case_id'] for r in fit[:8]];data=BrainCropDataset(p.CACHE,expected_split='train',case_ids=ids);image=torch.stack([data[i]['image'] for i in range(8)]).to(device)
    report=json.loads((p.WORK/'seed2026090711_control/TRAIN_DONE.json').read_text());p.checked(report['checkpoint']['path'],report['checkpoint']['sha256']);state=torch.load(report['checkpoint']['path'],map_location='cpu',weights_only=False)['model_state_dict']
    outputs=[];writer=MemoryWriter();recorded,counters=recording_locator(AutomaticPromptLocator,writer,ids,['fit_smoke']);torch.cuda.reset_peak_memory_stats(device)
    for cls in (AutomaticPromptLocator,recorded):
        model=cls(center_aware=False).to(device).eval();model.load_state_dict(state,strict=True)
        rng=torch.cuda.get_rng_state(device).clone()
        with torch.autocast(device_type='cuda',dtype=torch.bfloat16):value=model(image).segmentation_logits
        assert torch.equal(rng,torch.cuda.get_rng_state(device));outputs.append(value.float().cpu())
        del model,value;gc.collect();torch.cuda.empty_cache()
    assert torch.equal(outputs[0],outputs[1])
    assert torch.equal(torch.from_numpy(writer.rows[0][1]),outputs[0][:,0]>0)
    peak=torch.cuda.max_memory_allocated(device);assert peak<8*1024**3
    files=['sammed3d/prediction_recording.py','tests/test_prediction_recording.py','scripts/smoke_prediction_recording.py','sammed3d/packed_prediction_masks.py']
    p.write_new(out,{'status':'passed','created_utc':datetime.now(timezone.utc).isoformat(),'exact_output':True,'exact_saved_threshold_mask':True,'cuda_rng_unchanged':True,'observer_forwards':counters['forwards'],'total_fit_forwards':16,'fit_case_ids':ids,'checkpoint':report['checkpoint'],'peak_bytes':peak,'files':{n:p.sha(p.ROOT/n) for n in files},'new_candidate_checkpoint_loaded':False,'development_forwards':0,'test_access':False})
    print('transparent observer GPU fit8 outputs and masks exact',flush=True)


if __name__=='__main__':main()
