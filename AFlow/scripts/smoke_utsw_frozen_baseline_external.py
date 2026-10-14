"""Synthetic-only GPU smoke for both frozen UTSW baselines."""
from datetime import datetime,timezone
import json
from pathlib import Path
import torch
from scripts import utsw_frozen_baseline_external_protocol as p
from scripts.evaluate_utsw_frozen_baseline_external import load_model


@torch.inference_mode()
def main(device_name):
    from scripts.freeze_utsw_frozen_baseline_external import check_implementation
    check_implementation();p.verify_preaccess();device=torch.device(device_name)
    if device.type!="cuda" or not torch.cuda.is_available():raise RuntimeError("CUDA required")
    torch.cuda.set_device(device);torch.manual_seed(20261390);image=torch.randn(2,1,128,128,128,device=device);models={}
    for name in ("full972_kd168","gcsam184"):
        torch.cuda.reset_peak_memory_stats(device);model,payload,pre=load_model(name,device);outs=[]
        with torch.autocast(device_type="cuda",dtype=torch.bfloat16):outs.append(model(image).segmentation_logits.float())
        total=None
        for axes in p.AXES:
            view=torch.flip(image,axes) if axes else image
            with torch.autocast(device_type="cuda",dtype=torch.bfloat16):x=model(view).segmentation_logits.float()
            x=torch.flip(x,axes) if axes else x;total=x.clone() if total is None else total.add_(x)
        outs.append(total/8);models[name]={"file_sha256":pre["models"][name]["file_sha256"],"state_sha256":payload["model_state_sha256"],
            "identity_shape":list(outs[0].shape),"flip8_shape":list(outs[1].shape),"finite":all(bool(torch.isfinite(x).all()) for x in outs),
            "single_example_forwards":18,"peak_gpu_memory_bytes":torch.cuda.max_memory_allocated(device)};del model;torch.cuda.empty_cache()
    gates={"both_models":set(models)==set(p.MODELS),"shape":all(x["identity_shape"]==[2,1,128,128,128] and x["flip8_shape"]==[2,1,128,128,128] for x in models.values()),
        "finite":all(x["finite"] for x in models.values()),"forwards":sum(x["single_example_forwards"] for x in models.values())==36,
        "peak":all(x["peak_gpu_memory_bytes"]<=24*1024**3 for x in models.values())}
    report={"schema":"aflow.utsw_frozen_baseline_external.smoke","created_utc":datetime.now(timezone.utc).isoformat(),"status":"passed" if all(gates.values()) else "failed",
        "gates":gates,"models":models,"synthetic_only":True,"preaccess_sha256":p.sha(p.PREACCESS),"utsw_source_access":False,"utsw_metrics":False,"runner_sha256":p.sha(Path(__file__))}
    p.write_new(p.RECORD/"GPU_SMOKE_RECEIPT.json",report);print(json.dumps(report,indent=2,sort_keys=True))
    if not all(gates.values()):raise RuntimeError("smoke failed")


if __name__=="__main__":
    import argparse;ap=argparse.ArgumentParser();ap.add_argument("--device",default="cuda:0");main(ap.parse_args().device)
