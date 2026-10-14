#!/usr/bin/env python3
"""Synthetic-only CUDA smoke for MOTUM-HGG29."""
from __future__ import annotations
import argparse,json
from datetime import datetime,timezone
from pathlib import Path
import torch
import scripts.evaluate_ucsd_ptgbm_external_confirmation as io
from scripts import motum_hgg29_protocol as p
@torch.inference_mode()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--device',default='cuda:0');a=ap.parse_args();out=p.RECORD/'SMOKE_RECEIPT.json'
 if out.exists():raise FileExistsError(out)
 p.docs();device=torch.device(a.device)
 if device.type!='cuda' or not torch.cuda.is_available():raise RuntimeError('CUDA')
 x=torch.randn((1,1,128,128,128),generator=torch.Generator().manual_seed(p.BOOTSTRAP_SEED)).to(device);torch.cuda.reset_peak_memory_stats(device);full,_=io.load_model(p.FULL,p.FULL_SHA,'primary',device);total=None
 for axes in p.AXES:
  v=torch.flip(x,axes) if axes else x
  with torch.autocast(device_type='cuda',dtype=torch.bfloat16):y=full(v).segmentation_logits.float()
  y=torch.flip(y,axes) if axes else y;total=y.clone() if total is None else total+y
 del full;torch.cuda.empty_cache();gc,_=io.load_model(p.GCSAM,p.GCSAM_SHA,'secondary',device)
 with torch.autocast(device_type='cuda',dtype=torch.bfloat16):g=gc(x).segmentation_logits.float()
 if total.shape!=x.shape or g.shape!=x.shape or not torch.isfinite(total).all() or not torch.isfinite(g).all():raise RuntimeError('output')
 p.write_new(out,{'schema':'aflow.motum_hgg29.synthetic_smoke','created_at_utc':datetime.now(timezone.utc).isoformat(),'status':'passed','synthetic_only':True,'motum_source_access':False,'forwards':9,'peak_gpu_memory_bytes':torch.cuda.max_memory_allocated(device),'runner_sha256':p.sha(Path(__file__))});print(json.dumps(json.load(open(out)),indent=2))
if __name__=='__main__':main()
