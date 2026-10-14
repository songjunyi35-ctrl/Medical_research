#!/usr/bin/env python3
"""Train paired uniform-mean8 and batch-CVaR50 top4 continuations from KD200."""
from __future__ import annotations
import argparse, json, math, random, statistics, sys, time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from monai.losses import DiceCELoss

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from sammed3d.t1ce_wt_batch_cvar50 import BATCH_CVAR50_ARMS, aggregate, per_draw_dicece  # noqa:E402
from sammed3d.t1ce_wt_capg3d import AutomaticPromptLocator  # noqa:E402
from sammed3d.t1ce_wt_variable_resolution import VariableResolutionBrainCropDataset  # noqa:E402
from scripts.train_t1ce_wt_capg3d import augment_batch,case_view_metadata,dice_per_case,read_case_ids,sha256_file,subject_id,write_json_atomic  # noqa:E402
from scripts.train_t1ce_wt_plain_iad_kas_continuation import _architecture,_plain_reload_difference,cuda_rng_sha256,state_sha256,tensor_raw_sha256,text_lines_sha256  # noqa:E402

SCHEMA="aflow.sammed3d.t1ce_wt_plain_batch_cvar50_continuation"
CHECKPOINT_SCHEMA="aflow.sammed3d.t1ce_wt_batch_cvar50_checkpoint"
SMOKE_SCHEMA="aflow.sammed3d.t1ce_wt_batch_cvar50_gpu_smoke"
IMPLEMENTATION_RECEIPT_SCHEMA="aflow.sammed3d.t1ce_wt_batch_cvar50_implementation_receipt"
ARMS=("uniform_mean8","batch_cvar50_top4")
KD200_SHA="56f54898ded11a0208e75f8000d57d4d053e80d6dedd760402730ad345c111b7"
TRAIN_MANIFEST_SHA="a520439c5c4b7b6a630a43312c0ce0b2af8d29f671d1e7bea129084bdde74d16"
TRAIN_CACHE_SHA="d26b823a4715d9f3676d75cddabcf630180d48807352c23042be7faf240b622d"
WEIGHTS_SHA="757e463d8e60f96ef6429613f6783fb3aa1014b580932959f9ed354d37072124"
IDEA_SHA="a4ae824d1532981c52d72baf3985f9089bee6c35ebcf5d878a072494d11ecea4"
PREREG_SHA="a0935815c13be95055d0b5a67cb3ce800483bd3ccc2fd30859b9919a6bbcaa70"
AMENDMENT_SHA="81c38d1efd551e79b56e508e5483a5c929cb7b9f8c5bcf4f03833680eb1151ee"
AMENDMENT2_SHA="db188606b1715537a92654a38f5bd0523f7baa1f552fdc3630edff2d4b00fcec"
KAS_HELPER_SHA="289168d96d3494449d9b822513fd759c69aff1b4d1c502fc59b114c1cde62ea8"
EPOCHS=(208,216,224,232); GLOBAL_EPOCHS=tuple(range(201,233)); SEED=20260860
MEMORY_LIMIT=24*1024**3; RUNTIME_LIMIT=45*60; PARAMS=4_700_897
PAIR_FIELDS=("initial_student_state_sha256","first_batch_case_ids","first_batch_case_ids_sha256","first_batch_raw_image_sha256","first_batch_raw_label_sha256","first_batch_augmented_image_sha256","first_batch_augmented_label_sha256","epoch_case_order_sha256","epoch_cuda_rng_before_sha256","epoch_cuda_rng_after_sha256")

def _contract(arm):
    if tuple(BATCH_CVAR50_ARMS)!=ARMS or arm not in ARMS: raise ValueError("frozen arm contract changed")
    return {"arm":arm,"dice":{"include_background":True,"to_onehot_y":False,"sigmoid":True,"softmax":False,"squared_pred":False,"jaccard":False,"smooth_nr":1e-5,"smooth_dr":1e-5,"batch":False,"weight":None,"reduction":"none"},"bce":{"weight":None,"pos_weight":None,"reduction":"none","coefficient":0.1},"aggregation":"mean8" if arm==ARMS[0] else "stable detached top4-of8 mean","topk":8 if arm==ARMS[0] else 4,"teacher_forwards":0,"deploy":"plain final raw logit"}

def _frozen(args):
    for p,h,n in ((args.idea,IDEA_SHA,"IDEA"),(args.preregistration,PREREG_SHA,"PREREG"),(args.amendment,AMENDMENT_SHA,"AMENDMENT"),(args.amendment2,AMENDMENT2_SHA,"AMENDMENT2"),(args.initial_checkpoint,KD200_SHA,"KD200"),(args.train_case_ids,TRAIN_MANIFEST_SHA,"manifest"),(args.cache/"index.json",TRAIN_CACHE_SHA,"cache"),(ROOT/"scripts/train_t1ce_wt_plain_iad_kas_continuation.py",KAS_HELPER_SHA,"helper")):
        if sha256_file(p)!=h: raise ValueError(f"{n} SHA mismatch")
    _contract(args.arm)
    if (args.seed,args.batch_size,args.num_workers)!=(SEED,8,4): raise ValueError("seed/batch/workers differ")
    if (args.start_global_epoch,args.continuation_epochs)!=(200,32): raise ValueError("epoch schedule differs")
    if (args.lr,args.weight_decay,args.max_shift)!=(1e-5,1e-5,8): raise ValueError("optimizer/shift differs")

def _receipt(path):
    if path is None: raise ValueError("formal requires implementation receipt")
    d=json.loads(path.read_text());
    if d.get("schema")!=IMPLEMENTATION_RECEIPT_SCHEMA or d.get("schema_version")!=1: raise ValueError("receipt schema")
    if any(d.get(k) is not True for k in ("synthetic_all_gates_passed","gpu_smoke_all_gates_passed","legacy_parity_all_gates_passed","all_gates_passed")): raise ValueError("receipt gates")
    if any(d.get(k) is not False for k in ("validation_access","outer_selector_access","confirmation_access","test_access")): raise ValueError("receipt access")
    if d.get("implementation",{}).get("trainer_sha256")!=sha256_file(Path(__file__).resolve()): raise ValueError("receipt trainer SHA")
    return sha256_file(path)

def _model(args,device):
    d=torch.load(args.initial_checkpoint,map_location="cpu",weights_only=False)
    if d.get("arm")!="kd" or d.get("global_epoch")!=200: raise ValueError("not KD200")
    a=d["architecture"]; m=AutomaticPromptLocator(center_aware=False,init_filters=int(a["init_filters"]),dropout=float(a["dropout"])).to(device);m.load_state_dict(d["model_state_dict"],strict=True)
    if sum(x.numel() for x in m.state_dict().values())!=PARAMS: raise ValueError("parameter count")
    return m

def _optimizer(m,args):
    ps=list(m.parameters());o=torch.optim.AdamW(ps,lr=args.lr,weight_decay=args.weight_decay,betas=(.9,.999),eps=1e-8,amsgrad=False,maximize=False,foreach=False,fused=False,capturable=False);return o,ps

def _data(args):
    ids=read_case_ids(args.train_case_ids); d=VariableResolutionBrainCropDataset(args.cache,expected_resolution=128,expected_manifest_sha256=TRAIN_MANIFEST_SHA,case_ids=ids)
    if len(d)!=756 or d.index.get("test_access") is not False: raise ValueError("train data")
    w=d.sampling_weights();
    if tensor_raw_sha256(w)!=WEIGHTS_SHA: raise ValueError("weights SHA")
    g=torch.Generator().manual_seed(args.seed);l=DataLoader(d,batch_size=8,sampler=WeightedRandomSampler(w,num_samples=756,replacement=True,generator=g),num_workers=4,pin_memory=True,persistent_workers=True,drop_last=True)
    if len(l)!=94: raise ValueError("steps")
    return d,w,l

def _pair(batch,ri,rl,i,l):
    ids=[str(x) for x in batch["case_id"]]
    return {"first_batch_case_ids":ids,"first_batch_case_ids_sha256":text_lines_sha256(ids),"first_batch_raw_image_sha256":tensor_raw_sha256(ri),"first_batch_raw_label_sha256":tensor_raw_sha256(rl),"first_batch_augmented_image_sha256":tensor_raw_sha256(i),"first_batch_augmented_label_sha256":tensor_raw_sha256(l)}

def _loss(model,image,target,arm):
    with torch.autocast(device_type="cuda",dtype=torch.bfloat16):
        z=model(image).segmentation_logits
        per=per_draw_dicece(z,target)
    agg=aggregate(per.per_draw_loss,arm=arm)
    if per.per_draw_loss.shape!=(8,) or agg.loss.ndim!=0: raise RuntimeError("loss shapes")
    if not bool(torch.isfinite(per.per_draw_loss).all()) or not bool(torch.isfinite(agg.loss)): raise RuntimeError("nonfinite loss")
    return z,per,agg

def _checkpoint(args,m,ge,diag,pair,reload,receipt_sha):
    return {"schema":CHECKPOINT_SCHEMA,"schema_version":1,"arm":args.arm,"global_epoch":ge,"continuation_epoch":ge-200,"resolution":128,"learning_rate":args.lr,"model_state_dict":OrderedDict((n,x.detach().cpu().clone()) for n,x in m.state_dict().items()),"architecture":_architecture(),"batch_cvar50":_contract(args.arm),"loss_diagnostics":diag,"pairing":pair,"checkpoint_reload_max_abs_difference":reload,"stripped_deploy_max_abs_difference":0.0,"start_checkpoint_sha256":KD200_SHA,"runner_sha256":sha256_file(Path(__file__).resolve()),"implementation_receipt_sha256":receipt_sha,"idea_sha256":IDEA_SHA,"preregistration_sha256":PREREG_SHA,"amendment_sha256":AMENDMENT_SHA,"amendment2_sha256":AMENDMENT2_SHA,"start_checkpoint_historical_validation_access":True,"start_checkpoint_historical_test_access_after_selection":True,"validation_access":False,"outer_selector_access":False,"confirmation_access":False,"test_access":False}

def _smoke(args,m,o,ps,loader,init,device):
    write_json_atomic(args.output_dir/"SMOKE_STARTED.json",{"schema":SMOKE_SCHEMA+".started","arm":args.arm,"validation_access":False,"outer_selector_access":False,"confirmation_access":False,"test_access":False})
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED);torch.cuda.reset_peak_memory_stats(device);m.train();it=iter(loader);times=[];first=None;audit=None;legacy_parity=None;latest={};start=time.time()
    for step in range(4):
        b=next(it);ri=b["image"].to(device,dtype=torch.float32);rl=b["label"].to(device,dtype=torch.float32);im,la=augment_batch(ri.clone(),rl.clone(),max_shift=8)
        if first is None:first={"initial_student_state_sha256":init,**_pair(b,ri,rl,im,la)}
        o.zero_grad(set_to_none=True);torch.cuda.synchronize(device);t=time.time();z,per,agg=_loss(m,im,la,args.arm)
        if audit is None:
            expected=torch.arange(8,device=z.device) if args.arm==ARMS[0] else torch.argsort(per.per_draw_loss.detach(),descending=True,stable=True)[:4]
            grad=torch.autograd.grad(agg.loss,z,retain_graph=True)[0];audit={"selected_indices":agg.selected_indices.detach().cpu().tolist(),"manual_indices":expected.cpu().tolist(),"indices_match":torch.equal(agg.selected_indices,expected),"gradient_finite":bool(torch.isfinite(grad).all()),"gradient_nonzero":float(grad.float().norm())>0,"unselected_gradient_zero":True if args.arm==ARMS[0] else bool((grad.flatten(1)[~agg.selected_mask].abs().sum(dim=1)==0).all())}
            if args.arm==ARMS[0]:
                with torch.autocast(device_type="cuda",dtype=torch.bfloat16):
                    legacy=DiceCELoss(sigmoid=True,squared_pred=False,lambda_dice=1.0,lambda_ce=0.1)(z,la)
                legacy_grad=torch.autograd.grad(legacy,z,retain_graph=True)[0]
                legacy_parity={"loss_absolute_difference":float((legacy-agg.loss).detach().abs()),"raw_logit_gradient_max_absolute_difference":float((legacy_grad-grad).detach().abs().max()),"loss_tolerance":1e-7,"gradient_tolerance":1e-7}
        agg.loss.backward();gn=nn.utils.clip_grad_norm_(ps,5,error_if_nonfinite=True);o.step();torch.cuda.synchronize(device)
        if step:times.append(time.time()-t)
        latest={"loss":float(agg.loss.detach()),"per_draw":per.per_draw_loss.detach().cpu().tolist(),"grad_norm":float(gn),"train_dsc":float(dice_per_case(z.detach().float(),la).mean())}
    peak=int(torch.cuda.max_memory_allocated(device));est=statistics.median(times)*94*32;legacy_passed=args.arm!=ARMS[0] or (legacy_parity is not None and legacy_parity["loss_absolute_difference"]<=1e-7 and legacy_parity["raw_logit_gradient_max_absolute_difference"]<=1e-7);gates={"finite":all(math.isfinite(x) for x in [latest["loss"],latest["grad_norm"]]+latest["per_draw"]),"topk_and_gradient":all(audit[k] for k in ("indices_match","gradient_finite","gradient_nonzero","unselected_gradient_zero")),"legacy_gpu_loss_and_logit_gradient_parity":legacy_passed,"memory":peak<=MEMORY_LIMIT,"runtime":est<=RUNTIME_LIMIT}
    r={"schema":SMOKE_SCHEMA,"schema_version":1,"status":"complete" if all(gates.values()) else "failed_gate","arm":args.arm,"pairing":first,"audit":audit,"legacy_parity":legacy_parity,"losses":latest,"median_step_seconds":statistics.median(times),"estimated_formal_seconds":est,"peak_gpu_memory_bytes":peak,"gates":gates,"all_gates_passed":all(gates.values()),"runner_sha256":sha256_file(Path(__file__).resolve()),"validation_access":False,"outer_selector_access":False,"confirmation_access":False,"test_access":False};write_json_atomic(args.output_dir/"GPU_SMOKE.json",r)
    if not r["all_gates_passed"]:raise RuntimeError(f"smoke failed {gates}")
    return r

def _formal(args,m,o,ps,data,w,loader,init,receipt_sha,device):
    write_json_atomic(args.output_dir/"STARTED.json",{"schema":SCHEMA,"schema_version":1,"status":"before_optimizer_step","arm":args.arm,"runner_sha256":sha256_file(Path(__file__).resolve()),"implementation_receipt_sha256":receipt_sha,"idea_sha256":IDEA_SHA,"preregistration_sha256":PREREG_SHA,"amendment_sha256":AMENDMENT_SHA,"amendment2_sha256":AMENDMENT2_SHA,"validation_access":False,"outer_selector_access":False,"confirmation_access":False,"test_access":False})
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED);torch.cuda.reset_peak_memory_stats(device);m.train();hist=[];ck={};orders={};before={};after={};first=None;started=time.time();last=None
    for ce in range(1,33):
        ge=200+ce;cases=[];losses=[];means=[];selected=[];dsc=[];gns=[]
        for step,b in enumerate(loader):
            ri=b["image"].to(device,dtype=torch.float32);rl=b["label"].to(device,dtype=torch.float32)
            if step==0:before[str(ge)]=cuda_rng_sha256(device)
            im,la=augment_batch(ri.clone(),rl.clone(),max_shift=8)
            if first is None:first={"initial_student_state_sha256":init,**_pair(b,ri,rl,im,la)}
            cases.extend(str(x) for x in b["case_id"]);o.zero_grad(set_to_none=True);z,per,agg=_loss(m,im,la,args.arm);agg.loss.backward();gn=nn.utils.clip_grad_norm_(ps,5,error_if_nonfinite=True);o.step();losses.append(float(agg.loss.detach()));means.append(float(per.per_draw_loss.mean().detach()));selected.extend(agg.selected_indices.detach().cpu().tolist());dsc.extend(dice_per_case(z.detach().float(),la).cpu().tolist());gns.append(float(gn));last=im.detach()
        orders[str(ge)]=text_lines_sha256(cases);after[str(ge)]=cuda_rng_sha256(device);diag={"train_aggregated_loss":float(np.mean(losses)),"train_uniform_mean_loss":float(np.mean(means)),"train_dsc":float(np.mean(dsc)),"grad_norm_mean":float(np.mean(gns)),"selected_draw_histogram":[selected.count(i) for i in range(8)]};hist.append({"global_epoch":ge,"continuation_epoch":ce,"learning_rate":1e-5,"optimizer_steps":94,**diag,"elapsed_seconds":time.time()-started});pair={**first,"epoch_case_order_sha256":dict(orders),"epoch_cuda_rng_before_sha256":dict(before),"epoch_cuda_rng_after_sha256":dict(after)}
        if ge in EPOCHS:
            m.eval();
            with torch.inference_mode(),torch.autocast(device_type="cuda",dtype=torch.bfloat16):ref=m(last).segmentation_logits.detach()
            m.train();state=OrderedDict((n,x.detach().cpu().clone()) for n,x in m.state_dict().items());reload=_plain_reload_difference(state,_architecture(),ref,last,device)
            if reload!=0:raise RuntimeError("reload mismatch")
            val=_checkpoint(args,m,ge,diag,pair,reload,receipt_sha);p=args.output_dir/f"plain_128_{args.arm}_global_epoch{ge}.pth";torch.save(val,p);ck[str(ge)]={"path":str(p.resolve()),"sha256":sha256_file(p),"checkpoint_reload_max_abs_difference":reload,"stripped_deploy_max_abs_difference":0.0}
        write_json_atomic(args.output_dir/"progress.json",{"history":hist,"checkpoints":ck,"pairing":pair})
    peak=int(torch.cuda.max_memory_allocated(device));elapsed=time.time()-started;gates={"memory":peak<=MEMORY_LIMIT,"runtime_per_arm":elapsed<=RUNTIME_LIMIT,"checkpoints":set(ck)=={str(x) for x in EPOCHS},"reload":all(x["checkpoint_reload_max_abs_difference"]==0 for x in ck.values()),"history":len(hist)==32};status="complete" if all(gates.values()) else "failed_gate"
    r={"schema":SCHEMA,"schema_version":1,"status":status,"arm":args.arm,"history":hist,"checkpoints":ck,"pairing":pair,"formal_self_gates":gates,"all_self_gates_passed":all(gates.values()),"reload_max_abs_difference":0.0 if gates["reload"] else None,"stripped_deploy_max_abs_difference":0.0,"peak_gpu_memory_bytes":peak,"peak_gpu_reserved_bytes":int(torch.cuda.max_memory_reserved(device)),"elapsed_seconds":elapsed,"runner_sha256":sha256_file(Path(__file__).resolve()),"implementation_receipt_sha256":receipt_sha,"idea_sha256":IDEA_SHA,"preregistration_sha256":PREREG_SHA,"amendment_sha256":AMENDMENT_SHA,"amendment2_sha256":AMENDMENT2_SHA,"batch_cvar50":_contract(args.arm),"data_role":{"train":case_view_metadata(args.train_case_ids),"cases":756,"subjects":346,"cache_index_sha256":TRAIN_CACHE_SHA},"sampling_contract":{"generator_seed":SEED,"weighted_sampler_num_samples":756,"drop_last":True,"optimizer_consumed_draws":752,"dropped_draws":4,"epoch_case_order_scope":"752 consumed optimizer draws","weights_sha256":tensor_raw_sha256(w),"steps":94},"start_checkpoint_sha256":KD200_SHA,"start_checkpoint_historical_validation_access":True,"start_checkpoint_historical_test_access_after_selection":True,"validation_access":False,"outer_selector_access":False,"confirmation_access":False,"test_access":False};write_json_atomic(args.output_dir/"TRAIN_DONE.json",r)
    if status!="complete":raise RuntimeError(f"formal gates failed {gates}")
    return r

def run(args):
    _frozen(args)
    if args.output_dir.exists():raise FileExistsError("output reuse")
    args.output_dir.mkdir(parents=True);device=torch.device(args.device)
    if device.type!="cuda" or not torch.cuda.is_available():raise RuntimeError("CUDA required")
    torch.cuda.set_device(device);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;m=_model(args,device);init=state_sha256(m.state_dict());o,ps=_optimizer(m,args);d,w,l=_data(args)
    return _smoke(args,m,o,ps,l,init,device) if args.mode=="smoke" else _formal(args,m,o,ps,d,w,l,init,_receipt(args.implementation_receipt),device)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--arm",choices=ARMS,required=True);p.add_argument("--mode",choices=("smoke","formal"),required=True);p.add_argument("--cache",type=Path,required=True);p.add_argument("--train-case-ids",type=Path,required=True);p.add_argument("--initial-checkpoint",type=Path,required=True);p.add_argument("--idea",type=Path,required=True);p.add_argument("--preregistration",type=Path,required=True);p.add_argument("--amendment",type=Path,required=True);p.add_argument("--amendment2",type=Path,required=True);p.add_argument("--implementation-receipt",type=Path);p.add_argument("--output-dir",type=Path,required=True);p.add_argument("--device",default="cuda:0");p.add_argument("--seed",type=int,default=SEED);p.add_argument("--batch-size",type=int,default=8);p.add_argument("--num-workers",type=int,default=4);p.add_argument("--start-global-epoch",type=int,default=200);p.add_argument("--continuation-epochs",type=int,default=32);p.add_argument("--lr",type=float,default=1e-5);p.add_argument("--weight-decay",type=float,default=1e-5);p.add_argument("--max-shift",type=int,default=8);a=p.parse_args();print(json.dumps(run(a),default=str)[:1000])
if __name__=="__main__":main()
