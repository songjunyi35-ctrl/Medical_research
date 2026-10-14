#!/usr/bin/env python3
"""Paired dual-LR continuation for same-RF and DMRFB bottleneck adapters."""
from __future__ import annotations
import argparse,json,math,random,statistics,sys,time
from collections import OrderedDict
from datetime import datetime,timezone
from pathlib import Path
import numpy as np, torch
from monai.losses import DiceCELoss
from torch import nn
from torch.utils.data import DataLoader,WeightedRandomSampler
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from sammed3d.t1ce_wt_dmrfb_adapter import DMRFB_ARMS,DMRFBAdapterLocator  # noqa:E402
from sammed3d.t1ce_wt_capg3d import AutomaticPromptLocator  # noqa:E402
from sammed3d.t1ce_wt_variable_resolution import VariableResolutionBrainCropDataset  # noqa:E402
from scripts.train_t1ce_wt_capg3d import augment_batch,case_view_metadata,dice_per_case,read_case_ids,sha256_file,subject_id,write_json_atomic  # noqa:E402
from scripts.train_t1ce_wt_plain_iad_kas_continuation import cuda_rng_sha256,state_sha256,tensor_raw_sha256,text_lines_sha256  # noqa:E402
SCHEMA="aflow.sammed3d.t1ce_wt_plain_dmrfb_adapter_continuation";CK="aflow.sammed3d.t1ce_wt_dmrfb_adapter_checkpoint";SMOKE="aflow.sammed3d.t1ce_wt_dmrfb_adapter_gpu_smoke";RECEIPT="aflow.sammed3d.t1ce_wt_dmrfb_adapter_implementation_receipt";ARMS=("same_rf_control","dmrfb_dilation2")
KD="56f54898ded11a0208e75f8000d57d4d053e80d6dedd760402730ad345c111b7";MAN="a520439c5c4b7b6a630a43312c0ce0b2af8d29f671d1e7bea129084bdde74d16";CACHE="d26b823a4715d9f3676d75cddabcf630180d48807352c23042be7faf240b622d";WEIGHTS="757e463d8e60f96ef6429613f6783fb3aa1014b580932959f9ed354d37072124";IDEA="5e9bf8db4815535b6b259c111249a8056ad37b8ede79633468b90610197b6a8d";PREREG="46a14d76c719671777c6819e6eae51aad0dff49c6e94047bc252f10eec629773";SCREEN="09a4e0afb9b194ed938cea364c187628796205d49d8ac066e92b7f024bff35a5";AMENDMENT="8eb56acdefca704df3e939da7b2a426cf61efda028ed0b96fece6fbfba91fe88"
EPOCHS=(208,216,224,232);GLOBAL=tuple(range(201,233));LIMIT=24*1024**3;SEED=20260900;FIELDS=("initial_student_state_sha256","initial_adapter_state_sha256","first_batch_case_ids","first_batch_case_ids_sha256","first_batch_raw_image_sha256","first_batch_raw_label_sha256","first_batch_augmented_image_sha256","first_batch_augmented_label_sha256","epoch_case_order_sha256","epoch_cuda_rng_before_sha256","epoch_cuda_rng_after_sha256")
def contract(arm):return {"arm":arm,"branch3_dilation":1 if arm==ARMS[0] else 2,"branch3_padding":1 if arm==ARMS[0] else 2,"adapter_parameters":57088,"deploy_parameters":4757985,"base_lr":1e-5,"base_weight_decay":1e-5,"adapter_lr":1e-3,"adapter_weight_decay":.08,"teacher_forwards":0}
def frozen(a):
 for p,h in ((a.idea,IDEA),(a.preregistration,PREREG),(a.feasibility_screen,SCREEN),(a.amendment,AMENDMENT),(a.initial_checkpoint,KD),(a.train_case_ids,MAN),(a.cache/"index.json",CACHE)):
  if sha256_file(p)!=h:raise ValueError(f"frozen SHA {p}")
 if tuple(DMRFB_ARMS)!=ARMS or a.arm not in ARMS or (a.seed,a.batch_size,a.num_workers)!=(SEED,8,4) or (a.base_lr,a.base_weight_decay,a.adapter_lr,a.adapter_weight_decay)!=(1e-5,1e-5,1e-3,.08) or (a.start_global_epoch,a.continuation_epochs)!=(200,32):raise ValueError("frozen contract")
def receipt(p):
 d=json.loads(p.read_text())
 if d.get("schema")!=RECEIPT or any(d.get(x) is not True for x in ("synthetic_all_gates_passed","gpu_smoke_all_gates_passed","identity_all_gates_passed","two_step_gradient_all_gates_passed","all_gates_passed")) or d.get("implementation",{}).get("trainer_sha256")!=sha256_file(Path(__file__).resolve()):raise ValueError("implementation receipt")
 return sha256_file(p)
def data(a):
 ids=read_case_ids(a.train_case_ids);d=VariableResolutionBrainCropDataset(a.cache,expected_resolution=128,expected_manifest_sha256=MAN,case_ids=ids);w=d.sampling_weights()
 if len(d)!=756 or tensor_raw_sha256(w)!=WEIGHTS:raise ValueError("data")
 g=torch.Generator().manual_seed(SEED);l=DataLoader(d,batch_size=8,sampler=WeightedRandomSampler(w,num_samples=756,replacement=True,generator=g),num_workers=4,pin_memory=True,persistent_workers=True,drop_last=True);return d,w,l
def pair(b,ri,rl,i,l):
 ids=[str(x) for x in b["case_id"]];return {"first_batch_case_ids":ids,"first_batch_case_ids_sha256":text_lines_sha256(ids),"first_batch_raw_image_sha256":tensor_raw_sha256(ri),"first_batch_raw_label_sha256":tensor_raw_sha256(rl),"first_batch_augmented_image_sha256":tensor_raw_sha256(i),"first_batch_augmented_label_sha256":tensor_raw_sha256(l)}
def optimizer(m,a):
 bp=list(m.base_parameters());ap=list(m.adapter_parameters());
 if set(map(id,bp))&set(map(id,ap)) or len(set(map(id,bp+ap)))!=len(list(m.parameters())):raise ValueError("parameter groups")
 o=torch.optim.AdamW([{"params":bp,"lr":a.base_lr,"weight_decay":a.base_weight_decay},{"params":ap,"lr":a.adapter_lr,"weight_decay":a.adapter_weight_decay}],betas=(.9,.999),eps=1e-8,foreach=False,fused=False);return o,bp+ap
def loss(m,x,y,fn):
 with torch.autocast(device_type="cuda",dtype=torch.bfloat16):z=m(x).final_logits;l=fn(z,y)
 if not bool(torch.isfinite(l)):raise RuntimeError("loss")
 return z,l
def checkpoint(a,m,e,diag,pairing,reload,rsha):return {"schema":CK,"schema_version":1,"arm":a.arm,"global_epoch":e,"continuation_epoch":e-200,"model_state_dict":m.deploy_state_dict(cpu=True),"dmrfb":contract(a.arm),"loss_diagnostics":diag,"pairing":pairing,"checkpoint_reload_max_abs_difference":reload,"deploy_max_abs_difference":0.0,"start_checkpoint_sha256":KD,"runner_sha256":sha256_file(Path(__file__).resolve()),"implementation_receipt_sha256":rsha,"idea_sha256":IDEA,"preregistration_sha256":PREREG,"feasibility_sha256":SCREEN,"amendment_sha256":AMENDMENT,"validation_access":False,"outer_selector_access":False,"confirmation_access":False,"test_access":False}
def smoke(a,m,o,ps,l,init,adapter_sha,device):
 write_json_atomic(a.output_dir/"SMOKE_STARTED.json",{"schema":SMOKE+".started","arm":a.arm,"validation_access":False,"outer_selector_access":False,"confirmation_access":False,"test_access":False});random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED);torch.cuda.reset_peak_memory_stats(device);fn=DiceCELoss(sigmoid=True,squared_pred=False,lambda_dice=1,lambda_ce=.1);it=iter(l);times=[];audits=[];first=None;identity_diff=None
 for step in range(4):
  b=next(it);ri=b["image"].to(device,dtype=torch.float32);rl=b["label"].to(device,dtype=torch.float32);x,y=augment_batch(ri.clone(),rl.clone(),max_shift=8)
  if first is None:first={"initial_student_state_sha256":init,"initial_adapter_state_sha256":adapter_sha,**pair(b,ri,rl,x,y)}
  if identity_diff is None:
   cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all();was_training=m.training
   try:
    plain=AutomaticPromptLocator(center_aware=False,init_filters=16,dropout=.1).to(device).eval();plain.network.load_state_dict(m.network.state_dict(),strict=True);m.eval()
    with torch.inference_mode(),torch.autocast(device_type="cuda",dtype=torch.bfloat16):identity_diff=float((plain(x).segmentation_logits.float()-m(x).final_logits.float()).abs().max())
    del plain
   finally:
    torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state_all(cuda_rng);m.train(was_training)
  o.zero_grad(set_to_none=True);torch.cuda.synchronize();t=time.time();z,v=loss(m,x,y,fn);v.backward();names=list(m.adapter_named_parameters());audit={n:{"finite":p.grad is not None and bool(torch.isfinite(p.grad).all()),"nonzero":p.grad is not None and float(p.grad.float().norm())>0} for n,p in names};
  if step<2:audits.append(audit)
  nn.utils.clip_grad_norm_(ps,5,error_if_nonfinite=True);o.step();torch.cuda.synchronize();
  if step>0:times.append(time.time()-t)
 peak=int(torch.cuda.max_memory_allocated(device));est=statistics.median(times)*94*32;step1=all(v["finite"] and v["nonzero"] for n,v in audits[0].items() if n.startswith("adapter.projection"));step2=all(v["finite"] and v["nonzero"] for v in audits[1].values());g={"cuda_bfloat16_initial_plain_identity":identity_diff==0.0,"step1_projection":step1,"step2_all_adapter":step2,"memory":peak<=LIMIT,"runtime":est<=2700};r={"schema":SMOKE,"schema_version":1,"status":"complete" if all(g.values()) else "failed_gate","arm":a.arm,"pairing":first,"cuda_bfloat16_initial_plain_max_abs_difference":identity_diff,"gradient_audits":audits,"warmup_steps":1,"measured_steps":3,"peak_gpu_memory_bytes":peak,"estimated_formal_seconds":est,"gates":g,"all_gates_passed":all(g.values()),"runner_sha256":sha256_file(Path(__file__).resolve()),"validation_access":False,"outer_selector_access":False,"confirmation_access":False,"test_access":False};write_json_atomic(a.output_dir/"GPU_SMOKE.json",r)
 if not r["all_gates_passed"]:raise RuntimeError(g)
 return r
def formal(a,m,o,ps,d,w,l,init,adapter_sha,rsha,device):
 write_json_atomic(a.output_dir/"STARTED.json",{"schema":SCHEMA,"arm":a.arm,"runner_sha256":sha256_file(Path(__file__).resolve()),"implementation_receipt_sha256":rsha,"validation_access":False,"outer_selector_access":False,"confirmation_access":False,"test_access":False});random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED);torch.cuda.reset_peak_memory_stats(device);fn=DiceCELoss(sigmoid=True,squared_pred=False,lambda_dice=1,lambda_ce=.1);hist=[];cks={};orders={};before={};after={};first=None;start=time.time();last=None
 for ce in range(1,33):
  e=200+ce;cases=[];ls=[];ds=[]
  for s,b in enumerate(l):
   ri=b["image"].to(device,dtype=torch.float32);rl=b["label"].to(device,dtype=torch.float32)
   if s==0:before[str(e)]=cuda_rng_sha256(device)
   x,y=augment_batch(ri.clone(),rl.clone(),max_shift=8)
   if first is None:first={"initial_student_state_sha256":init,"initial_adapter_state_sha256":adapter_sha,**pair(b,ri,rl,x,y)}
   cases.extend(str(q) for q in b["case_id"]);o.zero_grad(set_to_none=True);z,v=loss(m,x,y,fn);v.backward();nn.utils.clip_grad_norm_(ps,5,error_if_nonfinite=True);o.step();ls.append(float(v));ds.extend(dice_per_case(z.detach().float(),y).cpu().tolist());last=x.detach()
  orders[str(e)]=text_lines_sha256(cases);after[str(e)]=cuda_rng_sha256(device);diag={"train_loss":float(np.mean(ls)),"train_dsc":float(np.mean(ds))};hist.append({"global_epoch":e,"continuation_epoch":ce,"base_lr":a.base_lr,"adapter_lr":a.adapter_lr,"optimizer_steps":94,**diag});pairing={**first,"epoch_case_order_sha256":dict(orders),"epoch_cuda_rng_before_sha256":dict(before),"epoch_cuda_rng_after_sha256":dict(after)}
  if e in EPOCHS:
   diff=m.deploy_max_abs_difference(last);state=m.deploy_state_dict(cpu=True);clone=DMRFBAdapterLocator(init_filters=16,dropout=.1,arm=a.arm).to(device);clone.load_state_dict(state,strict=True);clone.eval();m.eval();
   with torch.inference_mode(),torch.autocast(device_type="cuda",dtype=torch.bfloat16):reload=float((clone(last).final_logits.float()-m(last).final_logits.float()).abs().max())
   m.train();
   if diff!=0 or reload!=0:raise RuntimeError("deploy/reload")
   q=a.output_dir/f"plain_128_{a.arm}_global_epoch{e}.pth";torch.save(checkpoint(a,m,e,diag,pairing,reload,rsha),q);cks[str(e)]={"path":str(q.resolve()),"sha256":sha256_file(q),"checkpoint_reload_max_abs_difference":reload,"deploy_max_abs_difference":diff}
 peak=int(torch.cuda.max_memory_allocated(device));elapsed=time.time()-start;g={"memory":peak<=LIMIT,"runtime":elapsed<=2700,"checkpoints":set(cks)=={str(x) for x in EPOCHS},"reload":all(x["checkpoint_reload_max_abs_difference"]==0 and x["deploy_max_abs_difference"]==0 for x in cks.values())};r={"schema":SCHEMA,"schema_version":1,"status":"complete" if all(g.values()) else "failed_gate","arm":a.arm,"history":hist,"checkpoints":cks,"pairing":pairing,"formal_self_gates":g,"all_self_gates_passed":all(g.values()),"peak_gpu_memory_bytes":peak,"elapsed_seconds":elapsed,"reload_max_abs_difference":0.0,"deploy_max_abs_difference":0.0,"runner_sha256":sha256_file(Path(__file__).resolve()),"implementation_receipt_sha256":rsha,"dmrfb":contract(a.arm),"sampling_contract":{"draws":756,"optimized":752,"dropped":4,"steps":94,"weights_sha256":tensor_raw_sha256(w)},"validation_access":False,"outer_selector_access":False,"confirmation_access":False,"test_access":False};write_json_atomic(a.output_dir/"TRAIN_DONE.json",r)
 if r["status"]!="complete":raise RuntimeError(g)
 return r
def main():
 p=argparse.ArgumentParser();p.add_argument("--arm",choices=ARMS,required=True);p.add_argument("--mode",choices=("smoke","formal"),required=True);p.add_argument("--cache",type=Path,required=True);p.add_argument("--train-case-ids",type=Path,required=True);p.add_argument("--initial-checkpoint",type=Path,required=True);p.add_argument("--idea",type=Path,required=True);p.add_argument("--preregistration",type=Path,required=True);p.add_argument("--feasibility-screen",type=Path,required=True);p.add_argument("--amendment",type=Path,required=True);p.add_argument("--implementation-receipt",type=Path);p.add_argument("--output-dir",type=Path,required=True);p.add_argument("--device",default="cuda:0");p.add_argument("--seed",type=int,default=SEED);p.add_argument("--batch-size",type=int,default=8);p.add_argument("--num-workers",type=int,default=4);p.add_argument("--base-lr",type=float,default=1e-5);p.add_argument("--base-weight-decay",type=float,default=1e-5);p.add_argument("--adapter-lr",type=float,default=1e-3);p.add_argument("--adapter-weight-decay",type=float,default=.08);p.add_argument("--start-global-epoch",type=int,default=200);p.add_argument("--continuation-epochs",type=int,default=32);a=p.parse_args();frozen(a)
 if a.output_dir.exists():raise FileExistsError(a.output_dir)
 a.output_dir.mkdir(parents=True);device=torch.device(a.device);torch.cuda.set_device(device);plain=torch.load(a.initial_checkpoint,map_location="cpu",weights_only=False);m=DMRFBAdapterLocator.from_plain_checkpoint(plain,arm=a.arm).to(device);init=state_sha256(m.deploy_state_dict(cpu=True));adapter_sha=state_sha256(dict(m.adapter_named_parameters()));o,ps=optimizer(m,a);d,w,l=data(a);r=smoke(a,m,o,ps,l,init,adapter_sha,device) if a.mode=="smoke" else formal(a,m,o,ps,d,w,l,init,adapter_sha,receipt(a.implementation_receipt),device);print(json.dumps({"status":r["status"],"arm":a.arm}))
if __name__=="__main__":main()
