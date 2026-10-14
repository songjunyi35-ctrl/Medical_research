"""CPU-only synthetic math and existing-log audit. No patient/model inference."""
import csv,json,sys,math,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.sammed3d_graymean_single972 import gray_loss,FirstPromptLoss,sha,write,now
R=Path(__file__).parent
x=torch.tensor([0.,2.,0.,2.]).reshape(1,1,1,1,4);g=torch.tensor([0.,1.,0.,0.]).reshape_as(x)
correct=torch.tensor([-12.,12.,-12.,-12.]).reshape_as(x).requires_grad_(True)
wrong=torch.tensor([-12.,-12.,-12.,12.]).reshape_as(x).requires_grad_(True)
a=gray_loss(correct,x,g);b=gray_loss(wrong,x,g);b.backward()
def dice(z):
 p=z>0;return float(2*(p*g).sum()/(p.sum()+g.sum()))
assert torch.allclose(a,b) and dice(correct)==1 and dice(wrong)==0
z=torch.tensor([-2.,1.,-.5,2.]).reshape_as(x).requires_grad_(True)
base=gray_loss(z,x,g);scaled=gray_loss(z,x*10,g);shifted=gray_loss(z,x+7,g)
assert torch.allclose(scaled,base*100,rtol=1e-4)
assert torch.allclose(shifted,base,rtol=1e-4)
base.backward();assert torch.isfinite(z.grad).all() and z.grad.abs().sum()>0
f=FirstPromptLoss(x,1.)
first=f(z,g);second=f(z,g)
assert torch.allclose(first-second,base)
oldpath=ROOT/'workspace/BraTS/sammed3d_t1ce_wt/sammed3d_upstreamliteral11_officialval5_va_a100x2_train972_val108_e200_s2023/history.csv'
newpath=ROOT/'workspace/BraTS/sammed3d_t1ce_wt/graymean_single972_20260916/graymean/TRAIN.jsonl'
old=list(csv.DictReader(oldpath.open()));new=[json.loads(line) for line in newpath.read_text().splitlines()]
old_updates=np.cumsum([int(float(r['optimizer_applied_steps'])) for r in old]);new_updates=np.cumsum([r['steps'] for r in new]);new_val=[r for r in new if 'development' in r]
result={'utc':now(),'scope':'CPU synthetic tensors and existing logs only; no patient image/label access or model forward','synthetic':{'same_gray_loss_different_masks':{'correct_dice':dice(correct),'wrong_dice':dice(wrong),'correct_gray_loss':float(a.detach()),'wrong_gray_loss':float(b.detach())},'intensity_scale_times10_multiplies_loss_by':float((scaled/base).detach()),'constant_offset7_preserves_loss':bool(torch.allclose(shifted,base,rtol=1e-4)),'nonzero_finite_softmask_gradient':True,'first_prompt_only_gray_added':True},'optimizer_updates':{'old_selected_epoch21':int(old_updates[20]),'new_selected_epoch50':int(new_updates[49]),'old_epoch200':int(old_updates[199]),'new_epoch161':int(new_updates[160]),'old_epoch120':int(old_updates[119]),'new_epoch120':int(new_updates[119]),'new_first2_epochs':int(new_updates[1])},'pure_adamw_encoder_decay_multiplier_ignoring_gradient_updates':{'old_selected21':(1-.0008*.1)**int(old_updates[20]),'new_selected50':(1-.0008*.1)**int(new_updates[49]),'caveat':'mathematical isolated decay component, not measured total parameter norm or a causal attribution'},'old_baseline_curve':{'epoch21':old[20],'epoch200':old[199]},'new_curve':[{k:r[k] for k in ['epoch','steps','train','development']} for r in new_val],'hashes':{str(p.relative_to(ROOT)):sha(p) for p in [oldpath,newpath,Path(__file__),ROOT/'scripts/sammed3d_graymean_single972.py']}}
write(R/'LOCAL_DIAGNOSTICS.json',result)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,ax=plt.subplots(1,2,figsize=(11,4.2),sharey=True)
for a in ax:a.grid(alpha=.2);a.set_ylim(.28,.62);a.set_ylabel('Validation108 Scan Dice (best of 5 clicks)')
old_scores=[float(r['val_official_best5_dice']) for r in old];new_scores=[r['development']['scan_best5'] for r in new_val]
ax[0].plot([int(r['epoch']) for r in old],old_scores,label='Historical baseline',color='#3976c5')
ax[0].plot([r['epoch'] for r in new_val],new_scores,'o-',ms=3,label='Current gray-mean run',color='#d36b42')
ax[0].set_xlabel('Epoch');ax[0].set_title('Both runs decline after an early peak');ax[0].legend()
ax[1].plot(old_updates,old_scores,color='#3976c5')
ax[1].plot([new_updates[r['epoch']-1] for r in new_val],new_scores,'o-',ms=3,color='#d36b42')
ax[1].set_xscale('log');ax[1].set_xlabel('Applied optimizer updates (log scale)');ax[1].set_title('Epoch counts hide very different update counts')
fig.suptitle('Existing records only: different training recipes, not a controlled gray-loss ablation',fontsize=10)
fig.tight_layout();fig.savefig(R/'TRAINING_RECIPE_CURVES.png',dpi=160);plt.close(fig)
print(json.dumps({k:result[k] for k in ['synthetic','optimizer_updates','pure_adamw_encoder_decay_multiplier_ignoring_gradient_updates']}))
