"""No-patient/no-network gate for normalized-intensity TTA definitions."""
from types import SimpleNamespace
import itertools
import torch
from scripts.segresnet_auto_inference12h_20261028.common import *
from scripts.segresnet_auto_inference12h_20261028.render_v1 import forward_view


def specs():
    out=[]
    for gamma,gain,offset in itertools.product([.85,1.,1.15],[.85,1.,1.15],[-.15,0.,.15]):
        if gamma==gain==1. and offset==0.:continue
        out.append(dict(batch=8,gamma=gamma,gain=gain,offset=offset))
    out.extend(dict(batch=8,gaussian_sigma=s) for s in [.5,1.])
    return out


def main():
    class Identity:
        def __call__(self,x):return SimpleNamespace(segmentation_logits=x)
    x=torch.linspace(-2,2,16*24*32,device='cuda').reshape(1,1,16,24,32);x[:,:,:2]=0
    checks=[]
    for spec in specs():
        for flip in [0,7]:
            z=forward_view(Identity(),x,spec,flip)
            assert z.shape==x.shape and torch.isfinite(z).all() and (z[x==0]==0).all()
            if 'gaussian_sigma' not in spec:
                expected=x*spec['gain'];expected=torch.where(x!=0,expected+spec['offset'],expected);expected=expected.sign()*expected.abs().pow(spec['gamma'])
                assert torch.equal(z,expected)
            repeat=forward_view(Identity(),x,spec,flip);assert torch.equal(z,repeat)
        checks.append(dict(spec=spec,finite=True,zero_background_preserved=True,flip_roundtrip_exact=True,repeat_exact=True))
    write(R/'INTENSITY_SYNTHETIC_GATE.json',dict(at=now(),state='passed',checks=checks,source_sha256=sha(Path(__file__)),renderer_sha256=sha(ROOT/'scripts/segresnet_auto_inference12h_20261028/render_v1.py'),patient_forward=False,network_forward=False,normalization='gain then nonzero offset then signed-power gamma on existing standardized T1ce',native_neural_batch_gate='queued renderer runs actual synthetic batch8 before patient access'))
    print(dict(state='passed',specifications=len(checks),patient_forwards=0,network_forwards=0))

if __name__=='__main__':main()
