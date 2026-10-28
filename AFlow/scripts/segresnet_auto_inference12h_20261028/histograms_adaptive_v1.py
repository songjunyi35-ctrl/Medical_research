"""Exact code histograms: aggregate a block's common probability range locally."""
import torch
import triton
import triton.language as tl


@triton.jit
def kernel(P,Y,H,N:tl.constexpr,BINS:tl.constexpr,BLOCK:tl.constexpr,WIDTH:tl.constexpr):
    offsets=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);valid=offsets<N
    p=tl.load(P+offsets,valid,other=0).to(tl.float32);y=tl.load(Y+offsets,valid,other=0).to(tl.int32)
    code=(tl.minimum(tl.maximum(p,0.),1.)*(BINS-1)+.5).to(tl.int32)
    first=tl.load(P+tl.program_id(0)*BLOCK).to(tl.float32)
    first_code=(tl.minimum(tl.maximum(first,0.),1.)*(BINS-1)+.5).to(tl.int32)
    base=tl.minimum(tl.maximum(first_code-WIDTH//2,0),BINS-WIDTH)
    local=(code>=base)&(code<base+WIDTH)
    ma=valid&local&(y==0);mb=valid&local&(y!=0)
    a=tl.histogram(tl.where(ma,code-base,0),WIDTH);b=tl.histogram(tl.where(mb,code-base,0),WIDTH)
    ix=tl.arange(0,WIDTH)
    a=a-tl.where(ix==0,tl.sum((~ma).to(tl.int32),0),0);b=b-tl.where(ix==0,tl.sum((~mb).to(tl.int32),0),0)
    tl.atomic_add(H+base+ix,a,mask=a>0,sem='relaxed');tl.atomic_add(H+BINS+base+ix,b,mask=b>0,sem='relaxed')
    # Outlying values retain exact global codes; no approximation or probability truncation.
    tl.atomic_add(H+code+y*BINS,1,mask=valid&(~local),sem='relaxed')


def histogram(p,target,bins=65536):
    assert bins==65536 and p.is_cuda and target.is_cuda
    p=p.contiguous();target=target.contiguous();out=torch.zeros(2*bins,device=p.device,dtype=torch.int32)
    kernel[(triton.cdiv(p.numel(),2048),)](p,target,out,p.numel(),bins,2048,1024,enable_fp_fusion=False,num_warps=4)
    return out.reshape(2,bins).cpu().numpy().astype('int64')
