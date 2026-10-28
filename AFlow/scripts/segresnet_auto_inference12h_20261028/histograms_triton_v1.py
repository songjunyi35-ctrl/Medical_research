"""Exact uint16 probability-code counting with block-aggregated dense tails."""
import torch
import triton
import triton.language as tl


@triton.jit
def count_kernel(P,Y,H,N:tl.constexpr,BINS:tl.constexpr,BLOCK:tl.constexpr,WIDTH:tl.constexpr):
    offsets=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    valid=offsets<N
    p=tl.load(P+offsets,valid,other=0).to(tl.float32)
    y=tl.load(Y+offsets,valid,other=0).to(tl.int32)
    code=(tl.minimum(tl.maximum(p,0.),1.)*(BINS-1)+.5).to(tl.int32)
    low=code<WIDTH;high=code>=BINS-WIDTH
    ix=tl.arange(0,WIDTH)
    ma=valid&low&(y==0);mb=valid&low&(y!=0)
    mc=valid&high&(y==0);md=valid&high&(y!=0)
    a=tl.histogram(tl.where(ma,code,0),WIDTH)
    b=tl.histogram(tl.where(mb,code,0),WIDTH)
    c=tl.histogram(tl.where(mc,code-(BINS-WIDTH),0),WIDTH)
    d=tl.histogram(tl.where(md,code-(BINS-WIDTH),0),WIDTH)
    a=a-tl.where(ix==0,tl.sum((~ma).to(tl.int32),0),0)
    b=b-tl.where(ix==0,tl.sum((~mb).to(tl.int32),0),0)
    c=c-tl.where(ix==0,tl.sum((~mc).to(tl.int32),0),0)
    d=d-tl.where(ix==0,tl.sum((~md).to(tl.int32),0),0)
    tl.atomic_add(H+ix,a,mask=a>0,sem='relaxed')
    tl.atomic_add(H+BINS+ix,b,mask=b>0,sem='relaxed')
    tl.atomic_add(H+BINS-WIDTH+ix,c,mask=c>0,sem='relaxed')
    tl.atomic_add(H+2*BINS-WIDTH+ix,d,mask=d>0,sem='relaxed')
    tl.atomic_add(H+code+y*BINS,1,mask=valid&(~low)&(~high),sem='relaxed')


def histogram(p,target,bins=65536):
    assert bins==65536 and p.is_cuda and target.is_cuda
    p=p.contiguous();target=target.contiguous()
    result=torch.zeros(2*bins,device=p.device,dtype=torch.int32)
    count_kernel[(triton.cdiv(p.numel(),2048),)](p,target,result,p.numel(),bins,2048,256,enable_fp_fusion=False,num_warps=4)
    return result.reshape(2,bins).cpu().numpy().astype('int64')
