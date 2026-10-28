"""Exact sufficient counts for the39 fixed quantized thresholds j/40."""
import numpy as np
import torch
import triton
import triton.language as tl


@triton.jit
def count_kernel(P,Y,H,N:tl.constexpr,BLOCK:tl.constexpr):
    offsets=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);valid=offsets<N
    p=tl.load(P+offsets,valid,other=0).to(tl.float32);y=tl.load(Y+offsets,valid,other=0).to(tl.int32)
    code=(tl.minimum(tl.maximum(p,0.),1.)*65535+.5).to(tl.int32)
    bucket=tl.minimum(tl.maximum((code*40-1)//65535,0),39)
    key=tl.where(valid,bucket+y*64,0)
    hist=tl.histogram(key,128);ix=tl.arange(0,128)
    hist=hist-tl.where(ix==0,tl.sum((~valid).to(tl.int32),0),0)
    tl.atomic_add(H+ix,hist,mask=hist>0,sem='relaxed')


def histogram(p,target):
    p=p.contiguous();target=target.contiguous();out=torch.zeros(128,device=p.device,dtype=torch.int32)
    count_kernel[(triton.cdiv(p.numel(),2048),)](p,target,out,p.numel(),2048,enable_fp_fusion=False,num_warps=4)
    return out.reshape(2,64).cpu().numpy().astype(np.int64)


def scores(hist):
    tail=np.cumsum(hist[:,::-1],axis=1,dtype=np.int64)[:,::-1]
    tp=tail[1,1:40];fp=tail[0,1:40];positive=hist[1].sum()
    return (2*tp/np.maximum(tp+fp+positive,1)).astype(np.float32)
