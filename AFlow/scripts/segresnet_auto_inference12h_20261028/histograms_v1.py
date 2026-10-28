"""Exact integer histograms with dense probability tails handled in shared bins."""
import torch


def fast_histogram(p,target,bins):
    codes=(p.clamp(0,1).flatten()*(bins-1)+.5).long()
    labels=target.flatten().long()
    width=min(1024,bins//4)
    low=codes<width;high=codes>=bins-width;middle=~(low|high)
    inner=torch.bincount(codes[middle]+labels[middle]*bins,minlength=2*bins).reshape(2,bins)
    lo=torch.bincount(codes[low]+labels[low]*width,minlength=2*width).reshape(2,width)
    hi=torch.bincount(codes[high]-(bins-width)+labels[high]*width,minlength=2*width).reshape(2,width)
    inner[:,:width]=lo;inner[:,-width:]=hi
    return inner.cpu().numpy()
