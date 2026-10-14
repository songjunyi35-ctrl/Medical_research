"""Image-only mild blur, labels untouched and random stream isolated."""
import hashlib
import torch
import torch.nn.functional as F


def gaussian_blur(image,sigmas):
    if image.ndim!=5 or image.shape[1]!=1 or min(image.shape[-3:])<=3:raise ValueError('B,1,D,H,W with axes >3 required')
    sigmas=torch.as_tensor(sigmas,device=image.device,dtype=torch.float32).reshape(-1)
    if len(sigmas)!=len(image) or not bool(torch.isfinite(sigmas).all()) or not bool((sigmas>0).all()):raise ValueError('one positive finite sigma per image')
    taps=torch.arange(-3,4,device=image.device,dtype=torch.float32)
    kernel=torch.exp(-.5*(taps[None]/sigmas[:,None]).square());kernel=kernel/kernel.sum(1,keepdim=True)
    # Each case is an independent group; no batch mixing.
    x=image.transpose(0,1)
    for axis in range(3):
        shape=[len(image),1,1,1,1];shape[axis+2]=7
        pads=[0]*6;pads[2*(2-axis)]=3;pads[2*(2-axis)+1]=3
        x=F.conv3d(F.pad(x,pads,mode='reflect'),kernel.reshape(shape),groups=len(image))
    return x.transpose(0,1)


class ImageBlur:
    def __init__(self,seed,probability=.2):
        self.generator=torch.Generator(device='cpu').manual_seed(seed+300000)
        self.probability=probability;self.steps=self.samples=self.selected=0
        self.min_sigma=1.;self.max_sigma=0.;self.digest=hashlib.sha256()

    def __call__(self,image,labels):
        flags=torch.rand(len(image),generator=self.generator)<self.probability
        sigmas=torch.rand(len(image),generator=self.generator)*.5+.5
        self.digest.update(flags.numpy().tobytes());self.digest.update(sigmas.numpy().tobytes())
        self.steps+=1;self.samples+=len(image);self.selected+=int(flags.sum())
        if not flags.any():return image,labels
        ids=flags.nonzero().flatten().to(image.device)
        out=image.clone();out[ids]=gaussian_blur(image[ids],sigmas[flags])
        if not bool(torch.isfinite(out).all()):raise RuntimeError('nonfinite blur')
        self.min_sigma=min(self.min_sigma,float(sigmas[flags].min()));self.max_sigma=max(self.max_sigma,float(sigmas[flags].max()))
        return out,labels

    def receipt(self):
        return {'steps':self.steps,'samples':self.samples,'selected':self.selected,'min_observed_sigma':self.min_sigma,'max_observed_sigma':self.max_sigma,'probability':self.probability,'sigma_range':[.5,1.],'kernel_size':7,'padding':'reflect','independent_random_stream_sha256':self.digest.hexdigest(),'labels_unchanged_by_construction':True}
