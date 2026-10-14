"""Fixed small input-grid rotations with isolated CPU randomness."""
import hashlib
import torch
import torch.nn.functional as F


def rotation_matrix(angles):
    """Output-to-input Rz Ry Rx, angles in radians and grid coordinates x,y,z."""
    a,b,c=angles.unbind(-1)
    o=torch.ones_like(a);z=torch.zeros_like(a)
    rx=torch.stack((o,z,z,z,a.cos(),-a.sin(),z,a.sin(),a.cos()),-1).reshape(-1,3,3)
    ry=torch.stack((b.cos(),z,b.sin(),z,o,z,-b.sin(),z,b.cos()),-1).reshape(-1,3,3)
    rz=torch.stack((c.cos(),-c.sin(),z,c.sin(),c.cos(),z,z,z,o),-1).reshape(-1,3,3)
    return rz@ry@rx


def rotate_pair(image, labels, angles):
    if image.ndim!=5 or labels.ndim!=5 or image.shape[0]!=labels.shape[0] or image.shape[-3:]!=labels.shape[-3:]:
        raise ValueError('paired B,C,D,H,W required')
    if len(set(image.shape[-3:]))!=1:raise ValueError('registered cube only')
    matrix=rotation_matrix(angles.to(device=image.device,dtype=torch.float32))
    affine=torch.cat((matrix,torch.zeros((len(image),3,1),device=image.device)),dim=-1)
    grid=F.affine_grid(affine,image.shape,align_corners=False)
    return (F.grid_sample(image,grid,mode='bilinear',padding_mode='zeros',align_corners=False),
            F.grid_sample(labels,grid,mode='nearest',padding_mode='zeros',align_corners=False))


class SmallRotation:
    def __init__(self,seed,probability=.2,max_degrees=15.):
        self.generator=torch.Generator(device='cpu').manual_seed(seed+200000)
        self.probability=probability;self.max_degrees=max_degrees
        self.steps=self.samples=self.selected=self.incoming_empty=0
        self.max_angle=0.;self.random_digest=hashlib.sha256()

    def __call__(self,image,labels):
        n=len(image)
        flags=torch.rand(n,generator=self.generator)<self.probability
        angles=(torch.rand((n,3),generator=self.generator)*2-1)*self.max_degrees
        self.random_digest.update(flags.numpy().tobytes());self.random_digest.update(angles.numpy().tobytes())
        self.steps+=1;self.samples+=n;self.selected+=int(flags.sum())
        if not flags.any():return image,labels
        ids=flags.nonzero().flatten().to(image.device)
        original=labels[ids]
        if not bool(((original==0)|(original==1)).all()):raise ValueError('binary labels required')
        x,y=rotate_pair(image[ids],original,angles[flags]*torch.pi/180)
        before=original[:,0].flatten(1).sum(1);after=y[:,0].flatten(1).sum(1)
        self.incoming_empty+=int((before==0).sum())
        if bool(((before>0)&(after==0)).any()):raise RuntimeError('rotation erased a nonempty WT; stop, no retry')
        if y.shape[1]==2 and bool((y[:,0]*y[:,1]!=0).any()):raise RuntimeError('WT/RC overlap')
        if not bool(torch.isfinite(x).all()):raise RuntimeError('nonfinite rotated image')
        self.max_angle=max(self.max_angle,float(angles[flags].abs().max()))
        outx,outy=image.clone(),labels.clone();outx[ids]=x;outy[ids]=y
        return outx,outy

    def receipt(self):
        return dict(steps=self.steps,samples=self.samples,selected=self.selected,
                    incoming_empty_selected=self.incoming_empty,max_observed_degrees=self.max_angle,
                    probability=self.probability,max_degrees=self.max_degrees,
                    independent_random_stream_sha256=self.random_digest.hexdigest())
