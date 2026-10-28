"""Image-only brain support for volume decoding; score domain remains unchanged."""
import numpy as np
from scipy import ndimage as ndi


def supports(image):
    mask=ndi.binary_fill_holes(image!=0,structure=ndi.generate_binary_structure(3,1))
    assert mask.any()
    return {-1:np.ones(image.shape,bool),0:mask,1:ndi.binary_dilation(mask,structure=ndi.generate_binary_structure(3,1),iterations=1),2:ndi.binary_dilation(mask,structure=ndi.generate_binary_structure(3,1),iterations=2)}


def scores(hist,cuts,positive_total):
    tail=np.cumsum(np.asarray(hist[:,::-1],np.int64),axis=1)[:,::-1]
    tp=tail[1,cuts.astype(np.int64)];pred=tail[:,cuts.astype(np.int64)].sum(0)
    return (2*tp/np.maximum(pred+positive_total,1)).astype(np.float32)


def gate():
    image=np.zeros((9,11,13),np.float32);image[2:7,3:8,4:9]=1;valid=supports(image)[0]
    target=np.zeros_like(valid);target[0,0,0]=True;target[4,5,6]=True
    probability=np.full(image.shape,.25,np.float32);probability[4,5,6]=.9
    codes=(probability*65535+.5).astype(np.int64)
    h=np.bincount(codes[valid]+target[valid].astype(np.int64)*65536,minlength=131072).reshape(2,65536)
    cuts=np.array([0,32768,65535],np.uint16);actual=scores(h,cuts,int(target.sum()))
    for i,cut in enumerate(cuts):
        pred=(codes>=int(cut))&valid;expected=np.float32(2*(pred&target).sum()/max(int(pred.sum())+int(target.sum()),1));assert actual[i]==expected
    assert actual[1]<1.,'outside-support GT must remain a false negative'
    return dict(status='passed',outside_support_GT_retained=True,whole_original_crop_scored=True,GT_input_to_support=False,direct_mask_score_exact=True)
