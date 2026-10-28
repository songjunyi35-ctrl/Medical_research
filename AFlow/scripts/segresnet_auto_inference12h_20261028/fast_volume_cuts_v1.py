"""Same whole-bin decoder; exploit the unimodal prefix-ratio objective."""
import numpy as np


def fast_cuts(counts,q,betas,scales):
    counts=np.asarray(counts,np.int64);bins=len(counts);descending=counts[::-1];full_k=np.cumsum(descending,dtype=np.int64)
    assert full_k[-1]>0 and all(beta>0 for beta in betas)
    support=np.flatnonzero(descending);support_counts=descending[support]
    mass=np.cumsum(np.ascontiguousarray(q[:,support])*support_counts[None],axis=1,dtype=np.float64);total=mass[:,-1]
    rows=np.arange(len(q));k=full_k[support];m=len(support);blocks=[]
    for beta in betas:
        c=beta**2*total;lo=np.zeros(len(q),np.int64);hi=np.full(len(q),m-1,np.int64)
        while np.any(lo<hi):
            active=lo<hi;mid=(lo+hi)//2;following=np.minimum(mid+1,m-1)
            increasing=q[rows,support[following]]*(k[mid]+c)>mass[rows,mid]
            lo=np.where(active&increasing,mid+1,lo);hi=np.where(active&~increasing,mid,hi)
        # Evaluate the original floating-point objective locally to retain its
        # tie rule rather than selecting solely by a multiplied derivative.
        indices=(lo[:,None]+np.arange(-8,9)[None]).clip(0,m-1)
        objective=2*mass[rows[:,None],indices]/(k[indices]+c[:,None])
        chosen=np.argmax(objective,axis=1);best=indices[rows,chosen]
        edge=((chosen==0)&(lo>8))|((chosen==16)&(lo<m-9))
        # Defensive exact fallback if numerical non-unimodality reaches a window edge.
        for row in np.flatnonzero(edge):best[row]=int(np.argmax(2*mass[row]/(k+c[row])))
        base_k=k[best]
        for scale in scales:
            desired=np.rint(base_k*scale).clip(1,int(full_k[-1])).astype(np.int64)
            right=np.searchsorted(full_k,desired,side='left').clip(0,bins-1);left=np.maximum(right-1,0)
            index=np.where(abs(full_k[left]-desired)<=abs(full_k[right]-desired),left,right)
            blocks.append((bins-1-index).astype(np.uint16))
    return np.concatenate(blocks)
