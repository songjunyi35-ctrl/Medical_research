"""Compact, ordered, hash-auditable binary predictions; never modifies input masks."""
import hashlib
import math
import os
from pathlib import Path
import re
import shutil
import numpy as np


def file_sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(8<<20),b''):digest.update(block)
    return digest.hexdigest()


class PackedPredictionWriter:
    def __init__(self,directory,names,case_ids,shape=(128,128,128)):
        self.directory=Path(directory);self.names=list(names);self.case_ids=list(case_ids);self.shape=tuple(shape)
        if not self.names or len(self.names)!=len(set(self.names)) or any(not re.fullmatch(r'[a-zA-Z0-9_]+',n) for n in self.names):raise ValueError('unique safe output names required')
        if not self.case_ids or len(self.case_ids)!=len(set(self.case_ids)):raise ValueError('unique ordered case IDs required')
        if len(self.shape)!=3 or any(not isinstance(n,int) or n<1 for n in self.shape):raise ValueError('positive 3D shape required')
        self.bits_per_case=math.prod(self.shape);self.bytes_per_case=(self.bits_per_case+7)//8
        self.file_bytes=self.bytes_per_case*len(self.case_ids)
        if self.directory.exists():raise FileExistsError(self.directory)
        self.directory.parent.mkdir(parents=True,exist_ok=True)
        if shutil.disk_usage(self.directory.parent).free<self.file_bytes*len(self.names)+(1<<30):raise RuntimeError('prediction mask storage reserve')
        self.directory.mkdir(exist_ok=False);self.counts={name:0 for name in self.names};self.handles={};self.finished=False
        try:
            for name in self.names:
                fd=os.open(self.directory/f'{name}.bits',os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
                self.handles[name]=fd;os.ftruncate(fd,self.file_bytes)
        except BaseException:self.close();raise

    def append(self,name,masks,case_ids):
        if name not in self.handles or self.finished:raise ValueError('unknown or closed output')
        array=np.asarray(masks)
        if array.dtype!=np.bool_ or array.ndim!=4 or tuple(array.shape[1:])!=self.shape:raise ValueError('B,D,H,W boolean masks required')
        start=self.counts[name];cases=list(case_ids)
        if len(cases)!=len(array) or self.case_ids[start:start+len(cases)]!=cases:raise ValueError('case order/count mismatch')
        packed=np.packbits(array.reshape(len(array),-1),axis=1,bitorder='little')
        body=packed.tobytes()
        if os.pwrite(self.handles[name],body,start*self.bytes_per_case)!=len(body):raise IOError('short prediction write')
        self.counts[name]+=len(cases)

    def finalize(self):
        if self.finished or any(n!=len(self.case_ids) for n in self.counts.values()):raise RuntimeError('all outputs must be complete exactly once')
        for fd in self.handles.values():os.fsync(fd)
        self.close();self.finished=True
        return {'schema':'aflow.packed_prediction_masks','shape':list(self.shape),'case_ids':self.case_ids,'cases':len(self.case_ids),'bitorder':'little','flatten_order':'C','bits_per_case':self.bits_per_case,'bytes_per_case':self.bytes_per_case,'padding_bits_per_case':self.bytes_per_case*8-self.bits_per_case,'files':{name:{'path':str(self.directory/f'{name}.bits'),'sha256':file_sha(self.directory/f'{name}.bits'),'bytes':self.file_bytes,'cases':self.counts[name]} for name in self.names}}

    def close(self):
        for fd in self.handles.values():os.close(fd)
        self.handles={}

    def __enter__(self):return self
    def __exit__(self,*args):self.close()


def read_packed_case(index,name,slot,verify_file=False):
    if index['schema']!='aflow.packed_prediction_masks' or index['bitorder']!='little' or index['flatten_order']!='C':raise ValueError('unsupported mask index')
    if not 0<=slot<index['cases']:raise IndexError(slot)
    info=index['files'][name];path=Path(info['path'])
    if path.stat().st_size!=info['bytes']:raise ValueError('mask file length')
    if verify_file and file_sha(path)!=info['sha256']:raise ValueError('mask file hash')
    with path.open('rb') as f:
        f.seek(slot*index['bytes_per_case']);body=f.read(index['bytes_per_case'])
    if len(body)!=index['bytes_per_case']:raise ValueError('short mask read')
    bits=np.unpackbits(np.frombuffer(body,dtype=np.uint8),bitorder='little')
    if bits[index['bits_per_case']:].any():raise ValueError('nonzero padding bits')
    return bits[:index['bits_per_case']].reshape(index['shape']).astype(bool)
