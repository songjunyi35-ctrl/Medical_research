import numpy as np,time,resource
from scripts.segresnet_auto_inference12h_20261028.dense_crf_v1 import gate,infer
from scripts.segresnet_auto_inference12h_20261028.common import R,TMP,write,now

def main():
 vendor=TMP/'densecrf_vendor';result=gate(vendor);rng=np.random.default_rng(1)
 shape=(160,208,168);p=np.full(shape,.02,np.float32);p[55:105,70:140,55:110]=.9;image=rng.normal(size=shape).astype(np.float32)
 t=time.monotonic();out=infer(p,image,5.,.5,1.,iterations=(5,10),vendor=vendor);elapsed=time.monotonic()-t
 assert all(np.isfinite(q).all() for q in out.values());result.update(at=now(),full_canvas_seconds_for_5_and_10=elapsed,peak_RSS_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,patient_model_forward=False)
 write(R/'DENSE_CRF_FULL_CANVAS_GATE.json',result);print(result,flush=True)
if __name__=='__main__':main()
