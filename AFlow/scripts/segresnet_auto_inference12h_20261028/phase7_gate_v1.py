import torch
from scripts.segresnet_auto_inference12h_20261028.phase512_v2 import gate
from scripts.segresnet_auto_inference12h_20261028.common import numeric_mode,load_model,write,R

def main():
 numeric_mode();torch.cuda.set_per_process_memory_fraction(.65);model=load_model('cuda')
 result=gate(model,7);write(R/'PHASE512_FLIP7_SYNTHETIC_GATE.json',result);print(result)
if __name__=='__main__':main()
