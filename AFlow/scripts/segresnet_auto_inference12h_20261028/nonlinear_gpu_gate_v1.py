import torch
from scripts.segresnet_auto_inference12h_20261028.nonlinear_fusions_v1 import numerical_gate,definitions
from scripts.segresnet_auto_inference12h_20261028.common import numeric_mode,R,write

def main():
 numeric_mode()
 with torch.inference_mode():
  result=numerical_gate();result.update(recipes=len(definitions()),configurations=len(definitions())*39)
  write(R/'NONLINEAR_FUSION_GPU_GATE.json',result);print(result)
if __name__=='__main__':main()
