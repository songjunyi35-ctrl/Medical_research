import torch
from scripts.segresnet_auto_inference12h_20261028.joint_fusions_v1 import numerical_gate
from scripts.segresnet_auto_inference12h_20261028.common import numeric_mode,R,write

def main():
 numeric_mode()
 with torch.inference_mode():
  result=numerical_gate();write(R/'JOINT_FUSION_GPU_GATE.json',result);print(result)
if __name__=='__main__':main()
