import torch
from scripts.segresnet_auto_inference12h_20261028.case_consistency_v1 import numerical_gate,definitions
from scripts.segresnet_auto_inference12h_20261028.common import R,write,numeric_mode

def main():
 numeric_mode()
 with torch.inference_mode():
  result=numerical_gate();result['recipes']=len(definitions());result['configurations']=len(definitions())*39
  write(R/'CASE_CONSISTENCY_GPU_GATE.json',result);print(result)
if __name__=='__main__':main()
