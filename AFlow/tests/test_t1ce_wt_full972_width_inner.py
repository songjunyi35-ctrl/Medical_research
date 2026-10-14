from pathlib import Path
import hashlib,torch
from sammed3d.t1ce_wt_capg3d import AutomaticPromptLocator
import scripts.train_t1ce_wt_full972_width_inner as train
ROOT=Path(__file__).resolve().parents[1]; RECORD=ROOT/"archive/research_records/2026-08-26_full972_width24_fulltrajectory"
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def test_documents_and_arms():
 assert sha(RECORD/"IDEA.md")==train.IDEA_SHA256; assert sha(RECORD/"PREREG.md")==train.PREREG_SHA256
 assert train.WIDTH=={"width16_control":16,"width24_candidate":24}; assert train.PARAMETERS=={"width16_control":4700897,"width24_candidate":10574161}
 assert all(train.FIT[a][:2]==(864,395) for a in train.ARMS)
def test_exact_parameter_counts():
 for arm in train.ARMS:
  m=AutomaticPromptLocator(center_aware=False,init_filters=train.WIDTH[arm],dropout=.1); assert sum(p.numel() for p in m.parameters())==train.PARAMETERS[arm]
def test_compute_and_resource_gates():
 assert (train.SAMPLER_DRAWS,train.CONSUMED_DRAWS,train.STEPS_PER_EPOCH)==(972,968,121)
 assert train.MAX_PEAK_BYTES==32*1024**3; assert train.MAX_FORMAL_SECONDS==8*60*60
def test_same_seed_different_width_states():
 values=[]
 for arm in train.ARMS:
  torch.manual_seed(20260960); m=AutomaticPromptLocator(center_aware=False,init_filters=train.WIDTH[arm],dropout=.1); values.append(train.state_dict_sha256(m.state_dict()))
 assert values[0]!=values[1]
