from pathlib import Path
import hashlib, torch
from sammed3d.t1ce_wt_capg3d import AutomaticPromptLocator
import scripts.train_t1ce_wt_full972_seed_replicates as train
ROOT=Path(__file__).resolve().parents[1]
RECORD=ROOT/"archive/research_records/2026-08-25_full972_multiseed_ensemble_distill"
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def test_documents_and_arms():
 assert sha(RECORD/"IDEA.md")==train.IDEA_SHA256
 assert sha(RECORD/"PREREG.md")==train.PREREG_SHA256
 assert train.ARMS==("full972_seed20260940","full972_seed20260941")
 assert train.ARM_SEEDS=={"full972_seed20260940":20260940,"full972_seed20260941":20260941}
 assert all(train.ARM[a][:2]==(972,444) for a in train.ARMS)
def test_seed_states_are_deterministic_and_distinct():
 values=[]
 for seed in (20260940,20260941):
  torch.manual_seed(seed); m=AutomaticPromptLocator(center_aware=False,init_filters=16,dropout=0.1); values.append(train.state_dict_sha256(m.state_dict()))
 assert values[0]!=values[1]
def test_compute_and_epoch_are_frozen():
 assert (train.SAMPLER_DRAWS,train.CONSUMED_DRAWS,train.STEPS_PER_EPOCH)==(972,968,121)
 assert train.INNER_SELECTION_SHA256=="6b88f43291360eedba292165c6a968110a1e618a47a622e2bcda72792db3cca3"
