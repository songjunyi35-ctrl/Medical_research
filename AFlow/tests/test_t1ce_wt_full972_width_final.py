from pathlib import Path
import hashlib,json
import scripts.train_t1ce_wt_full972_width_final as train
ROOT=Path(__file__).resolve().parents[1]; RECORD=ROOT/"archive/research_records/2026-08-26_full972_width24_fulltrajectory"
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def test_selection_and_documents():
 assert sha(RECORD/"INNER_SELECTION.json")==train.INNER_SELECTION_SHA256; d=json.loads((RECORD/"INNER_SELECTION.json").read_text()); assert d["selected_epoch"]==166 and d["final_retraining_authorized"] is True
 assert sha(RECORD/"IDEA.md")==train.IDEA_SHA256; assert sha(RECORD/"PREREG.md")==train.PREREG_SHA256
def test_arms_and_widths():
 assert train.WIDTH=={"full972_width16":16,"full972_width24":24}; assert train.PARAMETERS=={"full972_width16":4700897,"full972_width24":10574161}; assert all(train.ARM[a][:2]==(972,444) for a in train.ARMS)
def test_resources_and_compute():
 assert (train.SAMPLER_DRAWS,train.CONSUMED_DRAWS,train.STEPS_PER_EPOCH)==(972,968,121); assert train.MAX_PEAK_BYTES==32*1024**3; assert train.MAX_FORMAL_SECONDS==8*60*60
