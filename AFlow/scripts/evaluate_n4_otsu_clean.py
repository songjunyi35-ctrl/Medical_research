"""Evaluate the frozen Otsu candidates against already-saved shared control rows."""
from datetime import datetime,timezone
import argparse
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.utils.data import DataLoader
from scripts import n4_otsu_protocol as p
from scripts import no_n4_clean_analysis as analysis
from scripts import train_t1ce_wt_gcsam_inner as base
from scripts.evaluate_no_n4_clean import evaluate
from scripts.freeze_n4_otsu_clean import check_implementation


def main(args):
    check_implementation()
    receipt_path=p.RECORD/"INNER_PREACCESS.json";receipt=json.loads(receipt_path.read_text())
    assert receipt["status"]=="frozen_family_before_first_ef_readout" and not receipt["first_ef_readout_occurred"]
    source_raw=p.SHARED_RECORD/"RAW_INNER.json";source=json.loads(source_raw.read_text())
    p.checked(p.SHARED_RECORD/"INNER_PREACCESS.json",source["preaccess_sha256"])
    split=json.loads((p.SHARED_RECORD/"SPLIT.json").read_text());p.checked(p.SHARED_RECORD/"SPLIT.json",p.SPLIT_SHA)
    cache_info=json.loads((p.RECORD/"CACHE_RECEIPT.json").read_text());cache=Path(cache_info["candidate_cache"])
    p.checked(cache/"index.json",cache_info["candidate_index_sha256"])
    for record in receipt["reports"].values():
        p.checked(record["control_report"],record["control_report_sha256"]);p.checked(record["candidate_report"],record["candidate_report_sha256"])
        p.checked(record["candidate_checkpoint"]["path"],record["candidate_checkpoint"]["sha256"])
    if (p.RECORD/"INNER_STARTED.json").exists():raise FileExistsError("readout already started")
    device=torch.device(args.device);assert device.type=="cuda" and torch.cuda.is_available()
    torch.set_num_threads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    p.write_new(p.RECORD/"INNER_STARTED.json",{"created_utc":datetime.now(timezone.utc).isoformat(),"preaccess_sha256":p.sha(receipt_path),"shared_control_raw_sha256":p.sha(source_raw)})
    raw={};comparisons={};start=time.monotonic()
    for fold in ("e","f"):
        ids=[r["case_id"] for r in p.read_rows(split["folds"][fold]["inner"]["path"])]
        dataset=base.BrainCropDataset(cache,expected_split="train",case_ids=ids)
        loader=DataLoader(dataset,batch_size=8,shuffle=False,num_workers=0)
        payload=torch.load(receipt["reports"][fold]["candidate_checkpoint"]["path"],map_location="cpu",weights_only=False)
        assert payload["fold"]==fold and payload["role"]=="otsu" and not payload["smoke"] and payload["epoch"]==168
        model=base.AutomaticPromptLocator(center_aware=False,init_filters=16,dropout=.1).to(device).eval()
        model.load_state_dict(payload["model_state_dict"],strict=True)
        raw[fold]=evaluate(model,loader,device);assert raw[fold]["forwards"]==336
        n=split["folds"][fold]["inner"]["subjects"]
        indices=np.random.default_rng(p.SEEDS[fold]["bootstrap"]).integers(0,n,size=(10000,n))
        comparisons[fold]={style:analysis.compare(source["raw_models"][fold]["canonical"]["per_case"][style],raw[fold]["per_case"][style],indices) for style in ("identity","flip8")}
        print(f"completed frozen Otsu/{fold}:42 cases",flush=True);del model;torch.cuda.empty_cache()
    aggregate=analysis.aggregate(comparisons)
    report={"schema":"aflow.n4_otsu_clean.inner","created_utc":datetime.now(timezone.utc).isoformat(),"candidates":raw,"comparisons":comparisons,"aggregate":aggregate,
        "stage_b_authorized":aggregate["all_gates_passed"],"preaccess_sha256":p.sha(receipt_path),"shared_control_raw_sha256":p.sha(source_raw),
        "new_forwards":672,"shared_control_forwards_reused":672,"elapsed_seconds":time.monotonic()-start,
        "bootstrap":{"unit":"patient_cluster_all_timepoints","samples":10000,"confidence":.95,"seeds":{fold:p.SEEDS[fold]["bootstrap"] for fold in ("e","f")}},
        "access":{"new_train_only_inner":True,"fixed_validation":False,"outer":False,"confirmation":False,"test270":False,"ucsd75":False,"rhuh":False}}
    p.write_new(p.RECORD/"RAW_INNER.json",report)
    print(json.dumps({"aggregate":aggregate,"stage_b_authorized":report["stage_b_authorized"]},indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--device",default="cuda:0");main(parser.parse_args())
