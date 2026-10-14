"""Create the two frozen42-case inners, excluding all pre-existing evaluation lists."""
from collections import Counter
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import no_n4_clean_protocol as p
from scripts.build_sammed3d_t1ce_wt_inner_split import choose_exact_validation_subjects


def choose(counts,excluded,seed,target=42):
    chosen=choose_exact_validation_subjects({k:v for k,v in counts.items() if k not in excluded},seed=seed,target_cases=target)
    if chosen&excluded or sum(counts[k] for k in chosen)!=target:raise ValueError("selection/exclusion failed")
    return chosen


def main():
    p.docs();p.checked(p.SOURCE,p.SOURCE_SHA)
    if p.SPLITS.exists():raise FileExistsError(p.SPLITS)
    rows=p.read_rows(p.SOURCE)
    assert len(rows)==972 and all(r["split"]=="train" for r in rows)
    counts=Counter(r["subject_id"] for r in rows);assert len(counts)==444
    root=ROOT/"data/datasets"
    exclusions=sorted(x for x in root.glob("*/*inner*cases.jsonl") if "smoke" not in str(x))
    exclusions.extend((root/"brats2024_t1ce_plain_locator_confirm_seed20260817_v1/selector_cases.jsonl",
                       root/"brats2024_t1ce_plain_locator_confirm_seed20260817_v1/sealed_confirmation/confirmation_cases.jsonl"))
    excluded=set();evidence=[]
    for path in exclusions:
        values=p.read_rows(path)
        subjects={str(r.get("subject_id") or "-".join(r["case_id"].split("-")[:3])) for r in values}
        excluded|=subjects
        evidence.append({"path":str(path.relative_to(ROOT)),"sha256":p.sha(path),"subjects":len(subjects)})
    eligible=set(counts)-excluded
    assert len(eligible)==38 and sum(counts[k] for k in eligible)==87
    selected={};blocked=set(excluded)
    for fold in ("e","f"):
        selected[fold]=choose(counts,blocked,p.SEEDS[fold]["split"])
        assert len(selected[fold])>=12
        blocked|=selected[fold]
    p.SPLITS.mkdir()
    result={"schema":"aflow.no_n4_clean.splits","created_utc":datetime.now(timezone.utc).isoformat(),
        "source_sha256":p.SOURCE_SHA,"idea_sha256":p.IDEA_SHA,"prereg_sha256":p.PREREG_SHA,
        "exclusions":evidence,"eligible_before_selection":{"subjects":38,"scans":87},"folds":{},
        "all_inner_metrics_unaccessed":True,"teacher_inner_overlap_must_be_zero":True,"builder_sha256":p.sha(Path(__file__))}
    for fold in ("e","f"):
        record={"seeds":p.SEEDS[fold]}
        for role in ("fit","inner"):
            values=[{"case_id":r["case_id"],"subject_id":r["subject_id"],"role":f"no_n4_{fold}_{role}"} for r in rows
                    if (r["subject_id"] in selected[fold])==(role=="inner")]
            output=p.SPLITS/f"{fold}_{role}.jsonl"
            with output.open("x") as f:
                for row in values:f.write(json.dumps(row,sort_keys=True)+"\n")
            record[role]={"path":str(output),"sha256":p.sha(output),"cases":len(values),"subjects":len({r["subject_id"] for r in values})}
        p.own_fit(record,p.read_rows(record["fit"]["path"]),p.read_rows(record["inner"]["path"]))
        result["folds"][fold]=record
    p.write_new(p.RECORD/"SPLIT.json",result)
    print(json.dumps({"folds":result["folds"],"eligible":result["eligible_before_selection"]},indent=2))


if __name__=="__main__":main()
