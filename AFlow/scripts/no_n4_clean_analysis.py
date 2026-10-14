"""Predeclared patient-cluster statistics and promotion gates."""
from collections import defaultdict
import numpy as np


def summary(rows):
    groups=defaultdict(list)
    for row in rows:groups[row["subject_id"]].append(float(row["dice"]))
    positive=[row for row in rows if row["target_voxels"]>0]
    return {"scans":len(rows),"subjects":len(groups),"scan_dice":float(np.mean([r["dice"] for r in rows])),
            "subject_dice":float(np.mean([np.mean(v) for v in groups.values()])),
            "point_hits":sum(bool(r["point_hit"]) for r in positive),"point_denominator":len(positive)}


def cluster_samples(rows,indices):
    subjects=sorted({r["subject_id"] for r in rows});by=defaultdict(list)
    for row in rows:by[row["subject_id"]].append(float(row["dice"]))
    sums=np.array([sum(by[s]) for s in subjects]);counts=np.array([len(by[s]) for s in subjects]);means=sums/counts
    return sums[indices].sum(1)/counts[indices].sum(1),means[indices].mean(1)


def interval(values):return [float(v) for v in np.percentile(values,[2.5,97.5],method="linear")]


def compare(control,candidate,indices):
    if [(r["case_id"],r["subject_id"],r["target_voxels"]) for r in control] != [(r["case_id"],r["subject_id"],r["target_voxels"]) for r in candidate]:
        raise ValueError("paired identities/labels differ")
    c=summary(control);a=summary(candidate)
    cs,cp=cluster_samples(control,indices);as_,ap=cluster_samples(candidate,indices)
    delta={"scan_dice":a["scan_dice"]-c["scan_dice"],"subject_dice":a["subject_dice"]-c["subject_dice"],"point_hits":a["point_hits"]-c["point_hits"],
           "scan_dice_95ci":interval(as_-cs),"subject_dice_95ci":interval(ap-cp)}
    gates={"scan_positive":delta["scan_dice"]>0,"subject_positive":delta["subject_dice"]>0,"point_within_one":delta["point_hits"]>=-1,
           "scan_ci_safe":delta["scan_dice_95ci"][0]>=-.003,"subject_ci_safe":delta["subject_dice_95ci"][0]>=-.003}
    return {"control":{**c,"scan_dice_95ci":interval(cs),"subject_dice_95ci":interval(cp)},
            "candidate":{**a,"scan_dice_95ci":interval(as_),"subject_dice_95ci":interval(ap)},"delta":delta,"gates":gates,"all_gates_passed":all(gates.values())}


def aggregate(comparisons):
    gates={}
    means={}
    for style in ("identity","flip8"):
        means[style]={metric:float(np.mean([comparisons[fold][style]["delta"][metric] for fold in ("e","f")])) for metric in ("scan_dice","subject_dice")}
        gates[style+"_mean_scan_above_001"]=means[style]["scan_dice"]>.001
        gates[style+"_mean_subject_positive"]=means[style]["subject_dice"]>0
    gates["all_fold_style_gates"]=all(comparisons[f][s]["all_gates_passed"] for f in ("e","f") for s in ("identity","flip8"))
    return {"means":means,"gates":gates,"all_gates_passed":all(gates.values())}
