#!/usr/bin/env python3
"""Compare five frozen train/external cache and input distributions."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "train972": (ROOT / "workspace/BraTS/cache/t1ce_wt_n4_braincrop_128_20260817/train", 972),
    "ucsd75_official_float": (Path("/tmp/aflow-ucsd-official-float-source-correction-cache-20260906"), 75),
    "africa95": (Path("/tmp/aflow-brats-africa95-cache-20260906"), 95),
    "cptac33": (Path("/tmp/aflow-cptac33-cache-20260906"), 33),
    "waw152": (ROOT / "workspace/BraTS/waw_gliomai_source_20260906/cache128", 152),
}
INDEX_SHA = {"train972": "7373cb306a01ec1ec311a584a088844439de84ff7289ed819127de6270839cfd", "ucsd75_official_float": "2540d5015008e22b40f3f94cc3f33b0d96cc12ee70a14b12c6d15a48e5a0856f", "africa95": "1493bfde0e82309c3bfc73e2b2b9e1ab631921a6aeb988288f16ddbd9cba1c2e", "cptac33": "9eeebdf66779496259b3a909ae8262dc380f3eaf73544059383b0d6c003cd140", "waw152": "a3831454785715ba48f845115571f99b8adfe61011d29fadaa0b91a4929adc7e"}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""): h.update(block)
    return h.hexdigest()


def quantiles(values) -> dict:
    q = np.quantile(np.asarray(values, dtype=np.float64), [0, .1, .25, .5, .75, .9, 1])
    return {name: float(value) for name, value in zip(("min","p10","p25","median","p75","p90","max"), q)}


def target_values(name, index):
    if name == "train972": return [int(x) for x in index["per_case_positive_voxels"].values()]
    if name == "ucsd75_official_float": return [int(x["resized_wt_voxels"]) for x in index["qc"]["cases"]]
    key = "resized_wt_voxels" if name == "africa95" else "resized_wt"
    return [int(x[key]) for x in index["details"].values()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--json", type=Path, required=True); parser.add_argument("--png", type=Path, required=True); parser.add_argument("--pdf", type=Path, required=True); args = parser.parse_args()
    if any(x.exists() for x in (args.json,args.png,args.pdf)): raise FileExistsError("output exists")
    result = {"schema": "aflow.input_domain_shift.audit", "schema_version": 2, "status": "descriptive_only_no_model_or_tuning", "cohorts": {}, "model_inference": False, "new_metric_access": False, "threshold_or_case_selection": False}
    per_case = {}
    for name,(root,count) in SOURCES.items():
        index_path=root/"index.json"
        if sha(index_path)!=INDEX_SHA[name]: raise ValueError(name+" index SHA")
        index=json.loads(index_path.read_text()); images_path=root/"images.f32"
        if images_path.stat().st_size != count*128**3*4: raise ValueError(name+" image bytes")
        images=np.memmap(images_path,dtype=np.float32,mode="r",shape=(count,128,128,128)); rows=[]
        for i in range(count):
            a=np.asarray(images[i],dtype=np.float32); nonzero=a!=0
            rows.append({"mean_all":float(np.mean(a,dtype=np.float64)),"std_all":float(np.std(a,dtype=np.float64)),"nonzero_fraction":float(nonzero.mean()),"mean_abs_all":float(np.mean(np.abs(a),dtype=np.float64)),"min":float(a.min()),"max":float(a.max())})
        targets=target_values(name,index); per_case[name]={"target_voxels":targets,**{key:[x[key] for x in rows] for key in rows[0]}}
        result["cohorts"][name]={"cases":count,"index_sha256":INDEX_SHA[name],"images_sha256_recorded":index["images_sha256"],"target_voxels":{"mean":float(np.mean(targets)),"fraction_of_128cubed_mean":float(np.mean(targets)/128**3),**quantiles(targets)},"input":{key:{"mean":float(np.mean([x[key] for x in rows])),**quantiles([x[key] for x in rows])} for key in rows[0]}}
        print(name,"done",flush=True)
    train=result["cohorts"]["train972"]
    result["contrasts"]={"external_relative_to_train":{name:{"target_mean_ratio":result["cohorts"][name]["target_voxels"]["mean"]/train["target_voxels"]["mean"],"target_median_ratio":result["cohorts"][name]["target_voxels"]["median"]/train["target_voxels"]["median"],"input_std_median_delta":result["cohorts"][name]["input"]["std_all"]["median"]-train["input"]["std_all"]["median"],"nonzero_fraction_median_delta":result["cohorts"][name]["input"]["nonzero_fraction"]["median"]-train["input"]["nonzero_fraction"]["median"]} for name in SOURCES if name!="train972"},"interpretation":"input support, scale, target extent, label semantics, acquisition, and treatment stage are confounded; no pooled ranking, causal, threshold, or calibration claim"}
    result["analyzer_sha256"]=sha(Path(__file__));args.json.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    fig,axes=plt.subplots(1,3,figsize=(16.5,4.3),constrained_layout=True);colors={"train972":"#4c72b0","ucsd75_official_float":"#55a868","africa95":"#c44e52","cptac33":"#8172b2","waw152":"#ccb974"};labels={"train972":"Train972","ucsd75_official_float":"UCSD75 official","africa95":"Africa95","cptac33":"CPTAC33","waw152":"WAW152"}
    for name in SOURCES:
        for ax,key,xlabel in ((axes[0],"target_voxels","WT voxels on 128³"),(axes[1],"std_all","cached image std (all voxels)"),(axes[2],"nonzero_fraction","cached nonzero fraction")):
            x=np.sort(np.asarray(per_case[name][key],dtype=np.float64));y=np.arange(1,len(x)+1)/len(x);ax.plot(x,y,label=labels[name],color=colors[name],lw=2)
            ax.set(xlabel=xlabel,ylabel="empirical CDF")
    axes[0].set_xscale("log");axes[0].set_title("Target extent shift");axes[1].set_title("Input scale after frozen pipeline");axes[2].set_title("Input support after 128³ resize");axes[0].legend(frameon=False)
    fig.suptitle("Five-cohort frozen cache audit (descriptive; no tuning)");fig.savefig(args.png,dpi=180);fig.savefig(args.pdf);plt.close(fig)
    print(json.dumps({"contrasts":result["contrasts"],"json_sha256":sha(args.json),"png_sha256":sha(args.png),"pdf_sha256":sha(args.pdf)},indent=2,sort_keys=True))


if __name__=="__main__":main()
