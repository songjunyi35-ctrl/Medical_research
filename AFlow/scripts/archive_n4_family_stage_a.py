"""Create self-contained Stage-A raw/hash/result artifacts without new inference."""
from datetime import datetime,timezone
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import no_n4_clean_protocol as base


def main(method):
    is_otsu=method=="otsu"
    record=ROOT/("archive/research_records/2026-09-05_n4_otsu_clean_replication" if is_otsu else "archive/research_records/2026-09-05_no_n4_clean_teacher_replication")
    work=ROOT/("workspace/BraTS/sammed3d_t1ce_wt/n4_otsu_clean_20260905" if is_otsu else "workspace/BraTS/sammed3d_t1ce_wt/no_n4_clean_20260905")
    raw=json.loads((record/"RAW_INNER.json").read_text())
    evidence=[]
    for name in ("IDEA.md","PREREG.md","CACHE_RECEIPT.json","IMPLEMENTATION.json","GPU_SMOKE_RECEIPT.json","INNER_PREACCESS.json","RAW_INNER.json"):
        path=record/name;evidence.append({"path":str(path.relative_to(ROOT)),"sha256":base.sha(path)})
    cache_receipt=json.loads((record/"CACHE_RECEIPT.json").read_text())
    index=Path(cache_receipt["candidate_cache"])/"index.json"
    base.checked(index,cache_receipt["candidate_index_sha256"])
    evidence.append({"path":str(index),"sha256":base.sha(index)})
    reports=[];shared=[]
    for fold in ("e","f"):
        for role in ("history","ema","canonical"):
            path=base.WORK/"formal"/f"{fold}_{role}"/"TRAIN_DONE.json"
            item=json.loads(path.read_text());assert item["all_gates_passed"]
            base.checked(item["checkpoint"]["path"],item["checkpoint"]["sha256"])
            shared.append(item)
            evidence.extend(({"path":str(path.relative_to(ROOT)),"sha256":base.sha(path)},item["checkpoint"]))
        role="otsu" if is_otsu else "no_n4"
        path=work/"formal"/f"{fold}_{role}"/"TRAIN_DONE.json"
        item=json.loads(path.read_text());assert item["all_gates_passed"]
        base.checked(item["checkpoint"]["path"],item["checkpoint"]["sha256"])
        reports.append(item);evidence.extend(({"path":str(path.relative_to(ROOT)),"sha256":base.sha(path)},item["checkpoint"]))
    cost={"shared_teacher_and_control_gpu_hours":sum(x["elapsed_seconds"] for x in shared)/3600,
          "incremental_candidate_gpu_hours":sum(x["elapsed_seconds"] for x in reports)/3600,
          "including_shared_gpu_hours":sum(x["elapsed_seconds"] for x in shared+reports)/3600,
          "max_peak_gpu_gib":max(x["peak_gpu_memory_bytes"] for x in shared+reports)/1024**3,
          "candidate_cache_bytes":972*128**3*5,"inference_forwards":raw.get("new_forwards",raw.get("total_forwards")),
          "deployed_parameter_count":4700897,"deployment_forwards":[1,8]}
    passed=raw["aggregate"]["all_gates_passed"]
    result={"status":"stage_a_passed_final_not_yet_run" if passed else "stage_a_failed_closed",
            "method":method,"created_utc":datetime.now(timezone.utc).isoformat(),"comparisons":raw["comparisons"],"aggregate":raw["aggregate"],
            "stage_b_authorized":passed,"full972_final_completed":False,"cost":cost,"evidence":evidence,
            "cohort":{"e":{"scans":42,"subjects":17},"f":{"scans":42,"subjects":19},"mutual_patient_overlap":0},
            "evidence_scope":"shared-prespecified-family post-test development; own-fold teachers and students exclude own inner; not independent external/blind",
            "access":raw["access"],"new_inference_in_this_archiver":False}
    base.write_new(record/"RESULT.json",result)
    name="N4仅用Otsu自动前景拟合" if is_otsu else "取消额外N4校正"
    lines=[f"# {name}：Stage A结果","",f"状态：{'冻结开发门通过；Full972 final尚未运行' if passed else '冻结开发门失败，当前实现关闭'}。","",
        "两个候选在任何E/F模型结果前预注册，共享同一canonical对照；E42 scans/17 subjects、F42/19，患者互斥。每fold教师与学生都只训练该fold930 fit，排除自身inner。不是独立外部或pristine blind证据。","",
        "| Fold | 推理 | Control Scan/Subject/Point | Candidate Scan/Subject/Point | Scan/Subject/Point差 |", "|---|---|---|---|---|"]
    for fold in ("e","f"):
        for style in ("identity","flip8"):
            row=raw["comparisons"][fold][style];c=row["control"];a=row["candidate"];d=row["delta"]
            lines.append(f"| {fold.upper()} | {style} | {c['scan_dice']:.6f}/{c['subject_dice']:.6f}/{c['point_hits']}/{c['point_denominator']} | {a['scan_dice']:.6f}/{a['subject_dice']:.6f}/{a['point_hits']}/{a['point_denominator']} | {d['scan_dice']:+.6f}/{d['subject_dice']:+.6f}/{d['point_hits']:+d} |")
    lines.extend(("","## 区间与冻结门","","10,000次患者簇bootstrap保留每位患者全部扫描；下面是逐比较95%区间，不作未注册的家族显著性声明。"))
    for fold in ("e","f"):
        for style in ("identity","flip8"):
            row=raw["comparisons"][fold][style];d=row["delta"]
            lines.append(f"\n- {fold.upper()} {style}：Scan CI {d['scan_dice_95ci']}；Subject CI {d['subject_dice_95ci']}；失败门 {[k for k,v in row['gates'].items() if not v]}。")
    lines.extend(("","## 因素与解释","", "唯一学生pair差异是N4拟合mask（正值→Otsu200），不删mask外组织。" if is_otsu else "唯一学生pair差异是是否执行N4；原正值支持、标签/几何/模型/教师权重/seed/损失/采样/168轮/阈值固定。",
        "工程/机制结果不能替代准确率；当前"+("通过" if passed else "未通过")+"全部预注册准确率门。两fold方向与区间按上表保留，不挑有利指标。",
        "","## 成本与下一决定","",f"共享教师/对照{cost['shared_teacher_and_control_gpu_hours']:.3f} GPU-hours；本候选新增{cost['incremental_candidate_gpu_hours']:.3f} GPU-hours；最高allocated显存{cost['max_peak_gpu_gib']:.3f}GiB。新缓存9.49GiB位于/tmp。部署4.70M参数，identity1×或固定Flip8 8×。",
        "通过后仍须按时间窗条件完整Full972重训并另行冻结新外部确认；失败则关闭当前实现，不调参数、加seed、重测旧cohort。",
        "","本实验只新增E/F开发评测；无新fixed validation/outer/confirmation/test270/UCSD75/RHUH模型评测。原训练972包括此前已回收为train的旧selector/confirmation病例，不能把它们再称独立证据。",
        "","## 当前项目可比成绩与证据","","本Stage A不改变旧排行榜。Reused test270总体（含同一checkpoint8镜像）最高点估计仍为Full972 KD168 Flip8：0.757270/0.741254/265；1×Scan最高GCSAM184为0.752003，1×Subject/Point最高Full972 KD168为0.737327/262。Mirror-uint8 UCSD75单列，不与内部test混排。",
        "","[七问](IDEA.md)、[协议](PREREG.md)、[原始结果](RAW_INNER.json)、[预访问收据](INNER_PREACCESS.json)、[机器结果与哈希](RESULT.json)。",
        "","[当前交接](../../../AFLOW_BRATS_HANDOFF_CURRENT.md)、[方法目录](../../../AFLOW_BRATS_METHOD_CATALOG.md)、[排行榜](../../../AFLOW_BRATS_LEADERBOARD.md)。"))
    for report in reports:
        path=Path(report["checkpoint"]["path"])
        lines.append(f"\n- {report['fold'].upper()} fit930 candidate：[checkpoint](../../../{path.relative_to(ROOT)})，SHA256 `{report['checkpoint']['sha256']}`。不是Full972 final。")
    with (record/"RESULT.md").open("x") as f:f.write("\n".join(lines)+"\n")
    print(json.dumps({"method":method,"status":result["status"],"cost":cost},indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("method",choices=("no_n4","otsu"));main(parser.parse_args().method)
