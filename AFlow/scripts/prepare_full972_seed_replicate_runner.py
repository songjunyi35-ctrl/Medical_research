#!/usr/bin/env python3
"""Mechanically specialize the copied fulltrajectory runner for two frozen seeds."""
from pathlib import Path
p=Path(__file__).with_name("train_t1ce_wt_full972_seed_replicates.py")
s=p.read_text()
replacements={
'"""Retrain full756/full972 random-init voxel-KD to the inner-selected epoch."""':'"""Train two preregistered Full972 random-init voxel-KD seed replicates."""',
'SCHEMA = "aflow.sammed3d.t1ce_wt_voxel_kd_fulltrajectory_final"':'SCHEMA = "aflow.sammed3d.t1ce_wt_full972_seed_replicate"',
'CHECKPOINT_SCHEMA = "aflow.sammed3d.t1ce_wt_voxel_kd_fulltrajectory_final_checkpoint"':'CHECKPOINT_SCHEMA = "aflow.sammed3d.t1ce_wt_full972_seed_replicate_checkpoint"',
'ARMS = ("full756_control", "full972_candidate")\nARM = {\n    "full756_control": (756, 346, "a520439c5c4b7b6a630a43312c0ce0b2af8d29f671d1e7bea129084bdde74d16"),\n    "full972_candidate": (972, 444, "6728d8d5e6846f90f6e5d0ad47acc4f2706eed42763ace4e228abcbbd6a4cf65"),\n}':'ARMS = ("full972_seed20260940", "full972_seed20260941")\nARM = {arm: (972, 444, "6728d8d5e6846f90f6e5d0ad47acc4f2706eed42763ace4e228abcbbd6a4cf65") for arm in ARMS}\nARM_SEEDS = {"full972_seed20260940": 20260940, "full972_seed20260941": 20260941}',
'IDEA_SHA256 = "f81eb110ba47083d9c07dafef7603dc16098842e54d442ff6e8d590f4046ac1a"':'IDEA_SHA256 = "5edb34e288b42a323cf068da4ce22604af70337613c0e9cc46b8b2c550d598a2"',
'PREREG_SHA256 = "0df7f883516ba0ad7d7dc0a67feffbb75c59b05ec808649b7d97d3cccd872a84"':'PREREG_SHA256 = "69174c0d3d01236d963bbc6bff27b2b2472fd54730576b63fff1b7e28877ff7a"',
'if args.seed != 20260932 or args.batch_size != 8:\n        raise ValueError("final protocol requires seed20260932 and batch8")':'expected_seed = ARM_SEEDS[args.arm]\n    if args.seed is None:\n        args.seed = expected_seed\n    if args.seed != expected_seed or args.batch_size != 8:\n        raise ValueError("seed replicate arm/seed or batch mismatch")',
'checkpoint_path = output_dir / f"plain_fulltrajectory_{args.arm}_epoch{completed_epoch}.pth"':'checkpoint_path = output_dir / f"plain_{args.arm}_epoch{completed_epoch}.pth"',
'record = ROOT / "archive/research_records/2026-08-24_voxel_kd_fulltrajectory_972"\n    prior':'record = ROOT / "archive/research_records/2026-08-25_full972_multiseed_ensemble_distill"\n    selection_record = ROOT / "archive/research_records/2026-08-24_voxel_kd_fulltrajectory_972"\n    prior',
'default=record / "INNER_SELECTION.json"':'default=selection_record / "INNER_SELECTION.json"',
'parser.add_argument("--seed", type=int, default=20260932)':'parser.add_argument("--seed", type=int, default=None)',
}
for old,new in replacements.items():
    if old not in s: raise RuntimeError(f"missing replacement source: {old[:80]}")
    s=s.replace(old,new,1)
p.write_text(s)
