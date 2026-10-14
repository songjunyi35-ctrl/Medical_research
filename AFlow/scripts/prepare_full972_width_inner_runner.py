#!/usr/bin/env python3
from pathlib import Path
p=Path(__file__).with_name("train_t1ce_wt_full972_width_inner.py"); s=p.read_text()
repls={
'SCHEMA = "aflow.sammed3d.t1ce_wt_voxel_kd_fulltrajectory_inner"':'SCHEMA = "aflow.sammed3d.t1ce_wt_full972_width_inner"',
'CHECKPOINT_SCHEMA = "aflow.sammed3d.t1ce_wt_voxel_kd_fulltrajectory_inner_checkpoint"':'CHECKPOINT_SCHEMA = "aflow.sammed3d.t1ce_wt_full972_width_inner_checkpoint"',
'ARMS = ("inner648_control", "inner864_candidate")\nFIT = {\n    "inner648_control": (648, 297, "724aaa38fcea4c3f9ad11d7b7828aeb625142911eb4d9c3d64bf71e197f10d61"),\n    "inner864_candidate": (864, 395, "6533098a1b0b6e0e7ea37bf166de0b2b0744d5404ec6dac91840e9cfaf8a6f3b"),\n}':'ARMS = ("width16_control", "width24_candidate")\nFIT = {arm: (864, 395, "6533098a1b0b6e0e7ea37bf166de0b2b0744d5404ec6dac91840e9cfaf8a6f3b") for arm in ARMS}\nWIDTH = {"width16_control": 16, "width24_candidate": 24}\nPARAMETERS = {"width16_control": 4_700_897, "width24_candidate": 10_574_161}',
'IDEA_SHA256 = "f81eb110ba47083d9c07dafef7603dc16098842e54d442ff6e8d590f4046ac1a"':'IDEA_SHA256 = "648581269c576141806f459230b94b85a4bbeb4a141d79f9d814a1829c2edc8b"',
'PREREG_SHA256 = "0df7f883516ba0ad7d7dc0a67feffbb75c59b05ec808649b7d97d3cccd872a84"':'PREREG_SHA256 = "7a443e1497c8a061df4de959b457373ff6279f927df432adcdd9cccddc247c95"',
'MAX_PEAK_BYTES = 24 * 1024**3':'MAX_PEAK_BYTES = 32 * 1024**3',
'MAX_FORMAL_SECONDS = 6 * 60 * 60':'MAX_FORMAL_SECONDS = 8 * 60 * 60',
'if args.seed != 20260931 or args.batch_size != 8:\n        raise ValueError("frozen inner protocol requires seed20260931 and batch8")':'if args.seed != 20260960 or args.batch_size != 8:\n        raise ValueError("frozen width inner protocol requires seed20260960 and batch8")\n    if args.init_filters is None:\n        args.init_filters = WIDTH[args.arm]\n    if args.expected_parameter_count is None:\n        args.expected_parameter_count = PARAMETERS[args.arm]\n    if args.init_filters != WIDTH[args.arm] or args.expected_parameter_count != PARAMETERS[args.arm]:\n        raise ValueError("arm width/parameter contract mismatch")',
'"peak_allocated_at_most_24gib"':'"peak_allocated_at_most_32gib"',
'"elapsed_within_6h"':'"elapsed_within_8h"',
'record = ROOT / "archive/research_records/2026-08-24_voxel_kd_fulltrajectory_972"':'record = ROOT / "archive/research_records/2026-08-26_full972_width24_fulltrajectory"',
'parser.add_argument("--expected-parameter-count", type=int, default=4_700_897)':'parser.add_argument("--expected-parameter-count", type=int, default=None)',
'parser.add_argument("--seed", type=int, default=20260931)':'parser.add_argument("--seed", type=int, default=20260960)',
'parser.add_argument("--init-filters", type=int, default=16)':'parser.add_argument("--init-filters", type=int, default=None)',
}
for old,new in repls.items():
 if old not in s: raise RuntimeError(f"missing source {old[:80]}")
 s=s.replace(old,new,1)
p.write_text(s)
