#!/usr/bin/env python3
from pathlib import Path
p=Path(__file__).with_name("train_t1ce_wt_full972_width_final.py"); s=p.read_text()
repls={
'SCHEMA = "aflow.sammed3d.t1ce_wt_voxel_kd_fulltrajectory_final"':'SCHEMA = "aflow.sammed3d.t1ce_wt_full972_width_final"',
'CHECKPOINT_SCHEMA = "aflow.sammed3d.t1ce_wt_voxel_kd_fulltrajectory_final_checkpoint"':'CHECKPOINT_SCHEMA = "aflow.sammed3d.t1ce_wt_full972_width_final_checkpoint"',
'ARMS = ("full756_control", "full972_candidate")\nARM = {\n    "full756_control": (756, 346, "a520439c5c4b7b6a630a43312c0ce0b2af8d29f671d1e7bea129084bdde74d16"),\n    "full972_candidate": (972, 444, "6728d8d5e6846f90f6e5d0ad47acc4f2706eed42763ace4e228abcbbd6a4cf65"),\n}':'ARMS = ("full972_width16", "full972_width24")\nARM = {arm: (972, 444, "6728d8d5e6846f90f6e5d0ad47acc4f2706eed42763ace4e228abcbbd6a4cf65") for arm in ARMS}\nWIDTH = {"full972_width16": 16, "full972_width24": 24}\nPARAMETERS = {"full972_width16": 4_700_897, "full972_width24": 10_574_161}',
'INNER_SELECTION_SHA256 = "6b88f43291360eedba292165c6a968110a1e618a47a622e2bcda72792db3cca3"':'INNER_SELECTION_SHA256 = "a2026376abed345ac35e51e743ec05f5b626371adab9c70514539e26e2d55e24"',
'IDEA_SHA256 = "f81eb110ba47083d9c07dafef7603dc16098842e54d442ff6e8d590f4046ac1a"':'IDEA_SHA256 = "648581269c576141806f459230b94b85a4bbeb4a141d79f9d814a1829c2edc8b"',
'PREREG_SHA256 = "0df7f883516ba0ad7d7dc0a67feffbb75c59b05ec808649b7d97d3cccd872a84"':'PREREG_SHA256 = "7a443e1497c8a061df4de959b457373ff6279f927df432adcdd9cccddc247c95"',
'MAX_PEAK_BYTES = 24 * 1024**3':'MAX_PEAK_BYTES = 32 * 1024**3',
'MAX_FORMAL_SECONDS = 6 * 60 * 60':'MAX_FORMAL_SECONDS = 8 * 60 * 60',
'if args.stop_epoch != 168 or args.schedule_total_epochs != 200:\n        raise ValueError("authorized final protocol requires stop=168 / schedule total=200")':'if args.stop_epoch != 166 or args.schedule_total_epochs != 200:\n        raise ValueError("authorized width final protocol requires stop=166 / schedule total=200")',
'if args.seed != 20260932 or args.batch_size != 8:\n        raise ValueError("final protocol requires seed20260932 and batch8")':'if args.seed != 20260961 or args.batch_size != 8:\n        raise ValueError("width final requires seed20260961 and batch8")\n    if args.init_filters is None:\n        args.init_filters = WIDTH[args.arm]\n    if args.expected_parameter_count is None:\n        args.expected_parameter_count = PARAMETERS[args.arm]\n    if args.init_filters != WIDTH[args.arm] or args.expected_parameter_count != PARAMETERS[args.arm]:\n        raise ValueError("width final arm contract mismatch")',
'"peak_allocated_at_most_24gib"':'"peak_allocated_at_most_32gib"',
'"elapsed_within_6h"':'"elapsed_within_8h"',
'record = ROOT / "archive/research_records/2026-08-24_voxel_kd_fulltrajectory_972"':'record = ROOT / "archive/research_records/2026-08-26_full972_width24_fulltrajectory"',
'parser.add_argument("--seed", type=int, default=20260932)':'parser.add_argument("--seed", type=int, default=20260961)',
'parser.add_argument("--stop-epoch", type=int, default=168)':'parser.add_argument("--stop-epoch", type=int, default=166)',
'parser.add_argument("--init-filters", type=int, default=16)':'parser.add_argument("--init-filters", type=int, default=None)',
'parser.add_argument("--expected-parameter-count", type=int, default=4_700_897)':'parser.add_argument("--expected-parameter-count", type=int, default=None)',
}
for old,new in repls.items():
 if old not in s: raise RuntimeError(f"missing source {old[:80]}")
 s=s.replace(old,new,1)
p.write_text(s)
