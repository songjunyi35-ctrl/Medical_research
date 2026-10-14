"""Dual-download UTSW362 for the frozen baseline panel, without voxel access."""
from datetime import datetime,timezone
import json
from pathlib import Path

from scripts import download_utsw_no_n4_paired_external as shared
from scripts import utsw_frozen_baseline_external_protocol as p


def main():
    from scripts.freeze_utsw_frozen_baseline_external import check_implementation
    check_implementation();pre=p.verify_preaccess()
    smoke=json.loads((p.RECORD/"GPU_SMOKE_RECEIPT.json").read_text())
    if smoke.get("status")!="passed" or smoke.get("preaccess_sha256")!=p.sha(p.PREACCESS):raise ValueError("baseline smoke mismatch")
    if not p.three_hour_gate():raise RuntimeError("three-hour gate expired")
    shared.p.checked(shared.ASCP,shared.ASCP_SHA);shared.p.checked(shared.KEY,shared.KEY_SHA);shared.p.checked(shared.CONFIG,shared.CONFIG_SHA)
    if p.SOURCE.exists():raise FileExistsError(p.SOURCE)
    rows=p.cohort();expected=shared.expected_paths(rows);p.SOURCE.mkdir()
    original=shared.p.SOURCE
    shared.p.SOURCE=p.SOURCE
    try:copies=[shared.transfer_copy(name,rows,expected) for name in ("copy_a","copy_b")]
    finally:shared.p.SOURCE=original
    maps=[{x["relative_path"]:x for x in c["files"]} for c in copies];comparisons=[]
    for name in sorted(expected):
        a,b=maps[0][name],maps[1][name];comparisons.append({"relative_path":name,"bytes":a.get("bytes"),"sha256":a.get("sha256"),
            "copy_exact":a.get("bytes")==b.get("bytes") and a.get("sha256")==b.get("sha256")})
    passed=all(c["all_sizes_exact"] for c in copies) and all(x["copy_exact"] for x in comparisons)
    report={"schema":"aflow.utsw_frozen_baseline_external.source_download","created_utc":datetime.now(timezone.utc).isoformat(),
        "status":"passed" if passed else "failed","files_per_copy":p.SOURCE_FILES,"bytes_per_copy":p.SOURCE_BYTES,
        "copies":copies,"comparisons":comparisons,"files_copy_exact":sum(x["copy_exact"] for x in comparisons),
        "preaccess_sha256":p.sha(p.PREACCESS),"voxel_arrays_accessed":False,"model_inference":False,
        "new_subjects_downloaded":359,"runner_sha256":p.sha(Path(__file__))}
    p.write_new(p.RECORD/"SOURCE_DOWNLOAD.json",report);print(json.dumps({"status":report["status"],"files_exact":report["files_copy_exact"],
        "sha256":p.sha(p.RECORD/"SOURCE_DOWNLOAD.json")},indent=2))
    if not passed:raise RuntimeError("UTSW baseline dual source failed")


if __name__=="__main__":main()
