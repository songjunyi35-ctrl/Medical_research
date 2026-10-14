"""Conditionally execute UTSW362 only after Stage A, Full972 and RHUH precedence."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

from scripts import utsw_no_n4_paired_external_protocol as p


def run_checked(command: list[str]) -> None:
    print("running " + " ".join(command), flush=True)
    subprocess.run(command, cwd=p.ROOT, check=True)


def gpu_idle() -> bool:
    reply = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    )
    rows = [tuple(int(piece.strip()) for piece in line.split(",")) for line in reply.stdout.splitlines() if line.strip()]
    return len(rows) == 2 and all(memory <= 512 and utilization == 0 for memory, utilization in rows)


def wait_idle() -> None:
    consecutive = 0
    while consecutive < 3:
        if datetime.now(timezone.utc) >= p.DEADLINE:
            raise RuntimeError("30h deadline reached waiting for UTSW GPU")
        consecutive = consecutive + 1 if gpu_idle() else 0
        print(f"UTSW idle GPU check {consecutive}/3", flush=True)
        if consecutive < 3:
            time.sleep(10)


def terminal(status: str, boundary: str) -> None:
    p.write_new(p.RECORD / "QUEUE_TERMINAL.json", {
        "schema": "aflow.utsw_no_n4_paired_external.queue_terminal",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": status, "boundary": boundary,
        "new_utsw_subjects_accessed": False, "model_inference": False,
    })


def main() -> None:
    from scripts.freeze_utsw_no_n4_paired_external import check_implementation

    check_implementation()
    p.write_new(p.RECORD / "QUEUE_RECEIPT.json", {
        "schema": "aflow.utsw_no_n4_paired_external.queue",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "waiting_without_cuda_or_new_utsw_access",
        "order": ["family_readout", "no_n4_pass", "Full972", "RHUH_terminal", "four_hour_gate", "preaccess", "smoke", "dual_source", "paired_cache", "evaluation"],
        "deadline_utc": p.DEADLINE.isoformat(),
        "runner_sha256": p.sha(Path(__file__)),
    })
    family = p.full.RECORD / "FAMILY_READOUT_DONE.json"
    while not family.is_file():
        if datetime.now(timezone.utc) >= p.DEADLINE:
            terminal("closed_before_metric", "deadline_before_family_readout")
            return
        time.sleep(30)
    decisions = p.full.recompute_decisions()
    if decisions["no_n4"]["performance_passed"] is not True:
        terminal("closed_before_metric", "no_n4_stage_a_failed")
        return
    full_done = p.full.RECORD / "FULL972_FORMAL_RUNNER_DONE.json"
    while not full_done.is_file():
        if datetime.now(timezone.utc) >= p.DEADLINE:
            terminal("closed_before_metric", "deadline_before_full972")
            return
        time.sleep(30)
    while not p.PRECEDENCE.is_file():
        if datetime.now(timezone.utc) >= p.DEADLINE:
            terminal("closed_before_metric", "deadline_before_rhuh_precedence")
            return
        time.sleep(30)
    p.verify_precedence()
    if not p.four_hour_gate():
        terminal("closed_before_metric", "four_hour_gate_failed")
        return
    wait_idle()
    run_checked([sys.executable, str(p.ROOT / "scripts/prepare_utsw_no_n4_paired_external.py")])
    run_checked([sys.executable, str(p.ROOT / "scripts/smoke_utsw_no_n4_paired_external.py"), "--device", "cuda:0"])
    run_checked([sys.executable, str(p.ROOT / "scripts/download_utsw_no_n4_paired_external.py")])
    run_checked([sys.executable, str(p.ROOT / "scripts/build_utsw_no_n4_paired_external_caches.py"), "--workers", "8"])
    wait_idle()
    run_checked([sys.executable, str(p.ROOT / "scripts/evaluate_utsw_no_n4_paired_external.py"), "--device", "cuda:0"])
    p.write_new(p.RECORD / "QUEUE_DONE.json", {
        "schema": "aflow.utsw_no_n4_paired_external.queue_done",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "raw_external_complete_pending_archival_and_canonical_updates",
        "raw_report": str(p.RESULT_WORK / "UTSW_PAIRED_EXTERNAL.json"),
        "raw_report_sha256": p.sha(p.RESULT_WORK / "UTSW_PAIRED_EXTERNAL.json"),
    })
    print("UTSW raw paired external evaluation complete; canonical archival remains", flush=True)


if __name__ == "__main__":
    main()
