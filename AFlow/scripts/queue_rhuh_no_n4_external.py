"""Conditionally run RHUH external confirmation only after no-N4 final eligibility."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import rhuh_no_n4_external_protocol as p
from scripts.freeze_rhuh_no_n4_external import check_implementation


def run_checked(command: list[str]) -> None:
    print("running " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def gpu_idle() -> bool:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    )
    rows = [tuple(int(piece.strip()) for piece in line.split(",")) for line in result.stdout.splitlines() if line.strip()]
    return len(rows) >= 2 and all(memory <= 512 and utilization == 0 for memory, utilization in rows[:2])


def wait_idle() -> None:
    consecutive = 0
    while consecutive < 3:
        if datetime.now(timezone.utc) >= p.DEADLINE:
            raise RuntimeError("30h deadline reached waiting for RHUH GPU")
        consecutive = consecutive + 1 if gpu_idle() else 0
        print(f"RHUH idle GPU check {consecutive}/3", flush=True)
        if consecutive < 3:
            time.sleep(10)


def main() -> None:
    check_implementation()
    print("RHUH queue waiting for family Stage A", flush=True)
    while not (p.full.RECORD / "FAMILY_READOUT_DONE.json").is_file():
        if datetime.now(timezone.utc) >= p.DEADLINE:
            raise RuntimeError("30h deadline reached before family readout")
        time.sleep(30)
    decisions = p.full.recompute_decisions()
    if decisions["no_n4"]["performance_passed"] is not True:
        print("no-N4 failed Stage A; RHUH queue exits before labels/GPU", flush=True)
        return
    print("no-N4 passed Stage A; waiting for conditional Full972", flush=True)
    while not (p.full.RECORD / "FULL972_FORMAL_RUNNER_DONE.json").is_file():
        if datetime.now(timezone.utc) >= p.DEADLINE:
            raise RuntimeError("30h deadline reached before no-N4 Full972")
        time.sleep(30)
    if not p.two_hour_gate():
        print("less than two hours remain; RHUH queue exits before labels/GPU", flush=True)
        return
    run_checked([sys.executable, str(ROOT / "scripts/prepare_rhuh_no_n4_external.py")])
    wait_idle()
    run_checked([sys.executable, str(ROOT / "scripts/smoke_rhuh_no_n4_external.py"), "--device", "cuda:0"])
    run_checked([sys.executable, str(ROOT / "scripts/download_rhuh_no_n4_external_labels.py")])
    run_checked([sys.executable, str(ROOT / "scripts/build_rhuh_no_n4_external_cache.py"), "--workers", "8"])
    wait_idle()
    run_checked([sys.executable, str(ROOT / "scripts/evaluate_rhuh_no_n4_external.py"), "--device", "cuda:0"])
    print("RHUH one-time raw external evaluation complete; archival/canonical update remains", flush=True)


if __name__ == "__main__":
    main()
