"""Wait for family readout, then conditionally run the preregistered Full972 stage."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import n4_family_full972_protocol as p
from scripts.freeze_n4_family_full972 import check_implementation


def gpu_idle() -> bool:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = []
    for line in result.stdout.splitlines():
        if line.strip():
            memory, utilization = (int(piece.strip()) for piece in line.split(","))
            rows.append((memory, utilization))
    return len(rows) >= 2 and all(memory <= 512 and utilization == 0 for memory, utilization in rows[:2])


def run_checked(command: list[str]) -> None:
    print("running " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    check_implementation()
    print("Full972 queue waiting for archived family readout", flush=True)
    while not (p.RECORD / "FAMILY_READOUT_DONE.json").is_file():
        if datetime.now(timezone.utc) >= p.DEADLINE:
            raise RuntimeError("30h deadline reached before family readout")
        time.sleep(30)
    admission_path = p.RECORD / "FULL972_ADMISSION_CHECK.json"
    if not admission_path.exists():
        run_checked([sys.executable, str(ROOT / "scripts/prepare_n4_family_full972_admission.py")])
    decisions = p.recompute_decisions()
    eligible = [name for name in p.CANDIDATES if decisions[name]["performance_passed"]]
    if not eligible:
        print("No family candidate passed Stage A; Full972 queue exits without CUDA", flush=True)
        return
    p.verify_admission(require_time=True)
    if shutil.disk_usage(ROOT).free < 5 * 1024**3:
        raise RuntimeError("less than 5GiB free before Full972 checkpoints")
    consecutive = 0
    while consecutive < 3:
        if datetime.now(timezone.utc) >= p.DEADLINE:
            raise RuntimeError("30h deadline reached while waiting for idle GPUs")
        consecutive = consecutive + 1 if gpu_idle() else 0
        print(f"Full972 idle GPU check {consecutive}/3", flush=True)
        if consecutive < 3:
            time.sleep(10)
    run_checked([sys.executable, str(ROOT / "scripts/run_n4_family_full972.py"), "smoke"])
    run_checked([sys.executable, str(ROOT / "scripts/run_n4_family_full972.py"), "formal"])
    print("Conditional Full972 training complete; evaluation remains unlaunched", flush=True)


if __name__ == "__main__":
    main()
