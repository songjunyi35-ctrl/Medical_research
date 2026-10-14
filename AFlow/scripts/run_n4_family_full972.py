"""Supervise the conditional Full972 smoke/formal waves; never evaluate them."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import n4_family_full972_protocol as p
from scripts.freeze_n4_family_full972 import check_implementation


def pairing_report(phase: str, eligible: list[str]) -> dict:
    reports = {
        role: json.loads((p.WORK / phase / role / "TRAIN_DONE.json").read_text())
        for role in ("canonical", *eligible)
    }
    canonical = reports["canonical"]
    comparisons = {}
    for candidate in eligible:
        other = reports[candidate]
        gates = {
            "same_initial_state": other["initial_state_sha256"] == canonical["initial_state_sha256"],
            "same_sampling_weights": other["sampling_weight_sha256"] == canonical["sampling_weight_sha256"],
            "same_sampler_all_epochs": [x["sampler_sha256"] for x in other["history"]]
            == [x["sampler_sha256"] for x in canonical["history"]],
            "same_augmentation_rng_before_all_epochs": [x["augmentation_rng_before_sha256"] for x in other["history"]]
            == [x["augmentation_rng_before_sha256"] for x in canonical["history"]],
            "same_augmentation_rng_after_all_epochs": [x["augmentation_rng_after_sha256"] for x in other["history"]]
            == [x["augmentation_rng_after_sha256"] for x in canonical["history"]],
            "same_first_augmented_label": other["history"][0]["first_batch"]["label"]
            == canonical["history"][0]["first_batch"]["label"],
            "same_teacher_checkpoints": other["teacher_checkpoints"] == canonical["teacher_checkpoints"],
            "different_input_cache": other["cache_index_sha256"] != canonical["cache_index_sha256"],
            "both_engineering_gates": other["all_gates_passed"] and canonical["all_gates_passed"],
        }
        comparisons[candidate] = {"gates": gates, "all_gates_passed": all(gates.values())}
    result = {
        "schema": "aflow.n4_family_full972.pairing",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
        "eligible_candidates": eligible,
        "comparisons": comparisons,
        "all_pairing_gates_passed": all(x["all_gates_passed"] for x in comparisons.values()),
        "new_evaluation": False,
    }
    return result


def launch_wave(phase: str, roles: list[str], env: dict, logs: Path, deadline) -> None:
    children = []
    handles = []
    try:
        for gpu, role in enumerate(roles):
            handle = (logs / f"{role}.log").open("x")
            handles.append(handle)
            command = [
                sys.executable,
                str(ROOT / "scripts/train_n4_family_full972.py"),
                "--role", role,
                "--device", f"cuda:{gpu}",
            ]
            if phase == "smoke":
                command.append("--smoke")
            child = subprocess.Popen(
                command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT
            )
            children.append((role, child))
            print(f"started Full972 {phase}/{role} pid={child.pid}", flush=True)
        while any(child.poll() is None for _, child in children):
            if datetime.now(timezone.utc) >= deadline:
                raise RuntimeError("30h deadline reached during Full972")
            if any(child.poll() not in (None, 0) for _, child in children):
                raise RuntimeError("Full972 child failed; inspect role log")
            time.sleep(3)
        if any(child.returncode != 0 for _, child in children):
            raise RuntimeError("Full972 child failed")
        for role, _ in children:
            report = json.loads((p.WORK / phase / role / "TRAIN_DONE.json").read_text())
            if report.get("all_gates_passed") is not True:
                raise RuntimeError(f"Full972 {role} engineering gates failed")
        print(f"completed Full972 {phase} wave {roles}", flush=True)
    finally:
        for _, child in children:
            if child.poll() is None:
                child.terminate()
        for _, child in children:
            if child.poll() is None:
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
        for handle in handles:
            handle.close()


def main(phase: str) -> None:
    check_implementation()
    eligible, admission = p.verify_admission(require_time=phase == "formal")
    if phase == "formal":
        smoke = json.loads((p.RECORD / "FULL972_GPU_SMOKE_RECEIPT.json").read_text())
        if smoke.get("status") != "passed" or smoke.get("eligible_candidates") != eligible:
            raise RuntimeError("matching Full972 smoke did not pass")
    logs = p.WORK / phase / "driver_logs"
    logs.mkdir(parents=True, exist_ok=False)
    launch = {
        "schema": "aflow.n4_family_full972.launch",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
        "eligible_candidates": eligible,
        "admission_sha256": p.sha(p.RECORD / "FULL972_ADMISSION_CHECK.json"),
        "admission_created_utc": admission["created_utc"],
        "implementation_sha256": p.sha(p.RECORD / "FULL972_IMPLEMENTATION.json"),
        "automatic_evaluation": False,
    }
    p.write_new(p.RECORD / f"FULL972_{phase.upper()}_LAUNCH_RECEIPT.json", launch)
    env = dict(
        os.environ,
        PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="1",
        MPLCONFIGDIR="/tmp/aflow_matplotlib_cache",
    )
    # Two independent teachers occupy the two GPUs together.
    launch_wave(phase, ["history", "ema"], env, logs, p.DEADLINE)
    # The same-seed canonical control is paired first with the fixed-priority candidate.
    student_roles = ["canonical", *eligible]
    for start in range(0, len(student_roles), 2):
        launch_wave(phase, student_roles[start : start + 2], env, logs, p.DEADLINE)
    pairing = pairing_report(phase, eligible)
    pair_path = p.RECORD / f"FULL972_{phase.upper()}_PAIRING.json"
    p.write_new(pair_path, pairing)
    if not pairing["all_pairing_gates_passed"]:
        raise RuntimeError("Full972 paired-control gates failed")
    if phase == "smoke":
        p.write_new(
            p.RECORD / "FULL972_GPU_SMOKE_RECEIPT.json",
            {
                "schema": "aflow.n4_family_full972.gpu_smoke",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "status": "passed",
                "eligible_candidates": eligible,
                "pairing_sha256": p.sha(pair_path),
                "all_pairing_gates_passed": True,
            },
        )
    p.write_new(
        p.RECORD / f"FULL972_{phase.upper()}_RUNNER_DONE.json",
        {
            "schema": "aflow.n4_family_full972.runner_done",
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "phase": phase,
            "status": "complete",
            "eligible_candidates": eligible,
            "pairing_sha256": p.sha(pair_path),
            "new_evaluation": False,
        },
    )
    print(f"Full972 {phase} complete; no validation/test was launched", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("smoke", "formal"))
    main(parser.parse_args().phase)
