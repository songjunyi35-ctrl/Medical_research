#!/usr/bin/env python3
"""CUDA BF16 rho=0 parity contract for the preregistered GCSAM update."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import torch

from scripts.gcsam_optimizer import PaperGCSAM
from scripts.train_t1ce_wt_capg3d import write_json_atomic
from scripts.train_t1ce_wt_voxel_kd_972_draw_budget import state_dict_sha256


class Tiny3D(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv3d(1, 4, 3, padding=1),
            torch.nn.GroupNorm(2, 4),
            torch.nn.GELU(),
            torch.nn.Dropout3d(0.1),
            torch.nn.Conv3d(4, 1, 1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing repeated CUDA parity output")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA BF16 parity requires CUDA")
    torch.cuda.set_device(device)
    torch.manual_seed(20261005)
    torch.cuda.manual_seed_all(20261005)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    control = Tiny3D().to(device)
    candidate = Tiny3D().to(device)
    candidate.load_state_dict(control.state_dict(), strict=True)
    initial_sha = state_dict_sha256(control.state_dict())
    inputs = torch.randn(2, 1, 16, 16, 16, device=device)
    targets = torch.rand(2, 1, 16, 16, 16, device=device)
    control_optimizer = torch.optim.AdamW(control.parameters(), lr=2e-4, weight_decay=1e-5)
    candidate_base = torch.optim.AdamW(candidate.parameters(), lr=2e-4, weight_decay=1e-5)
    candidate_optimizer = PaperGCSAM(candidate.parameters(), candidate_base, rho=0.0)

    cpu_rng = torch.random.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(device)
    control_optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        control_loss = torch.nn.functional.mse_loss(control(inputs), targets)
    control_loss.backward()
    control_grad = torch.nn.utils.clip_grad_norm_(control.parameters(), 5.0, error_if_nonfinite=True)
    control_optimizer.step()

    torch.random.set_rng_state(cpu_rng)
    torch.cuda.set_rng_state(cuda_rng, device)
    candidate_optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        first_loss = torch.nn.functional.mse_loss(candidate(inputs), targets)
    first_loss.backward()
    first = candidate_optimizer.first_step(zero_grad=True)
    torch.random.set_rng_state(cpu_rng)
    torch.cuda.set_rng_state(cuda_rng, device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        second_loss = torch.nn.functional.mse_loss(candidate(inputs), targets)
    second_loss.backward()
    candidate_grad = torch.nn.utils.clip_grad_norm_(candidate.parameters(), 5.0, error_if_nonfinite=True)
    candidate_optimizer.second_step(zero_grad=True)

    max_parameter_delta = max(
        float((left - right).abs().max())
        for left, right in zip(control.parameters(), candidate.parameters())
    )
    report = {
        "schema": "aflow.gcsam.cuda_bf16_rho0_parity",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "initial_state_sha256": initial_sha,
        "control_loss": float(control_loss),
        "candidate_first_loss": float(first_loss),
        "candidate_second_loss": float(second_loss),
        "control_grad_norm": float(control_grad),
        "candidate_grad_norm": float(candidate_grad),
        "perturbation_norm": float(first["perturbation_norm"]),
        "max_parameter_abs_delta": max_parameter_delta,
        "all_gates_passed": (
            float(control_loss) == float(first_loss) == float(second_loss)
            and float(control_grad) == float(candidate_grad)
            and float(first["perturbation_norm"]) == 0.0
            and max_parameter_delta == 0.0
        ),
        "access": {"train_only_inner": False, "fixed_validation108": False, "outer": False, "confirmation": False, "test270": False},
    }
    write_json_atomic(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["all_gates_passed"]:
        raise RuntimeError("CUDA BF16 rho0 parity failed")


if __name__ == "__main__":
    main()
