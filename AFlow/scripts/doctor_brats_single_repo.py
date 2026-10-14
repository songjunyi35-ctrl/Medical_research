#!/usr/bin/env python3
"""Fail-fast audit for the repo-local BraTS / SAM-Med3D research checkout."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ASSETS = {
    "base SAM-Med3D checkpoint": (
        Path("artifacts/checkpoints/sam_med3d_turbo.pth"),
        "899a46d04d3b70f723282ceb489149373558bf0aaba389a346f5ab57da5cdd3c",
    ),
    "upstream-literal baseline checkpoint": (
        Path("workspace/BraTS/sammed3d_t1ce_wt/")
        / "sammed3d_upstreamliteral11_officialval5_va_a100x2_train972_val108_e200_s2023"
        / "sam_model_val_official_best5.pth",
        "c0c85a0fb90f6301344053f97cabe8a4972bd9febd4c6ae248f3a0908598fe31",
    ),
    "baseline test report": (
        Path("workspace/BraTS/sammed3d_t1ce_wt/")
        / "sammed3d_upstreamliteral11_officialval5_va_a100x2_train972_val108_e200_s2023"
        / "test270_epoch21_official15_upstreamliteral5_s2023_20260812"
        / "test_report.json",
        "7bd59843459876bd006230b24bc111cdfd726896cb3e6748b54b24c0a9659b63",
    ),
    "coarse checkpoint for residual refiner": (
        Path("workspace/BraTS/sammed3d_t1ce_wt/")
        / "sammed3d_native_gtcrop_fullunfreeze_ddp2_train972_val108_e100_s20260805"
        / "sam_model_val_dice_best.pth",
        "42f96a0a2ade9d98d9db897a11a5a285f7a6351a7693c6884d6ac97a29dfde52",
    ),
    "legacy residual-refiner checkpoint": (
        Path("workspace/BraTS/sammed3d_t1ce_wt/")
        / "residual_refiner_stage2_gtcrop_h100_train972_val108_e60_s20260810_v2"
        / "refiner_val_dice_best.pth",
        "4513ea4544ae153064575ec2ed5c56e1b78adf5e89e9d094754227b3bd429f21",
    ),
    "robust refiner seed 20260813": (
        Path("workspace/BraTS/sammed3d_t1ce_wt/")
        / "residual_refiner_robustshift4_graymean_paired_a100_20260813"
        / "robustshift4_control/refiner_val_dice_best.pth",
        "011a7b0f4a9ee193883662f03fc6d8a54cef9fd3b4c08c2871cdddbdfbe73f6a",
    ),
    "robust refiner seed 20260814": (
        Path("workspace/BraTS/sammed3d_t1ce_wt/")
        / "residual_refiner_robustshift4_three_seed_ensemble_20260813"
        / "robustshift4_seed20260814/refiner_val_dice_best.pth",
        "c94f32c04cb89f51e1b83ae157bbce1d89371aa21b4267164e1ed645fa4d4cbc",
    ),
    "robust refiner seed 20260815": (
        Path("workspace/BraTS/sammed3d_t1ce_wt/")
        / "residual_refiner_robustshift4_three_seed_ensemble_20260813"
        / "robustshift4_seed20260815/refiner_val_dice_best.pth",
        "c5d2b48db2666f097cb0d86e78c20ea95d163c07f98401714d25d755e99aceb0",
    ),
    "final robust validation result": (
        Path("workspace/BraTS/sammed3d_t1ce_wt/")
        / "FINAL_VALIDATION_RESULT_RESIDUAL_REFINER_ROBUST_ENSEMBLE_20260813_V2.json",
        "74d5b3018be637bb1bd05adea0ceb99cdaeb38b5bb2af5f52b4566301df6f186",
    ),
    "oracle residual-refiner test result": (
        Path("workspace/BraTS/sammed3d_t1ce_wt/")
        / "FINAL_TEST_RESULT_RESIDUAL_REFINER_20260813.json",
        "149fc9a89d5877de9efa6e52f1a8a459d7c918eb2aabf06a3872846bd20fdcdd",
    ),
}
EXPECTED_MANIFESTS = {
    "official train": (
        Path("data/datasets/brats2024_t1ce_va_hdfs_20260812_v1/train_t1ce_wt.jsonl"),
        972,
        "6728d8d5e6846f90f6e5d0ad47acc4f2706eed42763ace4e228abcbbd6a4cf65",
    ),
    "official validation": (
        Path("data/datasets/brats2024_t1ce_va_hdfs_20260812_v1/validation_t1ce_wt.jsonl"),
        108,
        "f85f7df19908c7ab5bc7aafc2ce16fab1ff9786dbf3ea4ee72b97721888d35fb",
    ),
    "sealed test": (
        Path("data/datasets/brats2024_t1ce_va_hdfs_20260812_v1/sealed_test/test_cases_t1ce.jsonl"),
        270,
        "a5ddc0ed2c0b269431758b14061da3c6163dd0c48133b60a2ef8efd7f23aa8f9",
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def add_check(checks: list[dict], name: str, ok: bool, detail: str) -> None:
    checks.append({"name": name, "ok": bool(ok), "detail": detail})


def check_vendor(checks: list[dict]) -> None:
    vendor = Path(os.environ.get("AFLOW_SAMMED3D_ROOT", ROOT / "third_party/SAM-Med3D")).resolve()
    manifest = vendor / "SHA256SUMS"
    failures = []
    if manifest.is_file():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            expected, relative = line.split(maxsplit=1)
            path = vendor / relative.strip()
            actual = sha256_file(path) if path.is_file() else "MISSING"
            if actual != expected:
                failures.append(f"{relative}: {actual} != {expected}")
    else:
        failures.append(f"missing {manifest}")
    add_check(checks, "vendored source SHA256SUMS", not failures, "; ".join(failures) or str(vendor))

    sys.path.insert(0, str(vendor))
    import segment_anything  # noqa: PLC0415
    from sammed3d import engine  # noqa: PLC0415

    module_path = Path(segment_anything.__file__).resolve()
    inside_vendor = module_path.is_relative_to(vendor)
    engine_vendor = Path(engine.SAMMED3D_ROOT).resolve() == vendor
    add_check(
        checks,
        "vendored segment_anything import",
        inside_vendor and engine_vendor,
        f"module={module_path}; engine_root={engine.SAMMED3D_ROOT}",
    )

    upstream_expected = {
        vendor / "train.py": "474d9765e29c0196e9cf1da2242869c2cbd6970632d9f113357d38a8094e1e3f",
        vendor / "inference.py": "5a31aff32805f297eb5632b4362d1aa60bde4491d168868bc750a43e332c3960",
        vendor / "utils/click_method.py": "0838972fb2f33de579a2652d3225f862d84dc6df113b152ea6018730c9b792d7",
    }
    bad = [str(path) for path, expected in upstream_expected.items() if sha256_file(path) != expected]
    add_check(checks, "upstream-literal source binding", not bad, "; ".join(bad) or "3/3 SHA correct")


def check_gpu(checks: list[dict], allow_no_gpu: bool) -> None:
    import torch  # noqa: PLC0415

    names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    valid = torch.cuda.is_available() and len(names) >= 2 and all("A100" in n for n in names[:2])
    add_check(
        checks,
        "torch and two A100",
        valid or allow_no_gpu,
        f"python={sys.executable}; torch={torch.__version__}; cuda={torch.version.cuda}; devices={names}",
    )


def check_assets(checks: list[dict]) -> None:
    for name, (relative, expected) in EXPECTED_ASSETS.items():
        path = ROOT / relative
        actual = sha256_file(path) if path.is_file() else "MISSING"
        add_check(checks, name, actual == expected, f"{relative}; sha256={actual}")


def check_manifests_and_data(checks: list[dict], allow_missing_data: bool) -> None:
    sample_paths: list[Path] = []
    for name, (relative, expected_count, expected_sha) in EXPECTED_MANIFESTS.items():
        path = ROOT / relative
        rows = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
        actual = sha256_file(path) if path.is_file() else "MISSING"
        add_check(
            checks,
            f"{name} manifest",
            len(rows) == expected_count and actual == expected_sha,
            f"{relative}; rows={len(rows)}; sha256={actual}",
        )
        if rows:
            row = json.loads(rows[0])
            for key in ("image_path", "label_path", "seg_path", "source_case_dir"):
                if row.get(key):
                    sample_paths.append(Path(row[key]))
            sample_paths.extend(Path(v) for v in row.get("modalities", {}).values())
    missing = [str(path) for path in sample_paths if not path.exists()]
    add_check(
        checks,
        "manifest sample data paths",
        not missing or allow_missing_data,
        "all sampled paths exist" if not missing else "missing on this host: " + "; ".join(missing),
    )


def check_caches(checks: list[dict], allow_missing_refiner_cache: bool) -> None:
    source_root = ROOT / "workspace/BraTS/cache/t1ce_wt_gtcrop_n4_128"
    expected_source = {"train": (972, "99324e6ae3d71f38c058de8cb2c0afc7bee079fbaf6af521257e55d1259f9ebd"),
                       "validation": (108, "179a5ce1a89bf41b07ae633200950ead3d485e01e49ff8d9494437be531461da")}
    for split, (count, expected_sha) in expected_source.items():
        index_path = source_root / split / "index.json"
        actual = sha256_file(index_path) if index_path.is_file() else "MISSING"
        index = json.loads(index_path.read_text()) if index_path.is_file() else {}
        add_check(
            checks,
            f"GT-crop N4 source cache {split}",
            actual == expected_sha and index.get("count") == count,
            f"count={index.get('count')}; index_sha256={actual}",
        )

    refiner_root = ROOT / "workspace/BraTS/cache/t1ce_wt_refiner_graymean_20260813"
    if not refiner_root.exists() and allow_missing_refiner_cache:
        add_check(checks, "persistent frozen-logit refiner cache", True, "missing (temporarily allowed)")
        return
    from scripts.verify_t1ce_wt_refiner_cache import verify_split  # noqa: PLC0415

    details = [verify_split(refiner_root, split, False) for split in ("train", "validation")]
    add_check(checks, "persistent frozen-logit refiner cache", True, json.dumps(details, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-no-gpu", action="store_true", help="development host only")
    parser.add_argument("--allow-missing-data", action="store_true", help="development host only")
    parser.add_argument("--allow-missing-refiner-cache", action="store_true", help="migration only")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    checks: list[dict] = []
    try:
        check_gpu(checks, args.allow_no_gpu)
        check_vendor(checks)
        check_assets(checks)
        check_manifests_and_data(checks, args.allow_missing_data)
        check_caches(checks, args.allow_missing_refiner_cache)
    except Exception as error:  # Keep a readable doctor report instead of a traceback-only failure.
        add_check(checks, "doctor internal exception", False, f"{type(error).__name__}: {error}")
    report = {
        "schema": "aflow.brats.single_repo_doctor",
        "repo_root": str(ROOT),
        "ok": all(item["ok"] for item in checks),
        "checks": checks,
    }
    payload = json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True)
    print(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
