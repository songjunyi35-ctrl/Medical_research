#!/usr/bin/env python3
"""Prepare the frozen patient-disjoint UCSD-PTGBM confirmation cohort.

This is an isolated external-evidence entry point.  It accepts only the four
official TCIA GitHub releases frozen in the preregistration, pairs exact
post-contrast T1 and BraTS segmentation asset names, subtracts every subject in
the existing 1,350-case BraTS split, downloads only the frozen remainder, and
builds the historical N4 + image-only brain-crop 128^3 cache.

The commands are deliberately separate so ``inventory`` can be frozen and
hashed before ``download`` exposes eligible label voxels.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Iterable
from urllib.request import Request, urlopen

import nibabel as nib
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sammed3d.t1ce_binary import prepare_t1ce_supervised_arrays, residual_tumor_target  # noqa: E402
from sammed3d.t1ce_wt_cached import CACHE_SCHEMA, MODEL_SHAPE  # noqa: E402
from sammed3d.t1ce_wt_training import _load_canonical_nifti  # noqa: E402
from scripts.build_t1ce_wt_cache import n4_correct  # noqa: E402


SCHEMA = "aflow.ucsd_ptgbm.patient_disjoint_external_confirmation"
EXPECTED_RELEASES = {
    "ucsd-ptgbm": 302631741,
    "ucsd-ptgbm_1": 302636016,
    "ucsd-ptgbm_2": 302639054,
    "ucsd-ptgbm_3": 302642815,
}
PARTICIPANTS_SHA256 = "718854757eba9332f0e2872e93b88809987aff204a3d01350e138f8ae0a5960e"
TRAIN_SHA256 = "aa2e55915e2a4bc7c2af2e95d6e08f2199397694fe4a92ea66999b534a22f63d"
TEST_SHA256 = "7b91690c63a5eb359be105b3cc7c4ad563d41fca2b2c84e521251b7ed006b8d2"
EXPECTED_PARTICIPANT_ROWS = 243
EXPECTED_PARTICIPANT_SUBJECTS = 178
EXPECTED_MAIN_CASES = 184
EXPECTED_MAIN_SUBJECTS = 136
EXPECTED_OVERLAPPING_MAIN_CASES = 109
EXPECTED_ELIGIBLE_CASES = 75
EXPECTED_ELIGIBLE_SUBJECTS = 53
IMAGE_RE = re.compile(r"^(?P<subject>sub-\d{4})_ses-(?P<session>\d{2})_run-post_T1w\.nii\.gz$")
LABEL_RE = re.compile(r"^(?P<subject>sub-\d{4})_ses-(?P<session>\d{2})_BraTS_tumor_seg\.nii\.gz$")
BRATS_CASE_RE = re.compile(r"^BraTS-GLI-\d{5}-\d{3}$")
_CFG: dict[str, Any] = {}


class ExternalPreparationError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_jsonl(path: Path, expected_sha256: str) -> list[dict[str, Any]]:
    if sha256_file(path) != expected_sha256:
        raise ExternalPreparationError(f"frozen manifest SHA-256 mismatch: {path}")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return rows


def read_participants(path: Path) -> tuple[dict[str, str], dict[str, Any]]:
    if sha256_file(path) != PARTICIPANTS_SHA256:
        raise ExternalPreparationError("official participants.tsv SHA-256 mismatch")
    with path.open(encoding="utf-8", errors="replace", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    mapping: dict[str, str] = {}
    brats_ids: list[str] = []
    for row in rows:
        local_id = str(row.get("ID", "")).strip()
        brats_id = str(row.get("BraTS ID", "")).strip()
        match = re.fullmatch(r"(?P<subject>sub-\d{4})_(?P<session>\d{2})", local_id)
        if match is None or BRATS_CASE_RE.fullmatch(brats_id) is None:
            raise ExternalPreparationError(f"invalid participants mapping: {local_id!r} -> {brats_id!r}")
        key = f"{match.group('subject')}_ses-{match.group('session')}"
        if key in mapping or brats_id in brats_ids:
            raise ExternalPreparationError("participants contains duplicate local or BraTS IDs")
        mapping[key] = brats_id
        brats_ids.append(brats_id)
    subjects = {value.rsplit("-", 1)[0] for value in brats_ids}
    if len(rows) != EXPECTED_PARTICIPANT_ROWS or len(subjects) != EXPECTED_PARTICIPANT_SUBJECTS:
        raise ExternalPreparationError("participants count differs from frozen official contract")
    return mapping, {"rows": len(rows), "subjects": len(subjects)}


def _asset_record(asset: dict[str, Any], release: dict[str, Any], kind: str, key: str) -> dict[str, Any]:
    digest = str(asset.get("digest", ""))
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
        raise ExternalPreparationError(f"asset lacks official SHA-256: {asset.get('name')}")
    url = str(asset.get("browser_download_url", ""))
    expected_prefix = f"https://github.com/data-nih/tcia/releases/download/{release['tag_name']}/"
    if not url.startswith(expected_prefix):
        raise ExternalPreparationError(f"asset download URL escaped official release: {url}")
    size = int(asset.get("size", -1))
    if size <= 0:
        raise ExternalPreparationError(f"asset has invalid size: {asset.get('name')}")
    return {
        "kind": kind,
        "key": key,
        "name": str(asset["name"]),
        "size_bytes": size,
        "sha256": digest.split(":", 1)[1],
        "url": url,
        "release_tag": str(release["tag_name"]),
        "release_id": int(release["id"]),
    }


def read_release_assets(paths: Iterable[Path]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    releases: dict[str, dict[str, Any]] = {}
    pairs: dict[str, dict[str, Any]] = {}
    for path in paths:
        release = json.loads(path.read_text())
        tag = str(release.get("tag_name", ""))
        release_id = int(release.get("id", -1))
        if tag not in EXPECTED_RELEASES or release_id != EXPECTED_RELEASES[tag]:
            raise ExternalPreparationError(f"unexpected release identity: {tag}/{release_id}")
        if tag in releases:
            raise ExternalPreparationError(f"duplicate release: {tag}")
        html_url = str(release.get("html_url", ""))
        if html_url != f"https://github.com/data-nih/tcia/releases/tag/{tag}":
            raise ExternalPreparationError(f"unexpected release URL: {html_url}")
        releases[tag] = {
            "id": release_id,
            "assets": len(release.get("assets", [])),
            "api_source_sha256": sha256_file(path),
        }
        for asset in release.get("assets", []):
            name = str(asset.get("name", ""))
            image_match = IMAGE_RE.fullmatch(name)
            label_match = LABEL_RE.fullmatch(name)
            if image_match is None and label_match is None:
                continue
            match = image_match or label_match
            assert match is not None
            key = f"{match.group('subject')}_ses-{match.group('session')}"
            kind = "image" if image_match is not None else "label"
            pair = pairs.setdefault(key, {})
            if kind in pair:
                raise ExternalPreparationError(f"duplicate {kind} asset for {key}")
            pair[kind] = _asset_record(asset, release, kind, key)
    if set(releases) != set(EXPECTED_RELEASES):
        raise ExternalPreparationError("not all four frozen main releases were supplied")
    incomplete = {key: sorted(set(("image", "label")) - set(pair)) for key, pair in pairs.items() if set(pair) != {"image", "label"}}
    if incomplete:
        raise ExternalPreparationError(f"incomplete image/label asset pairs: {list(incomplete.items())[:5]}")
    if len(pairs) != EXPECTED_MAIN_CASES:
        raise ExternalPreparationError(f"main release has {len(pairs)} pairs, expected {EXPECTED_MAIN_CASES}")
    return pairs, releases


def build_inventory(args: argparse.Namespace) -> dict[str, Any]:
    mapping, participant_counts = read_participants(args.participants.resolve())
    pairs, release_receipts = read_release_assets([path.resolve() for path in args.release_json])
    missing = sorted(set(pairs) - set(mapping))
    if missing:
        raise ExternalPreparationError(f"main assets missing participant mappings: {missing[:5]}")
    main_subjects = {mapping[key].rsplit("-", 1)[0] for key in pairs}
    if len(main_subjects) != EXPECTED_MAIN_SUBJECTS:
        raise ExternalPreparationError("main release subject count differs from frozen contract")

    train_rows = read_jsonl(args.train_manifest.resolve(), TRAIN_SHA256)
    test_rows = read_jsonl(args.test_manifest.resolve(), TEST_SHA256)
    if len(train_rows) != 1080 or len(test_rows) != 270:
        raise ExternalPreparationError("existing BraTS split count mismatch")
    exposed_cases = {str(row["case_id"]) for row in train_rows + test_rows}
    exposed_subjects = {str(row["subject_id"]) for row in train_rows + test_rows}
    if len(exposed_cases) != 1350:
        raise ExternalPreparationError("existing BraTS union is not 1,350 unique cases")

    selected: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for key in sorted(pairs):
        brats_id = mapping[key]
        brats_subject = brats_id.rsplit("-", 1)[0]
        row = {
            "ucsd_case_id": key,
            "ucsd_subject_id": key.split("_ses-", 1)[0],
            "session_id": "ses-" + key.split("_ses-", 1)[1],
            "brats_case_id": brats_id,
            "brats_subject_id": brats_subject,
            "image": pairs[key]["image"],
            "label": pairs[key]["label"],
        }
        if brats_subject in exposed_subjects:
            row["exclusion_reason"] = "subject_present_in_existing_1350_case_union"
            excluded.append(row)
        else:
            selected.append(row)
    if len(excluded) != EXPECTED_OVERLAPPING_MAIN_CASES:
        raise ExternalPreparationError("overlapping main case count differs from frozen contract")
    if len(selected) != EXPECTED_ELIGIBLE_CASES:
        raise ExternalPreparationError("eligible case count differs from frozen contract")
    selected_subjects = {row["brats_subject_id"] for row in selected}
    if len(selected_subjects) != EXPECTED_ELIGIBLE_SUBJECTS:
        raise ExternalPreparationError("eligible subject count differs from frozen contract")
    if exposed_cases & {row["brats_case_id"] for row in selected}:
        raise ExternalPreparationError("eligible cases overlap existing cases")
    if exposed_subjects & selected_subjects:
        raise ExternalPreparationError("eligible subjects overlap existing subjects")

    inventory = {
        "schema": SCHEMA + ".inventory",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "repository": "data-nih/tcia",
            "official_collection": "UCSD-PTGBM",
            "release_receipts": release_receipts,
            "participants_path": str(args.participants.resolve()),
            "participants_sha256": PARTICIPANTS_SHA256,
            "participants": participant_counts,
            "separate_brats_challenge_test_package_used": False,
        },
        "existing_exposure": {
            "train_manifest": str(args.train_manifest.resolve()),
            "train_manifest_sha256": TRAIN_SHA256,
            "train_cases": len(train_rows),
            "test_manifest": str(args.test_manifest.resolve()),
            "test_manifest_sha256": TEST_SHA256,
            "reused_test_cases": len(test_rows),
            "union_cases": len(exposed_cases),
            "union_subjects": len(exposed_subjects),
        },
        "main_release": {
            "cases": len(pairs),
            "subjects": len(main_subjects),
            "overlapping_cases": len(excluded),
        },
        "selection": {
            "rule": "exclude every UCSD subject if any of its BraTS timepoints appears in the existing 1,350-case union",
            "cases": len(selected),
            "subjects": len(selected_subjects),
            "case_overlap": 0,
            "subject_overlap": 0,
            "selected_assets_bytes": sum(row[kind]["size_bytes"] for row in selected for kind in ("image", "label")),
            "selected": selected,
        },
        "excluded": {
            "cases": len(excluded),
            "subjects": len({row["brats_subject_id"] for row in excluded}),
            "case_ids": [row["brats_case_id"] for row in excluded],
        },
        "access": {
            "metadata_access": True,
            "eligible_image_voxel_access": False,
            "eligible_label_voxel_access": False,
            "model_inference": False,
        },
    }
    inventory["selection"]["selected_identity_sha256"] = stable_sha256(
        [{k: row[k] for k in ("ucsd_case_id", "brats_case_id", "brats_subject_id")} for row in selected]
    )
    write_json_atomic(args.output.resolve(), inventory)
    return inventory


def _download_one(url: str, output: Path, expected_sha256: str, expected_size: int) -> str:
    if output.exists():
        if output.stat().st_size == expected_size and sha256_file(output) == expected_sha256:
            return "already_verified"
        raise ExternalPreparationError(f"refusing mismatched existing download: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    if temporary.exists():
        temporary.unlink()
    request = Request(url, headers={"User-Agent": "AFlow-UCSD-external-audit/1"})
    digest = hashlib.sha256()
    size = 0
    try:
        with urlopen(request, timeout=120) as source, temporary.open("xb") as target:
            while True:
                block = source.read(1 << 20)
                if not block:
                    break
                target.write(block)
                digest.update(block)
                size += len(block)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    if size != expected_size or digest.hexdigest() != expected_sha256:
        temporary.unlink()
        raise ExternalPreparationError(f"download digest/size mismatch: {output.name}")
    os.replace(temporary, output)
    return "downloaded"


def download(args: argparse.Namespace) -> dict[str, Any]:
    inventory_path = args.inventory.resolve()
    if sha256_file(inventory_path) != args.expected_inventory_sha256:
        raise ExternalPreparationError("inventory SHA-256 mismatch")
    inventory = json.loads(inventory_path.read_text())
    selected = inventory["selection"]["selected"]
    if len(selected) != EXPECTED_ELIGIBLE_CASES or inventory["access"]["eligible_label_voxel_access"] is not False:
        raise ExternalPreparationError("inventory is not the frozen pre-access artifact")
    output = args.output_dir.resolve()
    if (output / "DOWNLOAD_RECEIPT.json").exists():
        raise ExternalPreparationError("refusing to overwrite completed download receipt")
    started = time.time()
    records = []
    for index, row in enumerate(selected, start=1):
        case_dir = output / row["ucsd_case_id"]
        record = {"ucsd_case_id": row["ucsd_case_id"], "files": {}}
        for kind in ("image", "label"):
            asset = row[kind]
            destination = case_dir / asset["name"]
            status = _download_one(asset["url"], destination, asset["sha256"], int(asset["size_bytes"]))
            record["files"][kind] = {
                "path": str(destination), "sha256": sha256_file(destination),
                "size_bytes": destination.stat().st_size, "status": status,
            }
        records.append(record)
        print(f"[{index}/{len(selected)}] verified {row['ucsd_case_id']}", flush=True)
    receipt = {
        "schema": SCHEMA + ".download_receipt",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inventory": str(inventory_path),
        "inventory_sha256": args.expected_inventory_sha256,
        "cases": len(records),
        "subjects": len({row["brats_subject_id"] for row in selected}),
        "downloaded_bytes": sum(item["files"][kind]["size_bytes"] for item in records for kind in ("image", "label")),
        "elapsed_seconds": time.time() - started,
        "records": records,
        "access": {"eligible_image_voxel_access": False, "eligible_label_voxel_access": False, "files_downloaded_but_not_opened_as_nifti": True},
    }
    write_json_atomic(output / "DOWNLOAD_RECEIPT.json", receipt)
    return receipt


def _worker_init(cfg: dict[str, Any]) -> None:
    for variable in ("OMP_NUM_THREADS", "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[variable] = "1"
    import SimpleITK as sitk
    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1)
    _CFG.update(cfg)
    _CFG["images_fd"] = os.open(cfg["images_path"], os.O_WRONLY)
    _CFG["labels_fd"] = os.open(cfg["labels_path"], os.O_WRONLY)


def _prepare_case(task: tuple[int, dict[str, Any]]) -> tuple[int, dict[str, Any]]:
    slot, row = task
    case_id = row["brats_case_id"]
    image_path = Path(row["local_image_path"])
    label_path = Path(row["local_label_path"])
    image_nifti = _load_canonical_nifti(image_path, f"{case_id} UCSD T1 post")
    label_nifti = _load_canonical_nifti(label_path, f"{case_id} UCSD BraTS segmentation")
    if image_nifti.shape != label_nifti.shape or not np.allclose(image_nifti.affine, label_nifti.affine, rtol=0.0, atol=1e-5):
        raise ExternalPreparationError(f"{case_id} image/label geometry mismatch")
    image = np.ascontiguousarray(np.asanyarray(image_nifti.dataobj))
    raw_label = np.ascontiguousarray(np.asanyarray(label_nifti.dataobj))
    if not np.isfinite(image).all():
        raise ExternalPreparationError(f"{case_id} image contains non-finite values")
    rounded = np.rint(raw_label)
    if not np.isfinite(raw_label).all() or not np.allclose(raw_label, rounded, rtol=0.0, atol=1e-5):
        raise ExternalPreparationError(f"{case_id} label is non-finite or non-integer")
    raw_values = sorted(int(value) for value in np.unique(rounded))
    if not set(raw_values) <= {0, 1, 2, 3, 4}:
        raise ExternalPreparationError(f"{case_id} unexpected raw labels: {raw_values}")
    spacing = nib.affines.voxel_sizes(image_nifti.affine)
    native_shape = [int(value) for value in image.shape]
    image = n4_correct(image, 2)
    nonzero = np.nonzero(image > 0)
    if any(axis.size == 0 for axis in nonzero):
        raise ExternalPreparationError(f"{case_id} has no positive image support after N4")
    origin = [int(axis.min()) for axis in nonzero]
    stop = [int(axis.max()) + 1 for axis in nonzero]
    box = tuple(slice(origin[axis], stop[axis]) for axis in range(3))
    cropped_image = image[box]
    cropped_label = raw_label[box]
    source_target = residual_tumor_target(cropped_label)
    source_wt = int(source_target.sum())
    view = prepare_t1ce_supervised_arrays(case_id, cropped_image, cropped_label, spacing, output_shape=MODEL_SHAPE)
    cached_image = np.ascontiguousarray(view.image.float().numpy()[0], dtype=np.float32)
    cached_label = np.ascontiguousarray(view.target.numpy()[0], dtype=np.uint8)
    if cached_image.shape != MODEL_SHAPE or cached_label.shape != MODEL_SHAPE or not np.isfinite(cached_image).all():
        raise ExternalPreparationError(f"{case_id} invalid cached tensors")
    voxels = int(np.prod(MODEL_SHAPE))
    os.pwrite(_CFG["images_fd"], cached_image.tobytes(), slot * voxels * 4)
    os.pwrite(_CFG["labels_fd"], cached_label.tobytes(), slot * voxels)
    return slot, {
        "case_id": case_id,
        "subject_id": row["brats_subject_id"],
        "ucsd_case_id": row["ucsd_case_id"],
        "native_shape": native_shape,
        "canonical_affine": np.asarray(image_nifti.affine).tolist(),
        "spacing_mm": [float(value) for value in spacing],
        "crop_origin": origin,
        "crop_extent": [stop[axis] - origin[axis] for axis in range(3)],
        "raw_labels": raw_values,
        "source_wt_voxels": source_wt,
        "resized_wt_voxels": int(cached_label.sum()),
        "source_wt_empty": source_wt == 0,
        "resized_wt_empty": int(cached_label.sum()) == 0,
    }


def build_cache(args: argparse.Namespace) -> dict[str, Any]:
    inventory_path = args.inventory.resolve()
    receipt_path = args.download_dir.resolve() / "DOWNLOAD_RECEIPT.json"
    if sha256_file(inventory_path) != args.expected_inventory_sha256:
        raise ExternalPreparationError("inventory SHA-256 mismatch")
    if sha256_file(receipt_path) != args.expected_download_receipt_sha256:
        raise ExternalPreparationError("download receipt SHA-256 mismatch")
    inventory = json.loads(inventory_path.read_text())
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("inventory_sha256") != args.expected_inventory_sha256 or receipt.get("cases") != EXPECTED_ELIGIBLE_CASES:
        raise ExternalPreparationError("download receipt is not bound to frozen inventory")
    receipt_by_case = {row["ucsd_case_id"]: row for row in receipt["records"]}
    rows = []
    for row in inventory["selection"]["selected"]:
        downloaded = receipt_by_case.get(row["ucsd_case_id"])
        if downloaded is None:
            raise ExternalPreparationError(f"missing download receipt for {row['ucsd_case_id']}")
        local = dict(row)
        for kind in ("image", "label"):
            path = Path(downloaded["files"][kind]["path"])
            if sha256_file(path) != row[kind]["sha256"]:
                raise ExternalPreparationError(f"downloaded {kind} digest mismatch for {row['ucsd_case_id']}")
            local[f"local_{kind}_path"] = str(path)
        rows.append(local)
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty cache: {output}")
    output.mkdir(parents=True, exist_ok=True)
    images_path = output / "images.f32"
    labels_path = output / "labels.u8"
    np.memmap(images_path, dtype=np.float32, mode="w+", shape=(len(rows), *MODEL_SHAPE)).flush()
    np.memmap(labels_path, dtype=np.uint8, mode="w+", shape=(len(rows), *MODEL_SHAPE)).flush()
    cfg = {"images_path": str(images_path), "labels_path": str(labels_path)}
    started = time.time()
    results = []
    with mp.get_context("spawn").Pool(processes=args.workers, initializer=_worker_init, initargs=(cfg,)) as pool:
        for completed, result in enumerate(pool.imap_unordered(_prepare_case, enumerate(rows), chunksize=1), start=1):
            results.append(result)
            print(f"[{completed}/{len(rows)}] cached external case", flush=True)
    results.sort(key=lambda item: item[0])
    qc = [item[1] for item in results]
    if [item["case_id"] for item in qc] != [row["brats_case_id"] for row in rows]:
        raise ExternalPreparationError("cache order differs from frozen inventory")
    if any(item["source_wt_empty"] != item["resized_wt_empty"] for item in qc):
        raise ExternalPreparationError("128^3 resize changed WT empty/non-empty status")
    index = {
        "schema": CACHE_SCHEMA,
        "schema_version": 1,
        "external_schema": SCHEMA + ".cache",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(inventory_path),
        "manifest_sha256": args.expected_inventory_sha256,
        "download_receipt": str(receipt_path),
        "download_receipt_sha256": args.expected_download_receipt_sha256,
        "count": len(rows),
        "subjects": len({row["brats_subject_id"] for row in rows}),
        "case_ids": [row["brats_case_id"] for row in rows],
        "subject_ids": [row["brats_subject_id"] for row in rows],
        "model_shape": list(MODEL_SHAPE),
        "image_dtype": "float32",
        "label_dtype": "uint8",
        "split": "external_confirmation",
        "test_access": True,
        "external_confirmation": True,
        "not_pristine_blind": False,
        "preprocessing": {
            "order": ["canonical_RAS", "N4_shrink_2", "image_nonzero_bounding_box_crop", "source_nonzero_population_zscore", "resize_to_128_cubic_grid"],
            "spatial": "braincrop_resize",
            "output_resolution": 128,
            "image_interpolation": "trilinear_align_corners_false",
            "label_interpolation": "nearest-exact",
            "normalization": "source nonzero-voxel population z-score before resize",
            "uses_gt_for_localization": False,
            "n4_bias_field_correction": True,
            "n4_shrink_factor": 2,
        },
        "target": {"name": "WT", "foreground_raw_labels": [1, 2, 3], "excluded_raw_labels": [4]},
        "qc": {
            "cases": qc,
            "source_wt_empty_cases": sum(item["source_wt_empty"] for item in qc),
            "resized_wt_empty_cases": sum(item["resized_wt_empty"] for item in qc),
            "all_raw_labels": sorted({value for item in qc for value in item["raw_labels"]}),
            "all_finite": True,
            "image_label_geometry_exact_after_canonicalization": True,
        },
        "images_sha256": sha256_file(images_path),
        "labels_sha256": sha256_file(labels_path),
        "builder": str(Path(__file__).resolve().relative_to(ROOT)),
        "builder_sha256": sha256_file(Path(__file__).resolve()),
        "elapsed_seconds": time.time() - started,
        "access": {"eligible_image_voxel_access": True, "eligible_label_voxel_access": True, "model_inference": False},
    }
    write_json_atomic(output / "index.json", index)
    return index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    inventory = sub.add_parser("inventory")
    inventory.add_argument("--participants", type=Path, required=True)
    inventory.add_argument("--release-json", type=Path, action="append", required=True)
    inventory.add_argument("--train-manifest", type=Path, required=True)
    inventory.add_argument("--test-manifest", type=Path, required=True)
    inventory.add_argument("--output", type=Path, required=True)
    downloader = sub.add_parser("download")
    downloader.add_argument("--inventory", type=Path, required=True)
    downloader.add_argument("--expected-inventory-sha256", required=True)
    downloader.add_argument("--output-dir", type=Path, required=True)
    cache = sub.add_parser("build-cache")
    cache.add_argument("--inventory", type=Path, required=True)
    cache.add_argument("--expected-inventory-sha256", required=True)
    cache.add_argument("--download-dir", type=Path, required=True)
    cache.add_argument("--expected-download-receipt-sha256", required=True)
    cache.add_argument("--output-dir", type=Path, required=True)
    cache.add_argument("--workers", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "inventory":
        result = build_inventory(args)
    elif args.command == "download":
        result = download(args)
    elif args.command == "build-cache":
        if args.workers <= 0:
            raise ValueError("workers must be positive")
        result = build_cache(args)
    else:
        raise AssertionError(args.command)
    print(json.dumps({"schema": result["schema"], "cases": result.get("cases", result.get("selection", {}).get("cases")), "output_complete": True}, indent=2))


if __name__ == "__main__":
    main()
