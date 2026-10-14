#!/usr/bin/env python3
"""Freeze the OpenNeuro ds007045 v1.0.1 T1ce/segmentation inventory.

This is metadata-only: it queries the versioned GraphQL tree and downloads only
the small versioned README/description/participants/CHANGES text files.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import io
import json
import re
import urllib.request
from pathlib import Path


GRAPHQL = "https://openneuro.org/crn/graphql"
DATASET = "ds007045"
TAG = "1.0.1"
ANNEX_RE = re.compile(r"^SHA256E-s(?P<size>\d+)--(?P<sha>[0-9a-f]{64})\.nii\.gz$")
SITE_NAMES = {
    "BA": "San Paolo Hospital, Bari",
    "BO": "DIBINEM Bologna",
    "FO": "Policlinico Riuniti Foggia",
    "JS": "Hospital Joan XXIII Tarragona",
    "KA": "Staedtisches Klinikum Karlsruhe",
    "ME": "University Hospital G. Martino Messina",
    "NSK": "Federal Neurosurgery Center Novosibirsk",
    "TO": "University of Turin",
}


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch(url: str, data: bytes | None = None, content_type: bool = False) -> bytes:
    headers = {"User-Agent": "AFlow-source-audit/1.0"}
    if content_type:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=180) as response:
        return response.read()


def graphql_tree() -> dict:
    query = (
        'query { snapshot(datasetId: "ds007045", tag: "1.0.1") '
        "{ id files(recursive: true) { id filename size directory annexed urls } } }"
    )
    payload = json.dumps({"query": query}).encode()
    parsed = json.loads(fetch(GRAPHQL, payload, content_type=True))
    if parsed.get("errors"):
        raise RuntimeError(parsed["errors"])
    return parsed["data"]["snapshot"]


def annex_record(row: dict) -> dict:
    match = ANNEX_RE.fullmatch(row["id"])
    if not match:
        raise ValueError(f"not SHA256 annex object: {row['filename']} {row['id']}")
    size = int(match.group("size"))
    if size != row["size"]:
        raise ValueError(f"annex size mismatch: {row['filename']}")
    urls = row.get("urls", [])
    if len(urls) != 1 or "versionId=" not in urls[0]:
        raise ValueError(f"not uniquely versioned: {row['filename']}")
    return {
        "path": row["filename"],
        "bytes": size,
        "sha256": match.group("sha"),
        "url": urls[0],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    snapshot = graphql_tree()
    if snapshot["id"] != f"{DATASET}:{TAG}":
        raise ValueError(snapshot["id"])
    files = {row["filename"]: row for row in snapshot["files"] if not row["directory"]}
    metadata = {}
    for name in ("README.txt", "dataset_description.json", "participants.tsv", "CHANGES"):
        row = files[name]
        if len(row.get("urls", [])) != 1 or "versionId=" not in row["urls"][0]:
            raise ValueError(f"metadata URL not versioned: {name}")
        body = fetch(row["urls"][0])
        if len(body) != row["size"]:
            raise ValueError(f"metadata size mismatch: {name}")
        metadata[name] = {
            "bytes": len(body),
            "git_object_id": row["id"],
            "sha256": sha_bytes(body),
            "url": row["urls"][0],
        }
        if name == "participants.tsv":
            participants_body = body
        elif name == "dataset_description.json":
            description = json.loads(body)
        elif name == "README.txt":
            readme = body.decode()
        elif name == "CHANGES":
            changes = body.decode()

    participants = list(csv.DictReader(io.StringIO(participants_body.decode()), delimiter="\t"))
    ids = [row["participant_id"] for row in participants]
    if len(ids) != len(set(ids)) or len(ids) != 337:
        raise ValueError(f"participant IDs: {len(ids)} total, {len(set(ids))} unique")

    selected = []
    exclusions = []
    for subject in sorted(ids):
        image_path = (
            f"derivatives/processing/{subject}/{subject}_t1ce_MNI152_brain.nii.gz"
        )
        label_paths = [
            f"derivatives/segmentation/{subject}/{subject}_MNI152_tumor-mask.nii.gz",
            f"derivatives/segmentation/{subject}/{subject}_MNI-152_tumor-mask.nii.gz",
        ]
        present_labels = [path for path in label_paths if path in files]
        if image_path not in files or len(present_labels) != 1:
            exclusions.append(
                {
                    "subject_id": subject,
                    "reason": "missing_expected_processing_t1ce"
                    if image_path not in files
                    else "missing_or_ambiguous_segmentation",
                    "image_present": image_path in files,
                    "label_paths_present": present_labels,
                }
            )
            continue
        label_path = present_labels[0]
        selected.append(
            {
                "subject_id": subject,
                "site_code": re.match(r"sub-([A-Z]+)", subject).group(1),
                "image": annex_record(files[image_path]),
                "label": annex_record(files[label_path]),
            }
        )
    expected_excluded = {"sub-BO97", "sub-BO199", "sub-NSK07"}
    if {row["subject_id"] for row in exclusions} != expected_excluded or len(selected) != 334:
        raise ValueError(f"unexpected pair inventory: {exclusions}")

    site_counts = collections.Counter(row["site_code"] for row in selected)
    if set(site_counts) != set(SITE_NAMES):
        raise ValueError(site_counts)
    selected_canonical = json.dumps(selected, sort_keys=True, separators=(",", ":")).encode()
    total_image = sum(row["image"]["bytes"] for row in selected)
    total_label = sum(row["label"]["bytes"] for row in selected)
    result = {
        "schema": "aflow.openneuro_ds007045.source_inventory",
        "schema_version": 1,
        "status": "passed_metadata_only_payload_not_accessed_model_not_accessed",
        "dataset": DATASET,
        "snapshot": TAG,
        "snapshot_id": snapshot["id"],
        "doi": description["DatasetDOI"],
        "license": description["License"],
        "participant_count_published": len(ids),
        "participant_count_frozen_intersection": len(selected),
        "pre_payload_exclusions": exclusions,
        "site_counts": {
            code: {"name": SITE_NAMES[code], "subjects": site_counts[code]}
            for code in sorted(site_counts)
        },
        "selected": {
            "image_variant": "MNI152 registered and skull-stripped; no publisher N4 or z-score",
            "image_count": len(selected),
            "label_count": len(selected),
            "image_bytes": total_image,
            "label_bytes": total_label,
            "total_bytes": total_image + total_label,
            "manifest_sha256": sha_bytes(selected_canonical),
            "cases": selected,
        },
        "metadata": metadata,
        "source_claims": {
            "preoperative_primary_gbm": "README and primary Scientific Data descriptor",
            "labels": [0, 1, 2, 3],
            "label_meaning": {
                "0": "background",
                "1": "necrotic core",
                "2": "non-enhancing/edematous tumor",
                "3": "contrast-enhancing tumor",
            },
            "expert_process": "SegResNet initialization; all reviewed by experienced neuroradiologists; 87 minor and 11 extensive manual corrections",
            "readme_has_all_eight_sites": all(name in readme for name in SITE_NAMES.values()),
        },
        "version_checks": {
            "changes_mentions_1_0_1": "1.0.1" in changes,
            "all_selected_urls_have_version_id": all(
                "versionId=" in row[k]["url"] for row in selected for k in ("image", "label")
            ),
            "all_selected_annex_ids_sha256": True,
        },
        "access": {
            "metadata_bytes_downloaded": sum(x["bytes"] for x in metadata.values()),
            "image_payload_downloaded": False,
            "label_payload_downloaded": False,
            "checkpoint_loaded": False,
            "model_forwards": 0,
            "performance_accessed": False,
        },
        "next_gate": "independent institution-overlap audit, then first-case image/label checksum-label-geometry-crop probe",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "subjects": len(selected),
                "sites": dict(sorted(site_counts.items())),
                "image_bytes": total_image,
                "label_bytes": total_label,
                "manifest_sha256": result["selected"]["manifest_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
