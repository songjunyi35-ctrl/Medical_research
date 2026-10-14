"""Write the DICOM SEG probe preaccess receipt before new series download."""
from datetime import datetime, timezone
import json
from pathlib import Path

from scripts import remind_dicomseg_probe_protocol as p


def main() -> None:
    p.docs()
    value = {
        "schema": "aflow.remind_dicomseg_source_probe.preaccess",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_before_dicomseg_download_or_array_decode",
        "case_id": "ReMIND-002",
        "assets": p.ASSETS,
        "dependency": {"version": "3.0.1", "wheel_sha256": p.WHEEL_SHA,
                       "extracted_manifest_sha256": p.DEPENDENCY_MANIFEST_SHA, "installed": False},
        "prior_access": {"old_probe_dicom_downloaded_not_decoded": True,
                         "old_probe_nrrd_downloaded_not_decoded": True,
                         "old_probe_result_sha256": p.OLD_RESULT_SHA,
                         "old_probe_download_sha256": p.OLD_DOWNLOAD_SHA},
        "authorized": "one new T1ce download and two independent downloads for each of tumor/cerebrum DICOM SEG; no other ReMIND series",
        "access": {"new_dicomseg": False, "dicom_arrays": False, "segmentation_arrays": False,
                   "other_candidate_pixels": False, "checkpoint": False, "model_forwards": False, "metrics": False},
        "idea_sha256": p.IDEA_SHA,
        "prereg_sha256": p.PREREG_SHA,
        "amendment_sha256": p.AMENDMENT_SHA,
        "runner_sha256": p.sha(Path(__file__)),
    }
    p.write_new(p.RECORD / "PROBE_PREACCESS.json", value)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
