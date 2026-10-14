from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import remind_dicomseg_probe_protocol as protocol
from scripts.audit_remind_dicomseg_probe import decode_seg, source_sort_key


def frame(sop, z):
    source = SimpleNamespace(ReferencedSOPInstanceUID=sop)
    derivation = SimpleNamespace(SourceImageSequence=[source])
    segment = SimpleNamespace(ReferencedSegmentNumber=1)
    position = SimpleNamespace(ImagePositionPatient=[0.0, 0.0, float(z)])
    return SimpleNamespace(DerivationImageSequence=[derivation],
                           SegmentIdentificationSequence=[segment], PlanePositionSequence=[position])


def test_dependency_manifest_and_wheel_are_exact():
    assert protocol.sha(protocol.WHEEL) == protocol.WHEEL_SHA
    assert protocol.dependency_manifest_sha() == protocol.DEPENDENCY_MANIFEST_SHA


def test_source_sort_uses_orientation_normal():
    a = SimpleNamespace(ImageOrientationPatient=[1, 0, 0, 0, 1, 0], ImagePositionPatient=[0, 0, 2])
    b = SimpleNamespace(ImageOrientationPatient=[1, 0, 0, 0, 1, 0], ImagePositionPatient=[0, 0, 5])
    assert source_sort_key(a) == 2 and source_sort_key(b) == 5


def test_seg_frames_map_by_sop_not_frame_order():
    dataset = SimpleNamespace(
        Modality="SEG", SegmentationType="BINARY", BitsAllocated=1,
        SegmentSequence=[SimpleNamespace(SegmentLabel="tumor")],
        pixel_array=np.asarray([[[1, 0], [0, 0]], [[0, 1], [0, 0]]], dtype=np.uint8),
        PerFrameFunctionalGroupsSequence=[frame("sop2", 2), frame("sop0", 0)],
    )
    mapping = {"sop0": 0, "sop1": 1, "sop2": 2}
    positions = {f"sop{i}": np.asarray([0.0, 0.0, float(i)]) for i in range(3)}
    mask, metadata = decode_seg(dataset, mapping, positions, 2, 2)
    assert mask.shape == (3, 2, 2)
    assert mask[0, 0, 1] == 1 and mask[2, 0, 0] == 1 and not mask[1].any()
    assert metadata["referenced_source_sops"] == 2 and metadata["position_max_abs_mm"] == 0
    dataset.PerFrameFunctionalGroupsSequence[1] = frame("sop2", 2)
    with pytest.raises(ValueError):
        decode_seg(dataset, mapping, positions, 2, 2)


def test_download_contract_ignores_only_license_and_repeats_seg():
    source = (Path(__file__).parents[1] / "scripts/download_remind_dicomseg_probe.py").read_text()
    assert 'item.filename.lower().endswith(".dcm")' in source
    assert '[item.filename for item in other] != ["LICENSE"]' in source
    assert 'download_series("tumor", "a")' in source and 'download_series("tumor", "b")' in source
    assert 'download_series("cerebrum", "a")' in source and 'download_series("cerebrum", "b")' in source
