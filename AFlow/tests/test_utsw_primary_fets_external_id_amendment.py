import pytest

from sammed3d.t1ce_binary import T1ceBinaryDataError
from scripts.run_utsw_primary_fets_no_n4_input_bridge_amendment_01 import external_case_id


def test_only_frozen_utsw_ids_are_added():
    assert external_case_id("BT0001") == "BT0001"
    assert external_case_id("BT0002") == "BT0002"
    assert external_case_id("BT0003") == "BT0003"
    with pytest.raises(T1ceBinaryDataError):
        external_case_id("BT0004")


def test_original_brats_id_semantics_remain():
    assert external_case_id("BraTS-GLI-00005-100") == "BraTS-GLI-00005-100"
