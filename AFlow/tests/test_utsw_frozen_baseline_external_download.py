from scripts import utsw_frozen_baseline_external_protocol as p
from scripts.download_utsw_no_n4_paired_external import expected_paths


def test_shared_transfer_path_contract_matches_baseline():
    value=expected_paths(p.cohort())
    assert len(value)==724 and sum(value.values())==p.SOURCE_BYTES
    assert len({x.split("/",1)[0] for x in value})==362
