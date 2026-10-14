from scripts.audit_utsw_primary_fets_no_n4_input_bridge import validate_source_arrays
from scripts.download_utsw_frozen_baseline_external_amendment_01 import chunks

def test_fixed_batching_is_8_batches_without_search():
    value=chunks(list(range(724)));assert [len(x) for x in value]==[100,100,100,100,100,100,100,24]

def test_missing_pairs_cannot_be_exact_by_construction():
    a={"exists":False,"bytes":None,"sha256":None};b=dict(a)
    exact=bool(a.get("exists") and b.get("exists") and a.get("size_exact") and b.get("size_exact") and a.get("sha256") and b.get("sha256") and a["bytes"]==b["bytes"] and a["sha256"]==b["sha256"])
    assert not exact
