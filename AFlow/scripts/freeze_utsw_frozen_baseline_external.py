"""Freeze/verify the UTSW362 frozen-baseline external implementation."""
from datetime import datetime,timezone
import json
from scripts import utsw_frozen_baseline_external_protocol as p

FILES=("scripts/utsw_frozen_baseline_external_protocol.py","scripts/prepare_utsw_frozen_baseline_external.py",
"scripts/download_utsw_frozen_baseline_external.py","scripts/build_utsw_frozen_baseline_external_cache.py",
"scripts/download_utsw_frozen_baseline_external_amendment_01.py",
"scripts/evaluate_utsw_frozen_baseline_external.py","scripts/smoke_utsw_frozen_baseline_external.py",
"scripts/freeze_utsw_frozen_baseline_external.py","scripts/download_utsw_no_n4_paired_external.py",
"scripts/utsw_no_n4_paired_external_protocol.py","scripts/build_no_n4_clean_cache.py","scripts/build_t1ce_wt_cache.py",
"scripts/audit_utsw_primary_fets_no_n4_input_bridge.py","scripts/train_t1ce_wt_gcsam_inner.py","sammed3d/t1ce_binary.py",
"tests/test_utsw_frozen_baseline_external_protocol.py","tests/test_utsw_frozen_baseline_external_download.py",
"tests/test_utsw_frozen_baseline_external_cache.py","tests/test_utsw_frozen_baseline_external_evaluation.py",
"tests/test_utsw_frozen_baseline_external_download_amendment.py")
RECEIPT=p.RECORD/"IMPLEMENTATION_AMENDMENT_01.json"
def current():return {x:p.sha(p.ROOT/x) for x in FILES}
def check_implementation():
    p.docs();v=json.loads(RECEIPT.read_text())
    if v.get("files")!=current():raise ValueError("baseline implementation hash mismatch")
    return v
def main():
    p.docs();v={"schema":"aflow.utsw_frozen_baseline_external.implementation","created_utc":datetime.now(timezone.utc).isoformat(),
    "status":"frozen_before_preaccess_source_or_gpu","files":current(),"cohort":{"cases":362,"subjects":362,"source_files":724,"bytes_per_copy":p.SOURCE_BYTES},
    "models":{k:{"file_sha256":x["file_sha256"],"state_sha256":x["state_sha256"],"epoch":x["epoch"],"primary_view":x["primary_view"]} for k,x in p.MODELS.items()},
    "views":["identity","flip8"],"forwards":6516,"bootstrap":{"samples":p.BOOTSTRAP_SAMPLES,"seed":p.BOOTSTRAP_SEED},"old_cohort_access":False}
    p.write_new(RECEIPT,v);check_implementation();print(json.dumps(v,indent=2,sort_keys=True))
if __name__=="__main__":main()
