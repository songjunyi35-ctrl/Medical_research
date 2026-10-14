"""Freeze source/model access for UTSW362 frozen baselines."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from scripts import utsw_frozen_baseline_external_protocol as p


def main() -> None:
    from scripts.freeze_utsw_frozen_baseline_external import check_implementation
    check_implementation()
    if not p.three_hour_gate(): raise RuntimeError("less than three hours remain")
    if shutil.disk_usage(p.SOURCE.parent).free < 12 * 1024**3: raise RuntimeError("less than12GiB /tmp free")
    rows=p.cohort();p.docs()
    models={name:{"path":str(row["path"]),"file_sha256":row["file_sha256"],"state_sha256":row["state_sha256"],
                       "schema":row["schema"],"epoch":row["epoch"],"primary_view":row["primary_view"]} for name,row in p.MODELS.items()}
    value={"schema":"aflow.utsw_frozen_baseline_external.preaccess","created_utc":datetime.now(timezone.utc).isoformat(),
        "status":"frozen_before_utsw_source_and_gpu_model_access","cohort":{"cases":362,"subjects":362,"files":724,"bytes_per_copy":p.SOURCE_BYTES},
        "models":models,"protocol":{"cache":"canonical positive-mask N4 primary FeTS","target":"UTSW label>0",
        "views":["identity","flip8"],"threshold":0.0,"bootstrap":{"samples":p.BOOTSTRAP_SAMPLES,"seed":p.BOOTSTRAP_SEED},
        "primary_absolute":{"full972_kd168":"flip8","gcsam184":"identity"},"absolute_gates":{"scan":0.70,"subject":0.70,"point_hits":326}},
        "time_gate":{"minimum_hours":3,"deadline_utc":p.DEADLINE.isoformat(),"passed":True},"disk_gate":{"minimum_gib":12,"passed":True},
        "training_source_caveat":"Kaggle mirror; official corrected-v2 byte parity unresolved",
        "prior_access":{"full972_kd168":["fixed_validation","reused_test270","mirror_uint8_ucsd75"],
                        "gcsam184":["train_inner_selection","reused_test270","mirror_uint8_ucsd75","later_replication_failed"]},
        "access":{"new_utsw_subjects":False,"model_checkpoint_loaded_on_gpu":False,"model_forwards":False,"external_metrics":False,
                  "fixed_validation":False,"outer":False,"confirmation":False,"test270":False,"ucsd75":False,"rhuh":False},
        "cohort_ids_sha256":p.source_contract.MANUAL_IDS_SHA,"runner_sha256":p.sha(Path(__file__))}
    p.write_new(p.PREACCESS,value);p.verify_preaccess();print(json.dumps(value,indent=2,sort_keys=True))


if __name__=="__main__":main()
