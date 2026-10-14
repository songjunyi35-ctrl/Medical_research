"""Frozen shared contract for the no-N4 clean-teacher experiment."""
from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]
RECORD=ROOT/"archive/research_records/2026-09-05_no_n4_clean_teacher_replication"
SPLITS=ROOT/"data/datasets/brats2024_t1ce_no_n4_clean_20260905"
WORK=ROOT/"workspace/BraTS/sammed3d_t1ce_wt/no_n4_clean_20260905"
SOURCE=ROOT/"data/datasets/brats2024_t1ce_va_hdfs_20260812_v1/train_t1ce_wt.jsonl"
CONTROL_CACHE=ROOT/"workspace/BraTS/cache/t1ce_wt_n4_braincrop_128_20260817/train"
NATIVE=ROOT/"archive/research_records/2026-08-21_voxel_kd_data972_native_sampling/NATIVE_WT_VOLUMES.json"
SOURCE_SHA="6728d8d5e6846f90f6e5d0ad47acc4f2706eed42763ace4e228abcbbd6a4cf65"
INDEX_SHA="7373cb306a01ec1ec311a584a088844439de84ff7289ed819127de6270839cfd"
LABELS_SHA="bf470d48a5f115f09407ad351bb6dee11ee36bdd0c2a4285ae0bd9cc22125724"
NATIVE_SHA="32be09856d71e78f512792c8bcf8f5f9ee137e6ba1e4358862ff2a61ba828c31"
IDEA_SHA="6ffe103b4c592850eac4f4f214c157bf603e97a8e5a436fec926401cb7a41921"
PREREG_SHA="1dbd9376d682f2893c0a99c5656995cbb9fe02ba6a5fca239251954e4fc7a519"
SPLIT_SHA="68975292cc10ab581657a97789050a28cc03d4a3bc3b0991682b7284bfa9c3f2"
AMENDMENT_SHA="9bbf43f991388ec2a9dea7aa1070ca73c783a25f8ae7cd416733fadbe7a72049"
SEEDS={"e":{"split":20261301,"history":20261311,"ema":20261312,"student":20261313,"bootstrap":20261331},
       "f":{"split":20261302,"history":20261321,"ema":20261322,"student":20261323,"bootstrap":20261332}}
EPOCHS=168
STEPS=121
DRAWS=972
PARAMETERS=4_700_897


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(8<<20),b""):h.update(b)
    return h.hexdigest()


def checked(path,expected):
    if sha(path)!=expected:raise ValueError(f"frozen hash mismatch: {path}")


def docs():
    checked(RECORD/"IDEA.md",IDEA_SHA);checked(RECORD/"PREREG.md",PREREG_SHA)
    checked(RECORD/"AMENDMENT_01.md",AMENDMENT_SHA)


def write_new(path,data):
    with Path(path).open("x") as f:json.dump(data,f,indent=2,sort_keys=True);f.write("\n")


def read_rows(path):return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def own_fit(record,fit_ids,inner_ids):
    fit={r["case_id"] for r in fit_ids};inner={r["case_id"] for r in inner_ids}
    fs={r["subject_id"] for r in fit_ids};ins={r["subject_id"] for r in inner_ids}
    if fit&inner or fs&ins or len(fit)!=record["fit"]["cases"] or len(inner)!=record["inner"]["cases"]:
        raise ValueError("fit/inner ownership violation")
    return True
