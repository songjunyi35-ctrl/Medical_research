"""Validate archived UCSD selected identities against TCIA's current official table."""
from pathlib import Path
from zipfile import ZipFile
import xml.etree.ElementTree as ET
import hashlib
import json
import re

ROOT=Path(__file__).resolve().parents[1]
RECORD=ROOT/"archive/reviews/2026-09-05_30h_chain_audit"


def main():
    path=RECORD/"source_metadata/UCSD_PTGBM-clinical-information_v3_2026-12-Mar.xlsx"
    ns={"m":"http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with ZipFile(path) as z:
        shared=["".join(x.itertext()) for x in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si",ns)]
        source_rows=ET.fromstring(z.read("xl/worksheets/sheet2.xml")).findall("m:sheetData/m:row",ns)
        pairs=[]
        for row in source_rows[1:]:
            values={}
            for cell in row.findall("m:c",ns):
                col=re.sub(r"\d","",cell.get("r",""))
                if col not in ("A","B"):continue
                value=cell.find("m:v",ns)
                text=value.text if value is not None else ""
                values[col]=shared[int(text)] if cell.get("t")=="s" and text else text
            if values.get("A"):pairs.append((values["A"],values.get("B","")))
    official=dict(pairs)
    assert len(pairs)==len(official)==243
    inventory_path=ROOT/"archive/research_records/2026-09-02_ucsd_ptgbm_external_confirmation/OFFICIAL_ASSET_INVENTORY.json"
    inventory=json.loads(inventory_path.read_text())
    differences=[]
    for row in inventory["selection"]["selected"]:
        match=re.fullmatch(r"sub-(\d{4})_ses-(\d{2})",row["ucsd_case_id"])
        assert match
        official_id="UCSD-PTGBM-"+match[1]+"_"+match[2]
        if official.get(official_id)!=row["brats_case_id"]:
            differences.append({"official_id":official_id,"official_brats_id":official.get(official_id),"archived_brats_id":row["brats_case_id"]})
    exposed=set()
    for path2 in (ROOT/"data/datasets/brats2024_random_subject_80_20_seed20260722_v1/train_cases.jsonl",
                  ROOT/"data/datasets/brats2024_random_subject_80_20_seed20260722_v1/sealed_test/test_cases.jsonl"):
        exposed.update(json.loads(line)["subject_id"] for line in path2.read_text().splitlines())
    exposed_local_subjects={identifier.split("_")[0] for identifier,brats in pairs if "-".join(brats.split("-")[:3]) in exposed}
    selected_local={"UCSD-PTGBM-"+row["ucsd_subject_id"].split("-")[1] for row in inventory["selection"]["selected"]}
    overlap=sorted(selected_local&exposed_local_subjects)
    result={"status":"passed" if not differences and not overlap else "failed",
        "official_rows":243,"selected_scans":75,"selected_subjects":len(selected_local),
        "selected_mapping_mismatches":differences,"official_patient_overlap":overlap,
        "official_clinical_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
        "archived_inventory_sha256":hashlib.sha256(inventory_path.read_bytes()).hexdigest(),
        "new_image_label_or_model_access":False,
        "interpretation":"official TCIA mapping directly verifies selected patient identifiers; mirror image bytes remain a separate provenance question"}
    with (RECORD/"UCSD_OFFICIAL_CLINICAL_MAPPING_AUDIT.json").open("x") as f:json.dump(result,f,indent=2,sort_keys=True);f.write("\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
