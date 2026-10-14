"""Download only TCIA's small UPENN metadata and report identity/timepoint columns."""
from pathlib import Path
from datetime import datetime,timezone
import csv
import hashlib
import io
import json
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"archive/reviews/2026-09-05_upenn_source_audit"
NAMES=("UPENN-GBM_clinical_info_v2.1.csv","radiology_mapping.csv","UPENN-GBM_acquisition.csv","UPENN-GBM_data_availability.csv")


def main():
    s=requests.Session();s.mount("https://",HTTPAdapter(max_retries=Retry(total=3,backoff_factor=.5)))
    folder=OUT/"source_metadata";folder.mkdir(exist_ok=False)
    report={"created_utc":datetime.now(timezone.utc).isoformat(),"image_or_label_download":False,"model_inference":False,"files":[]}
    for name in NAMES:
        url="https://www.cancerimagingarchive.net/wp-content/uploads/"+name
        response=s.get(url,timeout=30);response.raise_for_status()
        if len(response.content)>5_000_000:raise RuntimeError("unexpected metadata size")
        path=folder/name
        with path.open("xb") as f:f.write(response.content)
        text=response.content.decode("utf-8-sig")
        reader=csv.DictReader(io.StringIO(text));rows=list(reader)
        columns=reader.fieldnames
        identity_columns=[x for x in columns if any(t in x.lower() for t in ("id","time","scan","segment","brats","subject","patient"))]
        # Only print names/availability fields and a few identifiers, not clinical outcomes.
        item={"name":name,"url":url,"bytes":len(response.content),"sha256":hashlib.sha256(response.content).hexdigest(),
              "rows":len(rows),"columns":columns,"identity_columns":identity_columns,
              "first_identifiers":[{k:row[k] for k in identity_columns if k in row} for row in rows[:2]]}
        report["files"].append(item)
    with (OUT/"METADATA_RECEIPT.json").open("x") as f:json.dump(report,f,indent=2,sort_keys=True);f.write("\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()
