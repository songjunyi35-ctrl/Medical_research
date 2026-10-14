"""Mechanical link-context correction for this window's generated archive reports."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
ALLOWED={'letterbox_fidelity','development_error_anatomy','small_rotation','image_blur'}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--study',choices=sorted(ALLOWED),required=True);args=ap.parse_args()
    rec=ROOT/f'archive/research_records/2026-09-10_{args.study}';changes={}
    for name in ('RESULT.md','INTERPRETATION.md'):
        path=rec/name
        if not path.exists():continue
        old=path.read_text()
        new=re.sub(r'\]\((archive/|workspace/)',r'](../../../\1',old)
        if old==new:continue
        backup=rec/(name+'.BEFORE_LINK_CONTEXT_FIX')
        if not backup.exists():
            with backup.open('x') as f:f.write(old)
        path.write_text(new)
        changes[name]={'before_sha256':hashlib.sha256(old.encode()).hexdigest(),'after_sha256':hashlib.sha256(new.encode()).hexdigest()}
    if changes:
        receipt=rec/('LINK_CONTEXT_FIX_'+datetime.now(timezone.utc).strftime('%H%M%S%f')+'.json')
        with receipt.open('x') as f:json.dump({'changes':changes,'metrics_raw_json_and_prereg_unchanged':True,'card_root_fragments_intentionally_not_modified':True},f,indent=2)
    print(json.dumps(changes,indent=2))


if __name__=='__main__':main()
