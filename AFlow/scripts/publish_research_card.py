"""Publish an authored card as a backed-up deterministic canonical status block."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def publish(record):
    rec=(ROOT/record).resolve()
    assert rec.is_relative_to(ROOT/'archive/research_records')
    card=(rec/'CARD.md').read_text();tag=rec.name
    begin=f'<!-- study-card:{tag}:start -->';end=f'<!-- study-card:{tag}:end -->'
    block=begin+'\n'+card.rstrip()+'\n'+end+'\n'
    changes={}
    for name in ['AFLOW_BRATS_HANDOFF_CURRENT.md','AFLOW_BRATS_METHOD_CATALOG.md']:
        path=ROOT/name;before=path.read_bytes();old=before.decode()
        backup=rec/(name+'.BEFORE_CARD')
        if not backup.exists():
            with backup.open('xb') as f:f.write(before)
        if begin in old:
            start=old.index(begin);stop=old.index(end,start)+len(end)
            after=old[:start]+block.rstrip('\n')+old[stop:]
        else:after=block+'\n'+old
        assert path.read_bytes()==before,'concurrent canonical edit'
        path.write_text(after)
        changes[name]={'before_sha256':hashlib.sha256(before).hexdigest(),'after_sha256':hashlib.sha256(after.encode()).hexdigest()}
    now=datetime.now(timezone.utc)
    receipt={'created_utc':now.isoformat(),'card_sha256':hashlib.sha256((rec/'CARD.md').read_bytes()).hexdigest(),'changes':changes,'publisher_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    with (rec/f'CARD_PUBLICATION_{now.strftime("%H%M%S%f")}.json').open('x') as f:json.dump(receipt,f,indent=2)
    print('Published '+tag)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('record');publish(ap.parse_args().record)
