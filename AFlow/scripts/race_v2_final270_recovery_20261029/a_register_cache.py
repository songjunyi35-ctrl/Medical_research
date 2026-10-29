"""Freeze CPU data reconstruction separately so it can start without GPU wait."""
from scripts.race_v2_final270_recovery_20261029.a_common import *

def main():
    R.mkdir(parents=True,exist_ok=True);assert not (R/'EXECUTION_FREEZE.json').exists()
    sources={}
    for name in ('PREPARATION_FREEZE_V3.json','EXECUTION_FREEZE_V2.json'):
        for p,d in read(OLD/name)['source_sha256'].items():
            assert sha(ROOT/p)==d,p;sources[p]=d
    for name in ('__init__.py','a_common.py','a_cache.py','a_register_cache.py'):
        path=ROOT/'scripts/race_v2_final270_recovery_20261029'/name;sources[str(path.relative_to(ROOT))]=sha(path)
    write(R/'EXECUTION_FREEZE.json',dict(at=now(),source_sha256=sources,selection_sha256=SELECTION_SHA,
        cache_only=True,patient_model_forwards=0,worker=1154259,processes=23,
        four_payload_sha256=read(OLD/'CACHE_READY.json')['payload_sha256'],
        historical_native_index_sha256=read(OLD/'CACHE_READY.json')['index_sha256'],
        canonical_geometry_index_sha256=sha(CANON_INDEX),manifest_sha256=sha(MANIFEST),
        index_contract='new deterministic JSON SHA recorded honestly; compare canonical crop geometry, four payload hashes, and original forward SHA',
        original_selection_immutable=True,new_independent_confirmation=False))

if __name__=='__main__':main()
