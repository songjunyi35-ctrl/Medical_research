"""Keep only current machine leaderboard data at repository root."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
ALLOWED_JSON={'AFLOW_BRATS_LEADERBOARD.json'}
LEGACY_FILES={'FIXED1_DEPLOYMENT_CANDIDATE_20260814.json','MCTS_DEPLOYMENT_CANDIDATE_20260813.json',
              'HANDOFF_CLEANUP_20260813.json','CODE_ARCHIVE_CANDIDATES_20260813.sha256'}


def validate_root_artifacts(root):
    root=Path(root)
    found={p.name for p in root.glob('*.json') if p.is_file()}
    if found!=ALLOWED_JSON:raise ValueError(f'root JSON differs from current-only policy: {sorted(found)}')
    if list(root.glob('*.sha256')):raise ValueError('historical SHA lists belong in archive')
    if any((root/name).exists() or (root/name).is_symlink() for name in LEGACY_FILES):
        raise ValueError('legacy artifact or alias remains at root')
    stub=root/'AFLOW_BRATS_HANDOFF_20260813_SINGLE_REPO.md'
    if not stub.is_file() or len(stub.read_text().splitlines())>20:
        raise ValueError('historical root handoff must be a short compatibility page')
    if 'archive/legacy_root_artifacts_20260914/' not in stub.read_text():
        raise ValueError('historical handoff does not point to archived original')
    return {'status':'passed','root_json_files':sorted(found),'legacy_root_artifacts':0,'historical_handoff_lines':len(stub.read_text().splitlines())}


if __name__=='__main__':print(json.dumps(validate_root_artifacts(ROOT),ensure_ascii=False,indent=2))
