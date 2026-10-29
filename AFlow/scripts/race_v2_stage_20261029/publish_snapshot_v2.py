"""Publish this stage's code/text through the established research checkout.

Does not stage the AFlow worktree or upload weights, images or prediction arrays.
Run explicitly after reviewing a coherent preparation/startup/result snapshot.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from .governance import ROOT, RECORD, atomic_json, sha

RELEASE = Path('/tmp/medical_research_publish_20261014')
TARGET = 'https://github.com/songjunyi35-ctrl/Medical_research.git'
STAGES = ['2026-10-29_race_v2_stage', '2026-10-29_race_v2_training',
          '2026-10-29_race_v2_inference_a', '2026-10-29_race_v2_inference_b']
MODULES = ['race_v2_stage_20261029', 'race_v2_20261029',
           'race_v2_inference_a_20261029', 'race_v2_inference_b_20261029']
DOCS = ['AGENTS.md', 'AFLOW_BRATS_HANDOFF_CURRENT.md', 'AFLOW_BRATS_METHOD_CATALOG.md',
        'AFLOW_BRATS_RESULTS.md', 'AFLOW_BRATS_LEADERBOARD.md', 'AFLOW_BRATS_LEADERBOARD.json']


def git(*arguments, env=None):
    return subprocess.check_output(['git', *arguments], cwd=RELEASE, env=env, text=True).strip()


def main(message):
    assert git('remote', 'get-url', 'origin') == TARGET
    paths = {ROOT / name for name in DOCS}
    paths.update(path for module in MODULES for path in (ROOT / 'scripts' / module).rglob('*.py'))
    paths.update(path for stage in STAGES for path in (ROOT / 'archive/research_records' / stage).rglob('*')
                 if path.is_file() and path.suffix in ('.md', '.json', '.jsonl')
                 and 'docs_before_stage' not in path.parts and not path.name.startswith('PUBLICATION_RECEIPT'))
    allowed = {'AFlow/' + source.relative_to(ROOT).as_posix() for source in paths} | {'MANIFEST.sha256.tsv'}
    for line in git('status', '--porcelain').splitlines():
        assert len(line) >= 4 and line[3:] in allowed, 'publication checkout has unrelated changes'
    git('fetch', 'origin', 'main')
    git('merge', '--ff-only', 'FETCH_HEAD')
    copied = []
    for source in sorted(paths):
        assert not source.is_symlink(), source
        assert source.suffix in ('.py', '.md', '.json', '.jsonl'), source
        data = source.read_bytes()
        assert len(data) < 8 * 2**20, source
        destination = RELEASE / 'AFlow' / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        copied.append(destination)
    manifest = RELEASE / 'MANIFEST.sha256.tsv'
    lines = manifest.read_text().splitlines()
    order = [line.split('\t', 2)[2] for line in lines]
    entries = {line.split('\t', 2)[2]: line for line in lines}
    for path in copied:
        relative = path.relative_to(RELEASE).as_posix()
        entries[relative] = f'{sha(path)}\t{path.stat().st_size}\t{relative}'
    manifest.write_text('\n'.join([entries[name] for name in order] +
                                  [entries[name] for name in sorted(entries.keys() - set(order))]) + '\n')
    subprocess.run(['python3', 'verify_migration_bundle.py'], cwd=RELEASE, check=True)
    git('add', '--', *[path.relative_to(RELEASE).as_posix() for path in copied], 'MANIFEST.sha256.tsv')
    if git('diff', '--cached', '--name-only'):
        git('commit', '-m', message)
    environment = os.environ.copy()
    environment['GIT_SSH_COMMAND'] = ('ssh -i /tmp/medical_research_archive_20261028_deploy_key '
                                    '-o IdentitiesOnly=yes -o BatchMode=yes -o StrictHostKeyChecking=yes '
                                    '-o UserKnownHostsFile=/tmp/medical_research_github_known_hosts')
    git('push', 'ssh://git@github.com/songjunyi35-ctrl/Medical_research.git', 'HEAD:main', env=environment)
    remote = git('ls-remote', 'origin', 'refs/heads/main').split()[0]
    head = git('rev-parse', 'HEAD')
    assert head == remote
    receipt = dict(at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   state='pushed_and_remote_head_verified', commit=head,
                   url=f'https://github.com/songjunyi35-ctrl/Medical_research/commit/{head}',
                   copied_files=len(copied), clinical_arrays_or_weights_uploaded=False,
                   source_worktree_or_remote_modified=False,
                   source_snapshot_sha256={str(path.relative_to(RELEASE)): sha(path) for path in copied})
    atomic_json(RECORD / f'PUBLICATION_RECEIPT_{head[:12]}.json', receipt)
    print(json.dumps({key: value for key, value in receipt.items() if key != 'source_snapshot_sha256'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--message', required=True)
    main(parser.parse_args().message)
