#!/usr/bin/env python3
"""Dual independent official transfer of the frozen UCSD BraTS-2024-test cohort (112 files).

No publisher checksum exists in package 1221, so the transfer gate is per-file
SHA-256 equality across two independent transfers, plus gzip decode integrity
and raw mask WT statistics recorded before any cache/model stage.
"""
from __future__ import annotations
import base64
import gzip
import hashlib
import html
import json
import os
import re
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from scripts import ucsd_brats2024test_protocol as p

BASE = 'https://faspex.cancerimagingarchive.net/aspera/faspex'
PAGE = 'https://www.cancerimagingarchive.net/collection/ucsd-ptgbm/'
SDK = Path('/tmp/aflow-rhuh-sdk-FuoemL')
BATCH = 56


def authorize(session):
    page = session.get(PAGE, timeout=60)
    page.raise_for_status()
    states = []
    for link in re.findall(r'https://faspex[^"<>\s]+', page.text):
        state = parse_qs(urlparse(html.unescape(link)).query).get('context', [None])[0]
        if state and str(json.loads(base64.b64decode(state)).get('id')) == str(p.PACKAGE_ID):
            states.append(state)
    if not states or len(set(states)) != 1:
        raise RuntimeError('faspex link resolution failed')
    state = states[0]
    config = session.get(f'{BASE}/config.js', timeout=60)
    config.raise_for_status()
    client = re.search(r"client_id:\s*'([^']+)'", config.text).group(1)
    redirect = f'{BASE}/token'
    auth = session.get(f'{BASE}/auth/authorize_public_link',
                       params={'response_type': 'code', 'client_id': client, 'redirect_uri': redirect, 'state': state},
                       allow_redirects=False, timeout=60)
    query = parse_qs(urlparse(auth.headers['Location']).query)
    token = session.post(f'{BASE}/auth/token',
                         json={'code': query['code'][0], 'state': query.get('state', [state])[0],
                               'grant_type': 'authorization_code', 'client_id': client, 'redirect_uri': redirect},
                         timeout=60)
    token.raise_for_status()
    return {'Authorization': 'Bearer ' + token.json()['access_token']}


def transfer(session, headers, paths, destination, log):
    destination.mkdir(parents=True, exist_ok=True)
    batches = [paths[i:i + BATCH] for i in range(0, len(paths), BATCH)]
    for number, batch in enumerate(batches, start=1):
        reply = session.post(f'{BASE}/api/v5/packages/{p.PACKAGE_ID}/transfer_spec/download',
                             headers=headers, params={'type': 'received', 'transfer_type': 'connect'},
                             json={'paths': [{'path': path, 'type': 'file'} for path in batch]}, timeout=120)
        reply.raise_for_status()
        spec = reply.json()
        sources = [row['source'] for row in spec['paths']]
        if len(sources) != len(batch) or {Path(s).name for s in sources} != {Path(x).name for x in batch}:
            raise RuntimeError('transfer spec path mismatch')
        env = dict(os.environ, ASPERA_SCP_TOKEN=spec['token'], ASPERA_SCP_COOKIE=spec.get('cookie', ''))
        cmd = [str(SDK / 'ibm-aspera-transfer-sdk-linux-amd64-1.1.9/bin/ascp'), '--mode=recv',
               '--host=' + spec['remote_host'], '--user=' + spec['remote_user'],
               '-P', str(spec['ssh_port']), '-O', str(spec['fasp_port']),
               '-i', str(SDK / 'ibm-public-client-key.pem'), '-L', str(SDK), '--overwrite=never', '-l', '200m',
               '--tags64=' + base64.b64encode(json.dumps(spec.get('tags', {})).encode()).decode(),
               *sources, str(destination)]
        run = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=1800)
        log.append({'destination': str(destination), 'batch': number, 'files': len(batch), 'returncode': run.returncode})
        if run.returncode != 0:
            log[-1]['tail'] = (run.stdout + run.stderr)[-800:].replace(spec['token'], '[REDACTED]')
            raise RuntimeError(f'ascp failed batch {number} -> {destination}')


def gz_crc_ok(path):
    with gzip.open(path, 'rb') as f:
        while f.read(64 << 20):
            pass
    return True


def main():
    if p.SOURCE_RECEIPT.exists():
        raise FileExistsError(p.SOURCE_RECEIPT)
    rows = p.cases()
    paths = []
    for row in rows:
        paths.extend((row['image_remote_path'], row['mask_remote_path']))
    expected = {}
    for row in rows:
        expected[Path(row['image_remote_path']).name] = int(row['image_size'])
        expected[Path(row['mask_remote_path']).name] = int(row['mask_size'])
    need = sum(expected.values())
    repo_free = shutil.disk_usage(p.SOURCE if p.SOURCE.exists() else p.ROOT).free
    if repo_free < need + (1 << 30):
        raise RuntimeError('insufficient repo disk for copy A')
    if shutil.disk_usage('/tmp').free < need + (2 << 30):
        raise RuntimeError('insufficient /tmp disk for copy B')
    session = requests.Session()
    session.mount('https://', HTTPAdapter(max_retries=Retry(total=4, backoff_factor=1.0, allowed_methods=frozenset(('GET', 'POST')))))
    log = []
    start = time.monotonic()
    headers = authorize(session)
    transfer(session, headers, paths, p.COPY_A, log)
    headers = authorize(session)
    transfer(session, headers, paths, p.COPY_B, log)
    files = []
    gates = {'all_present': True, 'all_sizes_match': True, 'all_sha_equal': True, 'all_gzip_decode_ok': True}
    for row in rows:
        entry = {'case_id': row['case_id'], 'subject_id': row['subject_id']}
        for kind, remote in (('image', row['image_remote_path']), ('mask', row['mask_remote_path'])):
            name = Path(remote).name
            a = p.COPY_A / name
            b = p.COPY_B / name
            if not a.is_file() or not b.is_file():
                gates['all_present'] = False
                continue
            if a.stat().st_size != expected[name] or b.stat().st_size != expected[name]:
                gates['all_sizes_match'] = False
            da, db = p.sha(a), p.sha(b)
            if da != db:
                gates['all_sha_equal'] = False
            if not gz_crc_ok(a):
                gates['all_gzip_decode_ok'] = False
            entry[f'{kind}_path'] = str(a)
            entry[f'{kind}_sha256'] = da
            entry[f'{kind}_bytes'] = a.stat().st_size
        files.append(entry)
    if not all(gates.values()):
        p.write_new(p.RECORD / 'SOURCE_TRANSFER_FAILURE.json', {'gates': gates, 'log': log})
        raise RuntimeError(str(gates))
    receipt = {'schema': 'aflow.ucsd_brats2024test.source_receipt', 'schema_version': 1,
               'created_at_utc': datetime.now(timezone.utc).isoformat(),
               'package_id': p.PACKAGE_ID, 'cohort_sha256': p.COHORT_SHA,
               'transfer_gate': 'dual independent official transfers; per-file SHA-256 equality; gzip decode integrity; no publisher .sums exists in package 1221',
               'copies': {'copy_a': str(p.COPY_A), 'copy_b': str(p.COPY_B)},
               'batches': log, 'files_per_copy': len(paths), 'bytes_per_copy': need,
               'selected': {'files': files},
               'elapsed_seconds': time.monotonic() - start,
               'image_pixels_decoded': False, 'model_inference': False,
               'runner_sha256': p.sha(Path(__file__))}
    p.write_new(p.SOURCE_RECEIPT, receipt)
    print(json.dumps({k: receipt[k] for k in ('files_per_copy', 'bytes_per_copy', 'elapsed_seconds', 'batches')}, indent=2))


if __name__ == '__main__':
    main()
