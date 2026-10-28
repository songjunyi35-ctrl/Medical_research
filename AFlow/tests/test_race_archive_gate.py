"""An archive must not publish a partially written or unaudited metric result."""
import hashlib
import json
from scripts.race_seg_20261028 import archive_watch


def test_partial_result_is_not_publishable(tmp_path,monkeypatch):
    monkeypatch.setattr(archive_watch,'R',tmp_path)
    (tmp_path/'RESULT.json').write_text('{"state":"complete","clear_benefit_gate":true}')
    assert archive_watch.audited_result() is None
    (tmp_path/'RESULT.md').write_text('Draft outcome')
    (tmp_path/'COMPLETION_AUDIT.json').write_text('{"state":"complete","result_sha256":"stale"}')
    assert archive_watch.audited_result() is None


def test_only_matching_completion_audit_accepts_result(tmp_path,monkeypatch):
    monkeypatch.setattr(archive_watch,'R',tmp_path)
    result=tmp_path/'RESULT.json';result.write_text('{"state":"incomplete","clear_benefit_gate":false}')
    (tmp_path/'RESULT.md').write_text('Incomplete experiment, no gain claim')
    audit={'state':'incomplete','result_sha256':hashlib.sha256(result.read_bytes()).hexdigest()}
    (tmp_path/'COMPLETION_AUDIT.json').write_text(json.dumps(audit))
    assert archive_watch.audited_result()['clear_benefit_gate'] is False
    result.write_text('{"state":"complete","clear_benefit_gate":true}')
    assert archive_watch.audited_result() is None
