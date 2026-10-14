from datetime import timedelta
import sys
import pytest
from scripts import finalize_resumed_session as m


def test_cannot_finalize_before_window_end(monkeypatch):
    class Clock:
        @staticmethod
        def now(*args,**kwargs):return m.END-timedelta(seconds=1)
    monkeypatch.setattr(m,'datetime',Clock)
    monkeypatch.setattr(sys,'argv',['finalize','--stage','final'])
    def forbidden():raise RuntimeError('must not read result payloads before time guard')
    monkeypatch.setattr(m,'collect',forbidden)
    with pytest.raises(AssertionError,match='do not finalize'):m.main()


def test_cannot_overwrite_completed_report(tmp_path,monkeypatch):
    class Clock:
        @staticmethod
        def now(*args,**kwargs):return m.END+timedelta(seconds=1)
    monkeypatch.setattr(m,'datetime',Clock);monkeypatch.setattr(m,'REC',tmp_path)
    monkeypatch.setattr(sys,'argv',['finalize','--stage','final'])
    saved=tmp_path/'RESULT.json';saved.write_text('{"synthetic_fixture":true}')
    with pytest.raises(AssertionError):m.main()
    assert saved.read_text()=='{"synthetic_fixture":true}'
