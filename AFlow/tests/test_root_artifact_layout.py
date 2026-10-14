import pytest
from scripts.verify_aflow_root_artifacts import validate_root_artifacts


def prepare(root):
    (root/'AFLOW_BRATS_LEADERBOARD.json').write_text('{}')
    (root/'AFLOW_BRATS_HANDOFF_20260813_SINGLE_REPO.md').write_text('历史导航 archive/legacy_root_artifacts_20260914/\n')


def test_minimal_root_passes(tmp_path):
    prepare(tmp_path)
    assert validate_root_artifacts(tmp_path)['legacy_root_artifacts']==0


def test_extra_json_fails(tmp_path):
    prepare(tmp_path);(tmp_path/'new_session.json').write_text('{}')
    with pytest.raises(ValueError):validate_root_artifacts(tmp_path)


def test_sha_list_fails(tmp_path):
    prepare(tmp_path);(tmp_path/'old.sha256').write_text('old')
    with pytest.raises(ValueError):validate_root_artifacts(tmp_path)


def test_old_alias_fails(tmp_path):
    prepare(tmp_path)
    (tmp_path/'FIXED1_DEPLOYMENT_CANDIDATE_20260814.json').symlink_to(tmp_path/'missing')
    with pytest.raises(ValueError):validate_root_artifacts(tmp_path)


def test_long_handoff_fails(tmp_path):
    prepare(tmp_path)
    (tmp_path/'AFLOW_BRATS_HANDOFF_20260813_SINGLE_REPO.md').write_text('history\n'*21)
    with pytest.raises(ValueError):validate_root_artifacts(tmp_path)
