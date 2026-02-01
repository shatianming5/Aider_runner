from pathlib import Path

from tools.aider_runbook import SENTINEL, create_workspace, parse_last_rc


def test_parse_last_rc_none():
    assert parse_last_rc("no rc here\n") is None


def test_parse_last_rc_last_wins():
    text = f"foo\n{SENTINEL}1\nbar\n{SENTINEL}0\n"
    assert parse_last_rc(text) == 0


def test_create_workspace_under_parent(tmp_path):
    root, repo = create_workspace(parent=tmp_path)
    assert root.is_dir()
    assert repo.parent == root
    assert repo.name == "AIOpsLab"
    assert str(root).startswith(str(tmp_path))
