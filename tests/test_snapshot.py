from pathlib import Path

from runner.snapshot import build_snapshot, get_git_changed_files


def test_build_snapshot_non_git(tmp_path: Path):
    """中文说明：
    - 含义：验证在非 git 仓库目录下也能构造 snapshot 文本。
    - 内容：写入 PLAN.md，调用 `build_snapshot`，断言 snapshot 字段与输出文本包含预期段落。
    - 可简略：可能（可补充 git 仓库分支；当前覆盖 non-git 主路径）。
    """
    plan = tmp_path / "PLAN.md"
    plan.write_text("# PLAN\n", encoding="utf-8")
    snapshot, text = build_snapshot(tmp_path, plan)
    assert snapshot["repo"] == str(tmp_path)
    assert "[SNAPSHOT]" in text
    assert "plan_md:" in text
    assert "git_status_porcelain:" in text


def test_get_git_changed_files_non_git(tmp_path: Path):
    """中文说明：
    - 含义：验证 `get_git_changed_files` 在非 git 目录返回 None。
    - 内容：直接对临时目录调用并断言结果为 None。
    - 可简略：是（简单分支覆盖）。
    """
    assert get_git_changed_files(tmp_path) is None
