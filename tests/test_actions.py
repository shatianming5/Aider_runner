from pathlib import Path

from runner.actions import run_pending_actions


def _write_actions(path: Path, text: str) -> None:
    """中文说明：
    - 含义：测试用 helper：写入 `.aider_fsm/actions.yml`（并确保父目录存在）。
    - 内容：`mkdir(parents=True)` + `write_text(utf-8)`，避免每个测试重复样板代码。
    - 可简略：是（可在测试里直接写；保留主要为了可读性）。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_actions_write_file_protected_path_is_blocked(tmp_path: Path):
    """中文说明：
    - 含义：验证 actions 的 `write_file` 不能写入受保护路径（例如 PLAN.md）。
    - 内容：构造 actions.yml 让其尝试覆盖 PLAN.md，期望 runner 阻止并返回特定错误码/原因。
    - 可简略：可能（可参数化更多 protected 路径场景；但单测已覆盖关键规则）。
    """
    repo = tmp_path
    plan = repo / "PLAN.md"
    plan.write_text("# PLAN\n", encoding="utf-8")

    actions_path = repo / ".aider_fsm" / "actions.yml"
    _write_actions(
        actions_path,
        "\n".join(
            [
                "version: 1",
                "actions:",
                "  - id: p0",
                "    kind: write_file",
                "    path: PLAN.md",
                "    content: hacked",
                "",
            ]
        ),
    )

    stage = run_pending_actions(
        repo,
        pipeline=None,
        unattended="strict",
        actions_path=actions_path,
        artifacts_dir=repo / ".aider_fsm" / "artifacts",
        protected_paths=[plan],
    )
    assert stage is not None
    assert stage.ok is False
    assert stage.results[-1].rc == 2
    assert "path_is_protected" in stage.results[-1].stderr


def test_actions_run_cmd_default_safe_blocks_sudo(tmp_path: Path):
    """中文说明：
    - 含义：验证 actions 的 `run_cmd` 在默认安全模式下会拦截 `sudo`。
    - 内容：构造 actions.yml 执行 `sudo echo hi`，期望被 `cmd_allowed`/safe deny 拒绝。
    - 可简略：可能（也可补充其它危险命令；当前用 sudo 覆盖最常见风险）。
    """
    repo = tmp_path
    actions_path = repo / ".aider_fsm" / "actions.yml"
    _write_actions(
        actions_path,
        "\n".join(
            [
                "version: 1",
                "actions:",
                "  - id: a0",
                "    kind: run_cmd",
                "    cmd: sudo echo hi",
                "",
            ]
        ),
    )

    stage = run_pending_actions(
        repo,
        pipeline=None,
        unattended="strict",
        actions_path=actions_path,
        artifacts_dir=repo / ".aider_fsm" / "artifacts",
        protected_paths=[],
    )
    assert stage is not None
    assert stage.ok is False
    assert stage.results[-1].rc == 126
    assert "blocked_by_default_safe_deny" in stage.results[-1].stderr
