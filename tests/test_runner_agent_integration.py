import subprocess
import shutil
import shlex
import sys
from pathlib import Path

import pytest

from runner.agent_client import AgentResult
from runner.runner import RunnerConfig, run


class _FakeAgent:
    """中文说明：
    - 含义：测试用的 Agent 实现，用来模拟 LLM/agent 产生“非法改动”。
    - 内容：在不同 purpose 下主动写入不允许修改的文件（PLAN/pipeline/其它），用于验证 runner 的回滚保护。
    - 可简略：可能（可用 mock 替代；但保留类实现更接近真实 agent 交互）。
    """

    def __init__(self, repo: Path):
        """中文说明：
        - 含义：绑定一个可写的临时 repo 目录。
        - 内容：后续 run() 将在该 repo 内制造非法改动。
        - 可简略：是（测试样板）。
        """
        self._repo = repo

    def run(self, _text: str, *, fsm_state: str, iter_idx: int, purpose: str) -> AgentResult:
        """中文说明：
        - 含义：模拟 agent 的一次对话/行动，并根据 purpose 产生不同副作用。
        - 内容：plan_update_attempt_1 时尝试写 pipeline+普通文件；execute_step 时尝试写 PLAN+pipeline；最后返回固定 assistant_text。
        - 可简略：否（该副作用矩阵是本测试要覆盖的核心行为）。
        """
        # Plan update must not touch code/pipeline; runner should revert these.
        if purpose == "plan_update_attempt_1":
            (self._repo / "foo.txt").write_text(f"hacked at {fsm_state}/{iter_idx}\n", encoding="utf-8")
            (self._repo / "pipeline.yml").write_text("hacked pipeline\n", encoding="utf-8")

        # Execute must not touch PLAN/pipeline; runner should revert these by content.
        if purpose == "execute_step":
            (self._repo / "PLAN.md").write_text("hacked plan\n", encoding="utf-8")
            (self._repo / "pipeline.yml").write_text("hacked pipeline 2\n", encoding="utf-8")

        return AgentResult(assistant_text="ok")

    def close(self) -> None:
        """中文说明：
        - 含义：释放资源（测试中无实际资源）。
        - 内容：空实现以满足 Agent 协议。
        - 可简略：是（测试 stub）。
        """
        return


def _init_git_repo(repo: Path) -> None:
    """中文说明：
    - 含义：把目录初始化成最小 git 仓库（用于 runner 的 diff 检测与回滚）。
    - 内容：执行 `git init` 并配置 user.name/email，避免 commit 失败。
    - 可简略：可能（也可用更轻量的 fake git；但这里直接依赖真实 git 行为更可信）。
    """
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True, text=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=repo, check=True)


def test_runner_reverts_illegal_agent_edits(tmp_path: Path):
    """中文说明：
    - 含义：验证 runner 能回滚 agent 对受保护文件/非法文件的改动。
    - 内容：初始化 git repo 并提交一个 tracked 文件；运行一次带 FakeAgent 的 runner；断言 foo.txt/pipeline.yml/PLAN.md 被恢复或未被污染。
    - 可简略：否（属于 runner 安全边界与一致性的关键回归测试）。
    """
    if not shutil.which("git"):  # pragma: no cover
        pytest.skip("git not available")

    repo = tmp_path
    _init_git_repo(repo)

    # Track a file so `git diff --name-only` detects illegal edits.
    (repo / "foo.txt").write_text("original\n", encoding="utf-8")
    subprocess.run(["git", "add", "foo.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo, check=True, capture_output=True, text=True)

    # Pipeline exists to exercise content-based revert guards.
    (repo / "pipeline.yml").write_text("pipeline: original\n", encoding="utf-8")

    py = shlex.quote(sys.executable)
    ok_cmd = f'{py} -c "import sys; sys.exit(0)"'

    cfg = RunnerConfig(
        repo=repo,
        goal="g",
        model="openai/gpt-4o-mini",
        plan_rel="PLAN.md",
        pipeline_abs=(repo / "pipeline.yml"),
        pipeline_rel="pipeline.yml",
        pipeline=None,
        tests_cmds=[ok_cmd],
        effective_test_cmd=ok_cmd,
        artifacts_base=(repo / ".aider_fsm" / "artifacts"),
        seed_files=[],
        max_iters=1,
        max_fix=1,
        unattended="strict",
        preflight_only=False,
        opencode_url="",
        opencode_timeout_seconds=1,
        opencode_bash="restricted",
    )

    rc = run(cfg, agent=_FakeAgent(repo))
    assert rc == 1  # MAX_ITERS

    assert (repo / "foo.txt").read_text(encoding="utf-8") == "original\n"
    assert (repo / "pipeline.yml").read_text(encoding="utf-8") == "pipeline: original\n"
    assert "hacked plan" not in (repo / "PLAN.md").read_text(encoding="utf-8")
