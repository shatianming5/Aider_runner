from __future__ import annotations

from pathlib import Path

from runner.opencode_tooling import ToolPolicy, parse_tool_calls


def test_parse_tool_calls_detects_file_write_json_fence():
    """中文说明：
    - 含义：验证 `parse_tool_calls` 能识别 fenced `json` 里的 file-write payload。
    - 内容：提供一个包含 ```json ...``` 的文本，断言解析出 1 个 kind=file 且 payload 字段正确。
    - 可简略：可能（可并入更大的表驱动测试；但单测粒度清晰）。
    """
    text = "hi\n```json\n{\"filePath\":\"PLAN.md\",\"content\":\"# PLAN\\n\"}\n```\n"
    calls = parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].kind == "file"
    payload = calls[0].payload
    assert isinstance(payload, dict)
    assert payload["filePath"] == "PLAN.md"
    assert payload["content"].startswith("# PLAN")


def test_parse_tool_calls_detects_bash_call():
    """中文说明：
    - 含义：验证 `parse_tool_calls` 能识别 fenced `bash` 的工具调用。
    - 内容：提供一个包含 bash tool 的 JSON 参数，断言解析出 kind=bash 且 command 正确。
    - 可简略：可能（可参数化更多命令；当前覆盖主路径）。
    """
    text = "```bash\nbash\n{\"command\":\"git status --porcelain\"}\n```\n"
    calls = parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].kind == "bash"
    payload = calls[0].payload
    assert isinstance(payload, dict)
    assert payload["command"] == "git status --porcelain"


def test_parse_tool_calls_detects_self_closing_write_tag():
    """中文说明：
    - 含义：验证 `parse_tool_calls` 能识别自闭合 `<write ... />` 写文件标签。
    - 内容：输入 `<write filePath=... content=... />`，断言解析出 kind=file 且内容正确解码。
    - 可简略：可能（主要覆盖标签语法的兼容性）。
    """
    text = '<write filePath=\"hello.txt\" content=\"hello\\n\" />'
    calls = parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].kind == "file"
    payload = calls[0].payload
    assert isinstance(payload, dict)
    assert payload["filePath"] == "hello.txt"
    assert payload["content"] == "hello\n"


def test_tool_policy_plan_update_only_allows_plan_md(tmp_path: Path):
    """中文说明：
    - 含义：验证 plan_update_attempt_1 场景下 ToolPolicy 仅允许写 PLAN.md。
    - 内容：构造 policy 并分别尝试写 PLAN.md 与其它文件，断言允许/拒绝与 reason 符合预期。
    - 可简略：否（属于安全边界测试；建议保留以防回归）。
    """
    repo = tmp_path.resolve()
    plan = repo / "PLAN.md"
    pipeline = repo / "pipeline.yml"

    policy = ToolPolicy(
        repo=repo,
        plan_path=plan,
        pipeline_path=pipeline,
        purpose="plan_update_attempt_1",
        bash_mode="restricted",
        unattended="strict",
    )

    ok, reason = policy.allow_file_write(plan)
    assert ok and reason is None

    ok, reason = policy.allow_file_write(repo / "foo.txt")
    assert not ok
    assert reason == "plan_update_allows_only_plan_md"


def test_tool_policy_execute_step_denies_plan_and_pipeline(tmp_path: Path):
    """中文说明：
    - 含义：验证 execute_step 场景禁止写 PLAN.md 与 pipeline.yml，但允许写普通代码文件。
    - 内容：分别对 plan/pipeline/src 文件调用 `allow_file_write`，断言拒绝原因与允许结果正确。
    - 可简略：否（是执行阶段的关键保护，避免 agent 越权改契约/计划）。
    """
    repo = tmp_path.resolve()
    plan = repo / "PLAN.md"
    pipeline = repo / "pipeline.yml"

    policy = ToolPolicy(
        repo=repo,
        plan_path=plan,
        pipeline_path=pipeline,
        purpose="execute_step",
        bash_mode="restricted",
        unattended="strict",
    )

    ok, reason = policy.allow_file_write(plan)
    assert not ok and reason == "execute_step_disallows_plan_md"

    ok, reason = policy.allow_file_write(pipeline)
    assert not ok and reason == "execute_step_disallows_pipeline_yml"

    ok, reason = policy.allow_file_write(repo / "src" / "x.py")
    assert ok and reason is None


def test_tool_policy_scaffold_contract_allows_only_pipeline_and_aider_fsm(tmp_path: Path):
    """中文说明：
    - 含义：验证 scaffold_contract 场景只允许写 pipeline.yml 与 `.aider_fsm/*`。
    - 内容：policy.allow_file_write 对 pipeline/bootstrap.yml 放行，对 src/app.py 拒绝并返回原因。
    - 可简略：否（是 scaffold 合同的关键边界；建议保留）。
    """
    repo = tmp_path.resolve()
    plan = repo / "PLAN.md"
    pipeline = repo / "pipeline.yml"

    policy = ToolPolicy(
        repo=repo,
        plan_path=plan,
        pipeline_path=pipeline,
        purpose="scaffold_contract",
        bash_mode="restricted",
        unattended="strict",
    )

    ok, reason = policy.allow_file_write(pipeline)
    assert ok and reason is None

    ok, reason = policy.allow_file_write(repo / ".aider_fsm" / "bootstrap.yml")
    assert ok and reason is None

    ok, reason = policy.allow_file_write(repo / "src" / "app.py")
    assert not ok
    assert reason == "scaffold_contract_allows_only_pipeline_yml_and_aider_fsm"


def test_tool_policy_restricted_bash_blocks_shell_metacharacters(tmp_path: Path):
    """中文说明：
    - 含义：验证 restricted bash 模式会拦截包含重定向等 shell 元字符的命令。
    - 内容：调用 `allow_bash('echo \"hi\" > hello.txt')`，期望返回 not ok 且 reason 属于预期集合。
    - 可简略：否（属于安全防线测试；建议保留以防策略回退）。
    """
    repo = tmp_path.resolve()
    policy = ToolPolicy(
        repo=repo,
        plan_path=repo / "PLAN.md",
        pipeline_path=None,
        purpose="execute_step",
        bash_mode="restricted",
        unattended="strict",
    )

    ok, reason = policy.allow_bash('echo \"hi\" > hello.txt')
    assert not ok
    assert reason in ("blocked_shell_metacharacters", "blocked_by_restricted_bash_mode")
