from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace


def test_env_setup_mapping_coerces_path_fields(tmp_path: Path, monkeypatch):
    # 作用：pytest 测试用例：验证行为契约
    # 能否简略：否
    # 原因：测试代码（优先可读性）；规模≈32 行；引用次数≈1（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=tests/test_env_setup_mapping_paths.py:8；类型=function；引用≈1；规模≈32行
    called: dict[str, object] = {}

    from runner import env as runner_env
    from runner.env_local import EnvHandle
    from runner.pipeline_spec import PipelineSpec

    repo = tmp_path / "repo"
    repo.mkdir()
    pipeline_path = repo / "pipeline.yml"
    pipeline_path.write_text("version: 1\n", encoding="utf-8")

    def _fake_open_env(_target: str, **kwargs) -> EnvHandle:  # type: ignore[no-untyped-def]
        # 作用：内部符号：test_env_setup_mapping_coerces_path_fields._fake_open_env
        # 能否简略：部分
        # 原因：测试代码（优先可读性）；规模≈4 行；引用次数≈4（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
        # 证据：位置=tests/test_env_setup_mapping_paths.py:20；类型=function；引用≈4；规模≈4行
        called["clones_dir"] = kwargs.get("clones_dir")
        called["artifacts_dir"] = kwargs.get("artifacts_dir")
        return EnvHandle(repo=repo, pipeline_path=pipeline_path, pipeline=PipelineSpec())

    monkeypatch.setattr(runner_env, "open_env", _fake_open_env)
    monkeypatch.setattr(runner_env, "suggest_contract_hints", lambda _repo: SimpleNamespace(commands=[], anchors=[]))

    clones_dir = tmp_path / "clones"
    artifacts_dir = tmp_path / "artifacts"
    _ = runner_env.setup(
        "https://example.invalid/repo.git",
        clones_dir=str(clones_dir),
        artifacts_dir=str(artifacts_dir),
    )

    assert isinstance(called.get("clones_dir"), Path)
    assert called["clones_dir"] == clones_dir.resolve()
    assert isinstance(called.get("artifacts_dir"), Path)
    assert called["artifacts_dir"] == artifacts_dir.resolve()
