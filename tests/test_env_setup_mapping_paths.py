from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace


def test_env_setup_mapping_coerces_path_fields(tmp_path: Path, monkeypatch):
    called: dict[str, object] = {}

    from runner import env as runner_env
    from runner.env_local import EnvHandle
    from runner.pipeline_spec import PipelineSpec

    repo = tmp_path / "repo"
    repo.mkdir()
    pipeline_path = repo / "pipeline.yml"
    pipeline_path.write_text("version: 1\n", encoding="utf-8")

    def _fake_open_env(_target: str, **kwargs) -> EnvHandle:  # type: ignore[no-untyped-def]
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
