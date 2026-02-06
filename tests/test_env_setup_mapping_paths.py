from __future__ import annotations

from pathlib import Path


def test_env_setup_mapping_coerces_path_fields(tmp_path: Path, monkeypatch):
    import env

    called: dict[str, object] = {}

    def fake_setup(repo: str, **kwargs):  # type: ignore[no-untyped-def]
        called["repo"] = repo
        called["clones_dir"] = kwargs.get("clones_dir")
        called["artifacts_dir"] = kwargs.get("artifacts_dir")

        class DummySession:
            runtime_env_path = None

        return DummySession()

    monkeypatch.setattr(env, "_setup", fake_setup)

    clones_dir = tmp_path / "clones"
    artifacts_dir = tmp_path / "artifacts"
    _ = env.setup(
        {
            "repo": "https://example.invalid/repo.git",
            "clones_dir": str(clones_dir),
            "artifacts_dir": str(artifacts_dir),
        }
    )

    assert isinstance(called.get("clones_dir"), Path)
    assert called["clones_dir"] == clones_dir.resolve()
    assert isinstance(called.get("artifacts_dir"), Path)
    assert called["artifacts_dir"] == artifacts_dir.resolve()

