from __future__ import annotations

import importlib.util
from pathlib import Path


def test_no_repo_level_env_module() -> None:
    """The repo should not ship a top-level `env.py` compatibility module."""
    root = Path(__file__).resolve().parents[1]
    assert not (root / "env.py").exists()

    spec = importlib.util.find_spec("env")
    if spec is None or spec.origin is None:
        return
    # If an unrelated third-party `env` module exists, ensure it's not from this repo.
    assert not Path(spec.origin).resolve().is_relative_to(root)


def test_runner_env_public_surface_is_minimal() -> None:
    from runner import env as runner_env

    assert hasattr(runner_env, "setup")
    assert not hasattr(runner_env, "deploy")
    assert not hasattr(runner_env, "rollout")
    assert not hasattr(runner_env, "evaluation")
    assert not hasattr(runner_env, "rollout_and_evaluation")
    assert not hasattr(runner_env, "teardown")
    assert not hasattr(runner_env, "current_session")

    sess = runner_env.setup  # smoke import path
    assert callable(sess)

    assert hasattr(runner_env, "EnvSession")
    cls = runner_env.EnvSession
    assert hasattr(cls, "rollout")
    assert hasattr(cls, "evaluate")
    assert not hasattr(cls, "deploy")
    assert not hasattr(cls, "evaluation")
    assert not hasattr(cls, "rollout_and_evaluation")
    assert not hasattr(cls, "teardown")

