"""Top-level programmatic env API for single-file training scripts.

Example:

  import env

  sess = env.setup("https://github.com/<owner>/<repo>")
  sess.rollout("/abs/path/to/model_dir")
  sess.evaluate()
  sess.teardown()

This module primarily re-exports `runner.env`, but also provides a small
compatibility layer:

- `env.setup({...})` accepts a dict config with `repo=...`.
- `EnvSession.evaluate()` is an alias for `EnvSession.evaluation()`.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from runner.env import (
    EnvSession,
    current_session,
    deploy,
    evaluation,
    rollout,
    rollout_and_evaluation,
    setup as _setup,
    teardown,
)


def _as_path(value: object) -> Path | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    return Path(s).expanduser().resolve()


def setup(target: Any, **kwargs: Any) -> EnvSession:
    """Open an `EnvSession`.

    - `env.setup("https://...")` behaves exactly like `runner.env.setup(...)`.
    - `env.setup({"repo": "...", ...})` is a convenience form for config-driven code.
    """
    if isinstance(target, Mapping):
        cfg = dict(target)
        repo = cfg.get("repo") or cfg.get("target")
        if repo is None or not str(repo).strip():
            raise ValueError("env.setup(config): missing required field `repo`")

        allowed = {
            "clones_dir",
            "pipeline_rel",
            "require_metrics",
            "audit",
            "opencode_model",
            "opencode_repair_model",
            "opencode_url",
            "unattended",
            "opencode_timeout_seconds",
            "opencode_repair_timeout_seconds",
            "opencode_bash",
            "scaffold_opencode_bash",
            "strict_opencode",
            "artifacts_dir",
        }
        call_kwargs = {k: v for k, v in kwargs.items() if k in allowed}
        for k in allowed:
            if k in cfg and cfg.get(k) is not None:
                call_kwargs[k] = cfg.get(k)

        sess = _setup(str(repo), **call_kwargs)

        runtime_env_path = _as_path(cfg.get("runtime_env_path"))
        if runtime_env_path is not None:
            sess.runtime_env_path = runtime_env_path

        return sess

    return _setup(target, **kwargs)


def evaluate(*, session: EnvSession | None = None, **kwargs: Any):
    return evaluation(session=session, **kwargs)


def _sess_evaluate(self: EnvSession, **kwargs: Any):
    return self.evaluation(**kwargs)


if not hasattr(EnvSession, "evaluate"):
    setattr(EnvSession, "evaluate", _sess_evaluate)


__all__ = [
    "EnvSession",
    "setup",
    "deploy",
    "rollout",
    "evaluation",
    "evaluate",
    "rollout_and_evaluation",
    "teardown",
    "current_session",
]

