from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .paths import resolve_workdir
from .pipeline_spec import PipelineSpec
from .security import cmd_allowed, looks_interactive, safe_env
from .subprocess_utils import read_text_if_exists, run_cmd_capture, write_cmd_artifacts, write_json, write_text
from .types import CmdResult, StageResult


@dataclass(frozen=True)
class BootstrapSpec:
    version: int = 1
    cmds: list[str] = None  # type: ignore[assignment]
    env: dict[str, str] = None  # type: ignore[assignment]
    workdir: str | None = None
    timeout_seconds: int | None = None
    retries: int = 0

    def __post_init__(self) -> None:
        if self.cmds is None:
            object.__setattr__(self, "cmds", [])
        if self.env is None:
            object.__setattr__(self, "env", {})


_VAR_BRACE_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_VAR_BARE_RE = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)")
_DOLLAR_PLACEHOLDER = "\x00DOLLAR\x00"


def _expand_env_value(raw: str, env: dict[str, str]) -> str:
    # Minimal shell-style expansion for env values:
    # - supports ${VAR} and $VAR
    # - "$$" escapes to literal "$"
    s = str(raw or "")
    s = s.replace("$$", _DOLLAR_PLACEHOLDER)

    def _brace(m: re.Match[str]) -> str:
        return str(env.get(m.group(1)) or "")

    def _bare(m: re.Match[str]) -> str:
        return str(env.get(m.group(1)) or "")

    s = _VAR_BRACE_RE.sub(_brace, s)
    s = _VAR_BARE_RE.sub(_bare, s)
    return s.replace(_DOLLAR_PLACEHOLDER, "$")


def _apply_env_mapping(base: dict[str, str], mapping: dict[str, str]) -> tuple[dict[str, str], dict[str, str]]:
    env = dict(base)
    applied: dict[str, str] = {}
    for k, v in (mapping or {}).items():
        key = str(k or "").strip()
        if not key:
            continue
        value = _expand_env_value(str(v or ""), env)
        env[key] = value
        applied[key] = value
    return env, applied


def _is_sensitive_key(key: str) -> bool:
    k = (key or "").upper()
    return any(x in k for x in ("KEY", "TOKEN", "SECRET", "PASSWORD", "PASS", "PWD"))


def _redact_env(env: dict[str, str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in (env or {}).items():
        if _is_sensitive_key(k):
            out[str(k)] = "***redacted***"
        else:
            out[str(k)] = "" if v is None else str(v)
    return out


def load_bootstrap_spec(path: Path) -> tuple[BootstrapSpec, str]:
    try:
        import yaml  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Missing dependency: PyYAML. Install with `pip install PyYAML`.") from e

    raw = read_text_if_exists(path).strip()
    if not raw:
        raise ValueError(f"bootstrap file is empty: {path}")
    data = yaml.safe_load(raw)
    if not isinstance(data, dict):
        raise ValueError("bootstrap.yml must be a YAML mapping (dict) at the top level")

    version = int(data.get("version") or 1)
    if version != 1:
        raise ValueError(f"unsupported bootstrap version: {version}")

    cmds = data.get("cmds")
    if cmds is None and data.get("steps") is not None:
        cmds = data.get("steps")
    if cmds is None:
        cmds = []
    if not isinstance(cmds, list) or not all(isinstance(x, str) and x.strip() for x in cmds):
        raise ValueError("bootstrap.cmds must be a list of non-empty strings")

    env = data.get("env") or {}
    if env is None:
        env = {}
    if not isinstance(env, dict):
        raise ValueError("bootstrap.env must be a mapping")
    env_out: dict[str, str] = {}
    for k, v in env.items():
        if k is None:
            continue
        ks = str(k).strip()
        if not ks:
            continue
        env_out[ks] = "" if v is None else str(v)

    workdir = str(data.get("workdir")).strip() if data.get("workdir") else None
    timeout_seconds = int(data.get("timeout_seconds")) if data.get("timeout_seconds") else None
    retries = int(data.get("retries") or 0)

    return (
        BootstrapSpec(
            version=version,
            cmds=[c.strip() for c in cmds if c.strip()],
            env=env_out,
            workdir=workdir,
            timeout_seconds=timeout_seconds,
            retries=retries,
        ),
        raw,
    )


def run_bootstrap(
    repo: Path,
    *,
    bootstrap_path: Path,
    pipeline: PipelineSpec | None,
    unattended: str,
    artifacts_dir: Path,
) -> tuple[StageResult, dict[str, str]]:
    """Run bootstrap commands and return (stage_result, applied_env).

    applied_env are the expanded env vars from bootstrap.yml that callers may want to
    apply to subsequent stages (e.g. venv PATH changes).
    """
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    try:
        spec, raw = load_bootstrap_spec(bootstrap_path)
    except Exception as e:
        res = CmdResult(cmd=f"parse {bootstrap_path}", rc=2, stdout="", stderr=str(e), timed_out=False)
        write_cmd_artifacts(artifacts_dir, "bootstrap_parse", res)
        write_json(artifacts_dir / "bootstrap_summary.json", {"ok": False, "failed_index": 0, "total_results": 1})
        return StageResult(ok=False, results=[res], failed_index=0), {}

    write_text(artifacts_dir / "bootstrap.yml", raw)

    # Seed commonly-used variables for env expansion.
    env_base = dict(os.environ)
    env_base["AIDER_FSM_REPO_ROOT"] = str(repo.resolve())

    env_for_cmds, applied_env = _apply_env_mapping(env_base, spec.env)
    env_for_cmds = safe_env(env_for_cmds, {}, unattended=unattended)
    env_for_cmds["AIDER_FSM_STAGE"] = "bootstrap"
    env_for_cmds["AIDER_FSM_ARTIFACTS_DIR"] = str(artifacts_dir.resolve())
    env_for_cmds["AIDER_FSM_REPO_ROOT"] = str(repo.resolve())
    write_json(artifacts_dir / "bootstrap_env.json", _redact_env(applied_env))

    try:
        workdir = resolve_workdir(repo, spec.workdir)
    except Exception as e:
        res = CmdResult(cmd=f"resolve_workdir {spec.workdir}", rc=2, stdout="", stderr=str(e), timed_out=False)
        write_cmd_artifacts(artifacts_dir, "bootstrap_workdir_error", res)
        write_json(artifacts_dir / "bootstrap_summary.json", {"ok": False, "failed_index": 0, "total_results": 1})
        return StageResult(ok=False, results=[res], failed_index=0), applied_env

    results: list[CmdResult] = []
    failed_index: int | None = None

    # No cmds is valid: env-only bootstrap.
    for cmd_idx, raw_cmd in enumerate(spec.cmds, start=1):
        cmd = raw_cmd.strip()
        if not cmd:
            continue

        if unattended == "strict" and looks_interactive(cmd):
            res = CmdResult(
                cmd=cmd,
                rc=126,
                stdout="",
                stderr="likely_interactive_command_disallowed_in_strict_mode",
                timed_out=False,
            )
            results.append(res)
            failed_index = len(results) - 1
            write_cmd_artifacts(artifacts_dir, f"bootstrap_cmd{cmd_idx:02d}_try01", res)
            break

        allowed, reason = cmd_allowed(cmd, pipeline=pipeline)
        if not allowed:
            res = CmdResult(cmd=cmd, rc=126, stdout="", stderr=reason or "blocked", timed_out=False)
            results.append(res)
            failed_index = len(results) - 1
            write_cmd_artifacts(artifacts_dir, f"bootstrap_cmd{cmd_idx:02d}_try01", res)
            break

        eff_timeout = spec.timeout_seconds
        if pipeline and pipeline.security_max_cmd_seconds:
            eff_timeout = (
                int(pipeline.security_max_cmd_seconds)
                if eff_timeout is None
                else min(int(eff_timeout), int(pipeline.security_max_cmd_seconds))
            )

        for attempt in range(1, int(spec.retries) + 2):
            res = run_cmd_capture(cmd, workdir, timeout_seconds=eff_timeout, env=env_for_cmds, interactive=False)
            results.append(res)
            write_cmd_artifacts(artifacts_dir, f"bootstrap_cmd{cmd_idx:02d}_try{attempt:02d}", res)
            if res.rc == 0:
                break

        if results and results[-1].rc != 0:
            failed_index = len(results) - 1
            break

    ok = failed_index is None
    write_json(artifacts_dir / "bootstrap_summary.json", {"ok": ok, "failed_index": failed_index, "total_results": len(results)})

    return StageResult(ok=ok, results=results, failed_index=failed_index), applied_env
