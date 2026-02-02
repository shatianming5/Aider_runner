from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


STATE_VERSION = 1
STDIO_TAIL_CHARS = 8000
PLAN_TAIL_CHARS = 20000
ARTIFACT_TEXT_LIMIT_CHARS = 2_000_000
DEFAULT_DOCKERIO_MIRROR = "https://docker.m.daocloud.io"

_STEP_RE = re.compile(
    r"^\s*-\s*\[\s*([xX ])\s*\]\s*\(STEP_ID=([0-9]+)\)\s*(.*?)\s*$"
)

_KIND_IMAGE_RE = re.compile(r"^\s*image:\s*([^\s#]+)\s*$")

_HARD_DENY_PATTERNS: tuple[str, ...] = (
    r"(^|[;&|]\s*)\s*rm\s+-rf\s+/\s*($|[;&|])",
    r"(^|[;&|]\s*)\s*rm\s+-rf\s+/\*\s*($|[;&|])",
    r"(^|[;&|]\s*)\s*rm\s+-rf\s+~\s*($|[;&|])",
    r"(^|[;&|]\s*)\s*rm\s+-rf\s+\$HOME\s*($|[;&|])",
    r"(^|[;&|]\s*)\s*:\(\)\s*\{\s*:\|\:\s*&\s*\}\s*;\s*:\s*($|[;&|])",  # fork bomb
)

_SAFE_DEFAULT_DENY_PATTERNS: tuple[str, ...] = (
    r"\bsudo\b",
    r"\bbrew\s+uninstall\b",
    r"\bdocker\s+system\s+prune\b",
    r"\bdocker\s+volume\s+prune\b",
    r"\bmkfs\b",
    r"\bdd\b",
    r"\bshutdown\b",
    r"\breboot\b",
)


@dataclass(frozen=True)
class CmdResult:
    cmd: str
    rc: int
    stdout: str
    stderr: str
    timed_out: bool = False


@dataclass(frozen=True)
class StageResult:
    ok: bool
    results: list[CmdResult]
    failed_index: int | None = None


@dataclass(frozen=True)
class VerificationResult:
    ok: bool
    failed_stage: str | None
    auth: StageResult | None = None
    tests: StageResult | None = None
    deploy_setup: StageResult | None = None
    deploy_health: StageResult | None = None
    benchmark: StageResult | None = None
    metrics_path: str | None = None
    metrics: dict[str, Any] | None = None
    metrics_errors: list[str] | None = None


@dataclass(frozen=True)
class PipelineSpec:
    version: int = 1

    # tests
    tests_cmds: list[str] = None  # type: ignore[assignment]
    tests_timeout_seconds: int | None = None
    tests_retries: int = 0
    tests_env: dict[str, str] = None  # type: ignore[assignment]
    tests_workdir: str | None = None

    # deploy
    deploy_setup_cmds: list[str] = None  # type: ignore[assignment]
    deploy_health_cmds: list[str] = None  # type: ignore[assignment]
    deploy_teardown_cmds: list[str] = None  # type: ignore[assignment]
    deploy_timeout_seconds: int | None = None
    deploy_retries: int = 0
    deploy_env: dict[str, str] = None  # type: ignore[assignment]
    deploy_workdir: str | None = None
    deploy_teardown_policy: str = "always"  # always|on_success|on_failure|never

    kubectl_dump_enabled: bool = False
    kubectl_dump_namespace: str | None = None
    kubectl_dump_label_selector: str | None = None
    kubectl_dump_include_logs: bool = False

    # benchmark
    benchmark_run_cmds: list[str] = None  # type: ignore[assignment]
    benchmark_timeout_seconds: int | None = None
    benchmark_retries: int = 0
    benchmark_env: dict[str, str] = None  # type: ignore[assignment]
    benchmark_workdir: str | None = None
    benchmark_metrics_path: str | None = None
    benchmark_required_keys: list[str] = None  # type: ignore[assignment]

    # auth (optional)
    auth_cmds: list[str] = None  # type: ignore[assignment]
    auth_timeout_seconds: int | None = None
    auth_retries: int = 0
    auth_env: dict[str, str] = None  # type: ignore[assignment]
    auth_workdir: str | None = None
    auth_interactive: bool = False

    # artifacts
    artifacts_out_dir: str | None = None

    # tooling
    tooling_ensure_tools: bool = False
    tooling_ensure_kind_cluster: bool = False
    tooling_kind_cluster_name: str = "kind"
    tooling_kind_config: str | None = None

    # security
    security_mode: str = "safe"  # safe|system
    security_allowlist: list[str] = None  # type: ignore[assignment]
    security_denylist: list[str] = None  # type: ignore[assignment]
    security_max_cmd_seconds: int | None = None
    security_max_total_seconds: int | None = None

    def __post_init__(self) -> None:
        for attr in (
            "tests_cmds",
            "deploy_setup_cmds",
            "deploy_health_cmds",
            "deploy_teardown_cmds",
            "benchmark_run_cmds",
            "benchmark_required_keys",
            "auth_cmds",
            "security_allowlist",
            "security_denylist",
        ):
            if getattr(self, attr) is None:
                object.__setattr__(self, attr, [])
        for attr in ("tests_env", "deploy_env", "benchmark_env", "auth_env"):
            if getattr(self, attr) is None:
                object.__setattr__(self, attr, {})


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())


def _tail(s: str, n: int) -> str:
    if len(s) <= n:
        return s
    return s[-n:]


def _compile_patterns(patterns: list[str] | tuple[str, ...]) -> list[re.Pattern[str]]:
    compiled: list[re.Pattern[str]] = []
    for raw in patterns:
        p = str(raw)
        if not p.strip():
            continue
        try:
            compiled.append(re.compile(p, re.IGNORECASE))
        except re.error:
            compiled.append(re.compile(re.escape(p), re.IGNORECASE))
    return compiled


def _matches_any(patterns: list[re.Pattern[str]], text: str) -> str | None:
    for p in patterns:
        if p.search(text):
            return p.pattern
    return None


def _looks_interactive(cmd: str) -> bool:
    s = cmd.strip().lower()
    if not s:
        return False

    # Heuristics to avoid hanging in strict unattended runs.
    if s.startswith("docker login") and "--password-stdin" not in s and " -p " not in s and " --password " not in s:
        return True
    if " gh auth login" in f" {s}" and "--with-token" not in s:
        return True
    return False


def _safe_env(base: dict[str, str], extra: dict[str, str], *, unattended: str) -> dict[str, str]:
    env = dict(base)
    env.update({k: str(v) for k, v in extra.items()})
    if unattended == "strict":
        env.setdefault("CI", "1")
        env.setdefault("GIT_TERMINAL_PROMPT", "0")
        env.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
        env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def _resolve_workdir(repo: Path, workdir: str | None) -> Path:
    if not workdir or not str(workdir).strip():
        return repo
    p = Path(workdir).expanduser()
    if not p.is_absolute():
        p = repo / p
    p = p.resolve()
    if not _is_relative_to(p, repo):
        raise ValueError(f"workdir must be within repo: {p} (repo={repo})")
    return p


def _cmd_allowed(cmd: str, *, pipeline: PipelineSpec | None) -> tuple[bool, str | None]:
    cmd = cmd.strip()
    if not cmd:
        return False, "empty_command"

    hard_deny = _compile_patterns(_HARD_DENY_PATTERNS)
    hit = _matches_any(hard_deny, cmd)
    if hit:
        return False, f"blocked_by_hard_deny: {hit}"

    if pipeline is None:
        return True, None

    mode = (pipeline.security_mode or "safe").strip().lower()
    if mode not in ("safe", "system"):
        return False, f"invalid_security_mode: {mode}"

    deny_patterns = list(pipeline.security_denylist or [])
    if mode == "safe":
        deny_patterns.extend(list(_SAFE_DEFAULT_DENY_PATTERNS))
    deny = _compile_patterns(deny_patterns)
    hit = _matches_any(deny, cmd)
    if hit:
        return False, f"blocked_by_denylist: {hit}"

    allow_patterns = list(pipeline.security_allowlist or [])
    if allow_patterns:
        allow = _compile_patterns(allow_patterns)
        if _matches_any(allow, cmd) is None:
            return False, "blocked_by_allowlist"

    return True, None


def run_cmd(cmd: str, cwd: Path) -> tuple[int, str, str]:
    p = subprocess.run(
        cmd,
        cwd=str(cwd),
        shell=True,
        text=True,
        capture_output=True,
    )
    return p.returncode, _tail(p.stdout, STDIO_TAIL_CHARS), _tail(p.stderr, STDIO_TAIL_CHARS)


def run_cmd_capture(
    cmd: str,
    cwd: Path,
    *,
    timeout_seconds: int | None = None,
    env: dict[str, str] | None = None,
    interactive: bool = False,
) -> CmdResult:
    try:
        if interactive:
            p = subprocess.run(
                cmd,
                cwd=str(cwd),
                shell=True,
                text=True,
                env=env,
                timeout=timeout_seconds,
            )
            return CmdResult(cmd=cmd, rc=p.returncode, stdout="", stderr="", timed_out=False)

        p = subprocess.run(
            cmd,
            cwd=str(cwd),
            shell=True,
            text=True,
            capture_output=True,
            env=env,
            timeout=timeout_seconds,
        )
        return CmdResult(cmd=cmd, rc=p.returncode, stdout=p.stdout, stderr=p.stderr, timed_out=False)
    except subprocess.TimeoutExpired as e:
        stdout = e.stdout if isinstance(e.stdout, str) else (e.stdout or b"").decode(errors="replace")
        stderr = e.stderr if isinstance(e.stderr, str) else (e.stderr or b"").decode(errors="replace")
        return CmdResult(cmd=cmd, rc=124, stdout=stdout, stderr=stderr, timed_out=True)


def _limit_text(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]...\n"


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_limit_text(text, ARTIFACT_TEXT_LIMIT_CHARS), encoding="utf-8", errors="replace")


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _ensure_dirs(repo: Path) -> tuple[Path, Path, Path]:
    state_dir = repo / ".aider_fsm"
    logs_dir = state_dir / "logs"
    state_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)
    return state_dir, logs_dir, state_dir / "state.json"


def default_state(
    *, repo: Path, plan_rel: str, model: str, test_cmd: str, pipeline_rel: str | None = None
) -> dict[str, Any]:
    return {
        "version": STATE_VERSION,
        "repo": str(repo),
        "plan_path": plan_rel,
        "pipeline_path": pipeline_rel or "",
        "model": model,
        "test_cmd": test_cmd,
        "iter_idx": 0,
        "fsm_state": "S0_BOOTSTRAP",
        "current_step_id": None,
        "current_step_text": None,
        "fix_attempts": 0,
        "last_test_rc": None,
        "last_deploy_setup_rc": None,
        "last_deploy_health_rc": None,
        "last_benchmark_rc": None,
        "last_metrics_ok": None,
        "last_exit_reason": None,
        "updated_at": _now_iso(),
    }


def load_state(path: Path, defaults: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return dict(defaults)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return dict(defaults)
    if not isinstance(data, dict):
        return dict(defaults)
    merged = dict(defaults)
    merged.update(data)
    merged["updated_at"] = _now_iso()
    return merged


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _resolve_config_path(repo: Path, raw: str) -> Path:
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = repo / p
    return p.resolve()


def _read_text_if_exists(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
        return True
    except Exception:
        return False


def _relpath_or_none(path: Path, base: Path) -> str | None:
    if not _is_relative_to(path, base):
        return None
    return str(path.relative_to(base))


def load_pipeline_spec(path: Path) -> PipelineSpec:
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise RuntimeError("Missing dependency: PyYAML. Install with `pip install PyYAML`.") from e

    raw = _read_text_if_exists(path).strip()
    if not raw:
        raise ValueError(f"pipeline file is empty: {path}")
    data = yaml.safe_load(raw)
    if not isinstance(data, dict):
        raise ValueError("pipeline must be a YAML mapping (dict) at the top level")

    version = int(data.get("version") or 1)
    if version != 1:
        raise ValueError(f"unsupported pipeline version: {version}")

    def _as_mapping(value: Any, name: str) -> dict[str, Any]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError(f"pipeline.{name} must be a mapping")
        return value

    def _as_env(value: Any, name: str) -> dict[str, str]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError(f"pipeline.{name}.env must be a mapping")
        out: dict[str, str] = {}
        for k, v in value.items():
            if k is None:
                continue
            ks = str(k).strip()
            if not ks:
                continue
            out[ks] = "" if v is None else str(v)
        return out

    def _as_cmds(m: dict[str, Any], *, cmd_key: str, cmds_key: str) -> list[str]:
        if cmds_key in m and m.get(cmds_key) is not None:
            v = m.get(cmds_key)
            if not isinstance(v, list) or not all(isinstance(x, str) and x.strip() for x in v):
                raise ValueError(f"pipeline field {cmds_key} must be a list of non-empty strings")
            return [x.strip() for x in v if x.strip()]
        v = m.get(cmd_key)
        if v is None:
            return []
        if not isinstance(v, str) or not v.strip():
            raise ValueError(f"pipeline field {cmd_key} must be a non-empty string")
        return [v.strip()]

    tests = _as_mapping(data.get("tests"), "tests")
    deploy = _as_mapping(data.get("deploy"), "deploy")
    bench = _as_mapping(data.get("benchmark"), "benchmark")
    artifacts = _as_mapping(data.get("artifacts"), "artifacts")
    tooling = _as_mapping(data.get("tooling"), "tooling")
    auth = _as_mapping(data.get("auth"), "auth")
    security = _as_mapping(data.get("security"), "security")

    kubectl_dump = _as_mapping(deploy.get("kubectl_dump"), "deploy.kubectl_dump")

    required_keys = bench.get("required_keys") or []
    if required_keys is None:
        required_keys = []
    if not isinstance(required_keys, list) or not all(isinstance(k, str) for k in required_keys):
        raise ValueError("pipeline.benchmark.required_keys must be a list of strings")

    teardown_policy = str(deploy.get("teardown_policy") or "always").strip().lower()
    if teardown_policy not in ("always", "on_success", "on_failure", "never"):
        raise ValueError("pipeline.deploy.teardown_policy must be one of: always, on_success, on_failure, never")

    kind_cluster_name = str(tooling.get("kind_cluster_name") or "kind").strip() or "kind"

    security_mode = str(security.get("mode") or "safe").strip().lower()
    if security_mode not in ("safe", "system"):
        raise ValueError("pipeline.security.mode must be one of: safe, system")

    allowlist = security.get("allowlist") or []
    if allowlist is None:
        allowlist = []
    if not isinstance(allowlist, list) or not all(isinstance(x, str) for x in allowlist):
        raise ValueError("pipeline.security.allowlist must be a list of strings")

    denylist = security.get("denylist") or []
    if denylist is None:
        denylist = []
    if not isinstance(denylist, list) or not all(isinstance(x, str) for x in denylist):
        raise ValueError("pipeline.security.denylist must be a list of strings")

    # auth: accept cmds or steps as alias
    auth_cmds = _as_cmds(auth, cmd_key="cmd", cmds_key="cmds")
    if not auth_cmds and auth.get("steps") is not None:
        steps = auth.get("steps")
        if not isinstance(steps, list) or not all(isinstance(x, str) and x.strip() for x in steps):
            raise ValueError("pipeline.auth.steps must be a list of non-empty strings")
        auth_cmds = [x.strip() for x in steps if x.strip()]

    return PipelineSpec(
        version=version,
        tests_cmds=_as_cmds(tests, cmd_key="cmd", cmds_key="cmds"),
        tests_timeout_seconds=(int(tests.get("timeout_seconds")) if tests.get("timeout_seconds") else None),
        tests_retries=int(tests.get("retries") or 0),
        tests_env=_as_env(tests.get("env"), "tests"),
        tests_workdir=(str(tests.get("workdir")).strip() if tests.get("workdir") else None),
        deploy_setup_cmds=_as_cmds(deploy, cmd_key="setup_cmd", cmds_key="setup_cmds"),
        deploy_health_cmds=_as_cmds(deploy, cmd_key="health_cmd", cmds_key="health_cmds"),
        deploy_teardown_cmds=_as_cmds(deploy, cmd_key="teardown_cmd", cmds_key="teardown_cmds"),
        deploy_timeout_seconds=(int(deploy.get("timeout_seconds")) if deploy.get("timeout_seconds") else None),
        deploy_retries=int(deploy.get("retries") or 0),
        deploy_env=_as_env(deploy.get("env"), "deploy"),
        deploy_workdir=(str(deploy.get("workdir")).strip() if deploy.get("workdir") else None),
        deploy_teardown_policy=teardown_policy,
        kubectl_dump_enabled=bool(kubectl_dump.get("enabled") or False),
        kubectl_dump_namespace=(str(kubectl_dump.get("namespace")).strip() if kubectl_dump.get("namespace") else None),
        kubectl_dump_label_selector=(
            str(kubectl_dump.get("label_selector")).strip() if kubectl_dump.get("label_selector") else None
        ),
        kubectl_dump_include_logs=bool(kubectl_dump.get("include_logs") or False),
        benchmark_run_cmds=_as_cmds(bench, cmd_key="run_cmd", cmds_key="run_cmds"),
        benchmark_timeout_seconds=(int(bench.get("timeout_seconds")) if bench.get("timeout_seconds") else None),
        benchmark_retries=int(bench.get("retries") or 0),
        benchmark_env=_as_env(bench.get("env"), "benchmark"),
        benchmark_workdir=(str(bench.get("workdir")).strip() if bench.get("workdir") else None),
        benchmark_metrics_path=(str(bench.get("metrics_path")).strip() if bench.get("metrics_path") else None),
        benchmark_required_keys=[str(k).strip() for k in required_keys if str(k).strip()],
        auth_cmds=auth_cmds,
        auth_timeout_seconds=(int(auth.get("timeout_seconds")) if auth.get("timeout_seconds") else None),
        auth_retries=int(auth.get("retries") or 0),
        auth_env=_as_env(auth.get("env"), "auth"),
        auth_workdir=(str(auth.get("workdir")).strip() if auth.get("workdir") else None),
        auth_interactive=bool(auth.get("interactive") or False),
        artifacts_out_dir=(str(artifacts.get("out_dir")).strip() if artifacts.get("out_dir") else None),
        tooling_ensure_tools=bool(tooling.get("ensure_tools") or False),
        tooling_ensure_kind_cluster=bool(tooling.get("ensure_kind_cluster") or False),
        tooling_kind_cluster_name=kind_cluster_name,
        tooling_kind_config=(str(tooling.get("kind_config")).strip() if tooling.get("kind_config") else None),
        security_mode=security_mode,
        security_allowlist=[str(x).strip() for x in allowlist if str(x).strip()],
        security_denylist=[str(x).strip() for x in denylist if str(x).strip()],
        security_max_cmd_seconds=(int(security.get("max_cmd_seconds")) if security.get("max_cmd_seconds") else None),
        security_max_total_seconds=(int(security.get("max_total_seconds")) if security.get("max_total_seconds") else None),
    )


def load_actions_spec(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise RuntimeError("Missing dependency: PyYAML. Install with `pip install PyYAML`.") from e

    raw = _read_text_if_exists(path).strip()
    if not raw:
        raise ValueError(f"actions file is empty: {path}")
    data = yaml.safe_load(raw)
    if not isinstance(data, dict):
        raise ValueError("actions.yml must be a YAML mapping (dict) at the top level")
    version = int(data.get("version") or 1)
    if version != 1:
        raise ValueError(f"unsupported actions version: {version}")
    actions = data.get("actions") or []
    if actions is None:
        actions = []
    if not isinstance(actions, list) or not all(isinstance(a, dict) for a in actions):
        raise ValueError("actions.yml actions must be a list of mappings")
    return {"version": version, "actions": actions, "raw": raw}


def run_pending_actions(
    repo: Path,
    *,
    pipeline: PipelineSpec | None,
    unattended: str,
    actions_path: Path,
    artifacts_dir: Path,
    protected_paths: list[Path] | None = None,
) -> StageResult | None:
    if not actions_path.exists():
        return None

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    try:
        spec = load_actions_spec(actions_path)
    except Exception as e:
        res = CmdResult(cmd=f"parse {actions_path}", rc=2, stdout="", stderr=str(e), timed_out=False)
        _write_cmd_artifacts(artifacts_dir, "actions_parse", res)
        return StageResult(ok=False, results=[res], failed_index=0)

    _write_text(artifacts_dir / "actions.yml", spec["raw"])

    protected = {p.resolve() for p in (protected_paths or []) if p}

    env_base = dict(os.environ)
    env = _safe_env(env_base, {}, unattended=unattended)
    workdir = repo

    results: list[CmdResult] = []
    failed_index: int | None = None

    for idx, action in enumerate(spec["actions"], start=1):
        action_id = str(action.get("id") or f"action-{idx:03d}").strip() or f"action-{idx:03d}"
        kind = str(action.get("kind") or "run_cmd").strip().lower()
        cmd = str(action.get("cmd") or "").strip()
        timeout_seconds = action.get("timeout_seconds")
        retries = int(action.get("retries") or 0)

        if kind not in ("run_cmd", "install_tool", "start_service", "write_file"):
            res = CmdResult(
                cmd=cmd or f"<{action_id}>",
                rc=2,
                stdout="",
                stderr=f"unsupported_action_kind: {kind}",
                timed_out=False,
            )
            results.append(res)
            failed_index = len(results) - 1
            _write_cmd_artifacts(artifacts_dir, f"actions_{idx:02d}_{action_id}", res)
            break

        if not cmd and kind != "write_file":
            res = CmdResult(
                cmd=f"<{action_id}>",
                rc=2,
                stdout="",
                stderr="missing_cmd",
                timed_out=False,
            )
            results.append(res)
            failed_index = len(results) - 1
            _write_cmd_artifacts(artifacts_dir, f"actions_{idx:02d}_{action_id}", res)
            break

        if kind == "write_file":
            path_raw = str(action.get("path") or "").strip()
            content = str(action.get("content") or "")
            if not path_raw:
                res = CmdResult(cmd=f"<{action_id}>", rc=2, stdout="", stderr="missing_path", timed_out=False)
                results.append(res)
                failed_index = len(results) - 1
                _write_cmd_artifacts(artifacts_dir, f"actions_{idx:02d}_{action_id}", res)
                break
            try:
                out_path = _resolve_config_path(repo, path_raw)
                if not _is_relative_to(out_path, repo):
                    raise ValueError("path_outside_repo")
                if out_path in protected:
                    raise ValueError("path_is_protected")
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(content, encoding="utf-8")
                res = CmdResult(cmd=f"write_file {out_path}", rc=0, stdout="", stderr="", timed_out=False)
            except Exception as e:
                res = CmdResult(cmd=f"write_file {path_raw}", rc=2, stdout="", stderr=str(e), timed_out=False)

            results.append(res)
            _write_cmd_artifacts(artifacts_dir, f"actions_{idx:02d}_{action_id}", res)
            if res.rc != 0:
                failed_index = len(results) - 1
                break
            continue

        if unattended == "strict" and _looks_interactive(cmd):
            res = CmdResult(
                cmd=cmd,
                rc=126,
                stdout="",
                stderr="likely_interactive_command_disallowed_in_strict_mode",
                timed_out=False,
            )
            results.append(res)
            failed_index = len(results) - 1
            _write_cmd_artifacts(artifacts_dir, f"actions_{idx:02d}_{action_id}", res)
            break

        allowed, reason = _cmd_allowed(cmd, pipeline=pipeline)
        if not allowed:
            res = CmdResult(cmd=cmd, rc=126, stdout="", stderr=reason or "blocked", timed_out=False)
            results.append(res)
            failed_index = len(results) - 1
            _write_cmd_artifacts(artifacts_dir, f"actions_{idx:02d}_{action_id}", res)
            break

        eff_timeout: int | None = int(timeout_seconds) if timeout_seconds else None
        if pipeline and pipeline.security_max_cmd_seconds:
            eff_timeout = (
                int(pipeline.security_max_cmd_seconds)
                if eff_timeout is None
                else min(int(eff_timeout), int(pipeline.security_max_cmd_seconds))
            )

        for attempt in range(1, retries + 2):
            res = run_cmd_capture(cmd, workdir, timeout_seconds=eff_timeout, env=env, interactive=False)
            results.append(res)
            _write_cmd_artifacts(
                artifacts_dir,
                f"actions_{idx:02d}_{action_id}_try{attempt:02d}",
                res,
            )
            if res.rc == 0:
                break
        if results and results[-1].rc != 0:
            failed_index = len(results) - 1
            break

    ok = failed_index is None
    _write_json(
        artifacts_dir / "actions_summary.json",
        {"ok": ok, "failed_index": failed_index, "total_results": len(results)},
    )

    try:
        actions_path.unlink()
    except Exception:
        # Best effort. Keeping the file may cause repeated execution; record a warning.
        _write_text(artifacts_dir / "actions_warning.txt", f"failed to delete actions file: {actions_path}\n")

    return StageResult(ok=ok, results=results, failed_index=failed_index)


def plan_template(goal: str, test_cmd: str, *, pipeline: PipelineSpec | None = None) -> str:
    goal = goal.strip() or "<fill goal>"
    acceptance: list[str] = [f"- [ ] TEST_CMD passes: `{test_cmd}`"]
    if pipeline:
        if pipeline.deploy_setup_cmds or pipeline.deploy_health_cmds:
            acceptance.append("- [ ] Deploy succeeds (see pipeline.yml)")
        if pipeline.benchmark_run_cmds:
            acceptance.append("- [ ] Benchmark succeeds (see pipeline.yml)")
        if pipeline.benchmark_metrics_path or pipeline.benchmark_required_keys:
            acceptance.append("- [ ] Metrics file/keys present (see pipeline.yml)")
    return (
        "# PLAN\n"
        "\n"
        "## Goal\n"
        f"- {goal}\n"
        "\n"
        "## Acceptance\n"
        + "\n".join(acceptance)
        + "\n"
        "\n"
        "## Next (exactly ONE item)\n"
        "- [ ] (STEP_ID=001) 生成 Backlog：把目标拆成最小不可分的若干步（每步一次编辑+一次验收可闭合）\n"
        "\n"
        "## Backlog\n"
        "\n"
        "## Done\n"
        "- [x] (STEP_ID=000) 初始化计划文件\n"
        "\n"
        "## Notes\n"
        "- \n"
    )


def ensure_plan_file(plan_abs: Path, goal: str, test_cmd: str, *, pipeline: PipelineSpec | None = None) -> None:
    if plan_abs.exists():
        return
    plan_abs.parent.mkdir(parents=True, exist_ok=True)
    plan_abs.write_text(plan_template(goal, test_cmd, pipeline=pipeline), encoding="utf-8")


def _extract_section_lines(lines: list[str], heading_prefix: str) -> list[str] | None:
    start = None
    for i, line in enumerate(lines):
        if line.strip().startswith(heading_prefix):
            start = i + 1
            break
    if start is None:
        return None
    end = len(lines)
    for j in range(start, len(lines)):
        if lines[j].strip().startswith("## "):
            end = j
            break
    return lines[start:end]


def _parse_step_lines(section_lines: list[str]) -> tuple[list[dict[str, Any]], list[str]]:
    steps: list[dict[str, Any]] = []
    bad: list[str] = []
    for line in section_lines:
        if not re.match(r"^\s*-\s*\[", line):
            continue
        m = _STEP_RE.match(line)
        if not m:
            bad.append(line.strip())
            continue
        checked = m.group(1).lower() == "x"
        steps.append({"id": m.group(2), "text": m.group(3).strip(), "checked": checked})
    return steps, bad


def find_duplicate_step_ids(plan_text: str) -> list[str]:
    lines = plan_text.splitlines()
    ids: list[str] = []
    for heading in ("## Next", "## Backlog", "## Done"):
        section = _extract_section_lines(lines, heading)
        if section is None:
            continue
        steps, _bad = _parse_step_lines(section)
        ids.extend([s["id"] for s in steps])
    counts: dict[str, int] = {}
    for sid in ids:
        counts[sid] = counts.get(sid, 0) + 1
    return [sid for sid, c in counts.items() if c > 1]


def parse_next_step(plan_text: str) -> tuple[dict[str, str] | None, str | None]:
    lines = plan_text.splitlines()
    section = _extract_section_lines(lines, "## Next")
    if section is None:
        return None, "missing_next_section"
    steps, bad = _parse_step_lines(section)
    if bad:
        return None, "bad_next_line"
    if len(steps) == 0:
        return None, None
    if len(steps) != 1:
        return None, "next_count_not_one"
    if steps[0]["checked"]:
        return None, "next_is_checked"
    dups = find_duplicate_step_ids(plan_text)
    if dups:
        return None, "duplicate_step_id"
    return {"id": steps[0]["id"], "text": steps[0]["text"]}, None


def parse_backlog_open_count(plan_text: str) -> tuple[int, str | None]:
    lines = plan_text.splitlines()
    section = _extract_section_lines(lines, "## Backlog")
    if section is None:
        return 0, "missing_backlog_section"
    steps, bad = _parse_step_lines(section)
    if bad:
        return 0, "bad_backlog_line"
    return sum(1 for s in steps if not s["checked"]), None


def parse_plan(plan_text: str) -> dict[str, Any]:
    next_step, next_err = parse_next_step(plan_text)
    backlog_open, backlog_err = parse_backlog_open_count(plan_text)
    errors: list[str] = []
    if next_err:
        errors.append(next_err)
    if backlog_err:
        errors.append(backlog_err)
    return {
        "next_step": next_step,
        "backlog_open_count": backlog_open,
        "errors": errors,
    }


def build_snapshot(repo: Path, plan_abs: Path, pipeline_abs: Path | None = None) -> tuple[dict[str, Any], str]:
    plan_text = ""
    if plan_abs.exists():
        plan_text = plan_abs.read_text(encoding="utf-8", errors="replace")
    plan_text = _tail(plan_text, PLAN_TAIL_CHARS)

    pipeline_text = ""
    if pipeline_abs and pipeline_abs.exists():
        pipeline_text = pipeline_abs.read_text(encoding="utf-8", errors="replace")
    pipeline_text = _tail(pipeline_text, PLAN_TAIL_CHARS)

    actions_text = ""
    actions_path = repo / ".aider_fsm" / "actions.yml"
    if actions_path.exists():
        actions_text = actions_path.read_text(encoding="utf-8", errors="replace")
    actions_text = _tail(actions_text, PLAN_TAIL_CHARS)

    state_text = ""
    state_path = repo / ".aider_fsm" / "state.json"
    if state_path.exists():
        state_text = state_path.read_text(encoding="utf-8", errors="replace")
    state_text = _tail(state_text, PLAN_TAIL_CHARS)

    rc1, gs, ge = run_cmd("git status --porcelain", repo)
    rc2, diff, de = run_cmd("git diff --stat", repo)
    rc3, names, ne = run_cmd("git diff --name-only", repo)

    env_probe = {
        "pyproject.toml": (repo / "pyproject.toml").exists(),
        "package.json": (repo / "package.json").exists(),
        "pytest.ini": (repo / "pytest.ini").exists(),
        "requirements.txt": (repo / "requirements.txt").exists(),
        "setup.cfg": (repo / "setup.cfg").exists(),
    }

    snapshot = {
        "repo": str(repo),
        "plan_path": str(plan_abs),
        "pipeline_path": str(pipeline_abs) if pipeline_abs else "",
        "env_probe": env_probe,
        "git": {
            "status_rc": rc1,
            "status": gs,
            "status_err": ge,
            "diff_stat_rc": rc2,
            "diff_stat": diff,
            "diff_stat_err": de,
            "diff_names_rc": rc3,
            "diff_names": names,
            "diff_names_err": ne,
        },
        "plan_md": plan_text,
        "pipeline_yml": pipeline_text,
        "actions_yml": actions_text,
        "state_json": state_text,
    }

    env_lines = "\n".join([f"- {k}: {'yes' if v else 'no'}" for k, v in env_probe.items()])
    snapshot_text = (
        "[SNAPSHOT]\n"
        f"repo: {repo}\n"
        f"plan_path: {plan_abs}\n"
        f"pipeline_path: {pipeline_abs}\n"
        "env_probe:\n"
        f"{env_lines}\n"
        "git_status_porcelain:\n"
        f"{gs}\n"
        "git_diff_stat:\n"
        f"{diff}\n"
        "changed_files:\n"
        f"{names}\n"
        "state_json:\n"
        f"{state_text}\n"
        "pipeline_yml:\n"
        f"{pipeline_text}\n"
        "actions_yml:\n"
        f"{actions_text}\n"
        "plan_md:\n"
        f"{plan_text}\n"
    )
    return snapshot, snapshot_text


def get_git_changed_files(repo: Path) -> list[str] | None:
    rc, out, _err = run_cmd("git diff --name-only", repo)
    if rc != 0:
        return None
    return [line.strip() for line in out.splitlines() if line.strip()]


def git_checkout(repo: Path, paths: list[str]) -> tuple[int, str, str]:
    if not paths:
        return 0, "", ""
    cmd = "git checkout -- " + " ".join(shlex.quote(p) for p in paths)
    return run_cmd(cmd, repo)


def non_plan_changes(changed_files: list[str], plan_rel: str) -> list[str]:
    return [p for p in changed_files if p != plan_rel]


def create_coder(*, model_name: str, fnames: list[str]):
    try:
        from aider.coders import Coder
        from aider.io import InputOutput
        from aider.models import Model
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Failed to import aider. Install deps with `pip install -r requirements.txt`."
        ) from e

    io = InputOutput(yes=True)
    model = Model(model_name)
    return Coder.create(main_model=model, fnames=fnames, io=io)


def _run_py_script(script: Path, args: list[str], *, cwd: Path) -> tuple[int, str, str]:
    cmd = [sys.executable, str(script), *args]
    p = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True)
    return p.returncode, _tail(p.stdout, STDIO_TAIL_CHARS), _tail(p.stderr, STDIO_TAIL_CHARS)


def _find_python312() -> str | None:
    for candidate in ("python3.12", "/opt/homebrew/bin/python3.12"):
        if "/" in candidate:
            if Path(candidate).exists():
                return candidate
            continue
        path = shutil.which(candidate)
        if path:
            return path
    return None


def _dockerhub_registry_reachable(repo: Path) -> bool:
    rc, _out, _err = run_cmd("curl -I -sS --max-time 5 https://registry-1.docker.io/v2/ >/dev/null", repo)
    return rc == 0


def _dockerio_mirror_url() -> str | None:
    v = (os.environ.get("KIND_DOCKERIO_MIRROR") or os.environ.get("AIDER_FSM_DOCKERIO_MIRROR") or "").strip()
    if v.lower() in ("0", "false", "none", "off"):
        return None
    return v or DEFAULT_DOCKERIO_MIRROR


def _strip_url_scheme(url: str) -> str:
    return re.sub(r"^https?://", "", url.strip())


def _dockerhub_mirror_ref(image: str, mirror_host: str) -> str | None:
    image = image.strip()
    if not image:
        return None

    # If the image already specifies a non-docker.io registry, don't rewrite.
    first = image.split("/", 1)[0]
    if "." in first or ":" in first:
        if first == "docker.io" and "/" in image:
            image = image.split("/", 1)[1]
        else:
            return None

    if "/" not in image:
        image = f"library/{image}"

    mirror_host = mirror_host.strip().rstrip("/")
    return f"{mirror_host}/{image}"


def _docker_image_exists(repo: Path, image: str) -> bool:
    rc, _out, _err = run_cmd(f"docker image inspect {shlex.quote(image)} >/dev/null 2>&1", repo)
    return rc == 0


def _docker_pull(repo: Path, image: str) -> bool:
    rc, _out, _err = run_cmd(f"docker pull {shlex.quote(image)}", repo)
    return rc == 0


def _dockerfile_base_image(dockerfile: Path) -> str | None:
    for raw in dockerfile.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.upper().startswith("FROM "):
            return line.split(None, 1)[1].strip()
    return None


def _is_aiopslab_kind_node_image(image: str) -> bool:
    return bool(re.search(r"/aiopslab-kind-(arm|x86):latest$", image))


def _ensure_aiopslab_kind_node_image(repo: Path, image: str, *, mirror_url: str | None) -> None:
    dockerfile = repo / "kind" / "Dockerfile"
    if not dockerfile.exists():
        raise RuntimeError("AIOpsLab kind image build requires kind/Dockerfile but it was not found.")

    kind_context = repo / "kind"

    base = _dockerfile_base_image(dockerfile)
    if not base:
        raise RuntimeError("Failed to determine base image from kind/Dockerfile.")

    if not _docker_image_exists(repo, base):
        if not _docker_pull(repo, base):
            if not mirror_url:
                raise RuntimeError("Docker Hub is unreachable and KIND_DOCKERIO_MIRROR is disabled.")
            mirror_host = _strip_url_scheme(mirror_url)
            mirror_ref = _dockerhub_mirror_ref(base, mirror_host)
            if not mirror_ref:
                raise RuntimeError(f"Cannot mirror-pull non-dockerhub base image: {base}")
            if not _docker_pull(repo, mirror_ref):
                raise RuntimeError(f"Failed to pull base image from mirror: {mirror_ref}")
            run_cmd(f"docker tag {shlex.quote(mirror_ref)} {shlex.quote(base)}", repo)

    rc, out, err = run_cmd(
        " ".join(
            [
                "docker",
                "build",
                "-t",
                shlex.quote(image),
                "-f",
                shlex.quote(str(dockerfile)),
                shlex.quote(str(kind_context)),
            ]
        ),
        repo,
    )
    if rc != 0:
        raise RuntimeError(f"Failed to build kind node image {image}:\n{out}\n{err}")


def _read_kind_node_images(cfg: Path) -> list[str]:
    images: list[str] = []
    for line in cfg.read_text(encoding="utf-8", errors="replace").splitlines():
        m = _KIND_IMAGE_RE.match(line)
        if not m:
            continue
        img = m.group(1).strip()
        if img and img not in images:
            images.append(img)
    return images


def _write_kind_config_with_dockerio_mirror(cfg: Path, mirror_url: str, out_dir: Path) -> Path:
    mirror_url = mirror_url.strip()
    if not mirror_url:
        return cfg

    text = cfg.read_text(encoding="utf-8", errors="replace")
    if "containerdConfigPatches" in text:
        return cfg

    lines = text.splitlines(True)
    insert_at = None
    for i, line in enumerate(lines):
        if line.strip().startswith("apiVersion:"):
            insert_at = i + 1
            break
    if insert_at is None:
        insert_at = 0

    block = (
        "containerdConfigPatches:\n"
        "  - |-\n"
        "    [plugins.\"io.containerd.grpc.v1.cri\".registry.mirrors.\"docker.io\"]\n"
        f"      endpoint = [\"{mirror_url}\"]\n"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{cfg.stem}-mirror{cfg.suffix}"
    out_path.write_text("".join(lines[:insert_at]) + block + "".join(lines[insert_at:]), encoding="utf-8")
    return out_path


def _ensure_kind_can_pull_image(repo: Path, image: str) -> None:
    name = "aider-fsm-pulltest"
    run_cmd(f"kubectl delete pod {shlex.quote(name)} --ignore-not-found", repo)
    run_cmd(
        " ".join(
            [
                "kubectl",
                "run",
                shlex.quote(name),
                f"--image={shlex.quote(image)}",
                "--restart=Never",
                "--command",
                "--",
                "sleep",
                "3600",
            ]
        ),
        repo,
    )
    rc, out, err = run_cmd(f"kubectl wait --for=condition=Ready pod/{shlex.quote(name)} --timeout=120s", repo)
    run_cmd(f"kubectl delete pod {shlex.quote(name)} --ignore-not-found --wait=false", repo)
    if rc != 0:
        raise RuntimeError(f"Kind cannot pull images from docker.io (or mirror):\n{out}\n{err}")


def _ensure_aiopslab_config(repo: Path) -> None:
    example = repo / "aiopslab" / "config.yml.example"
    cfg = repo / "aiopslab" / "config.yml"
    if cfg.exists():
        return
    if not example.exists():
        raise RuntimeError("Full QuickStart requires aiopslab/config.yml.example but it was not found.")

    text = example.read_text(encoding="utf-8", errors="replace").splitlines(True)
    user = os.environ.get("USER") or os.environ.get("LOGNAME") or "user"
    out: list[str] = []
    for line in text:
        if line.lstrip().startswith("k8s_host:"):
            out.append("k8s_host: kind\n")
        elif line.lstrip().startswith("k8s_user:"):
            out.append(f"k8s_user: {user}\n")
        else:
            out.append(line)
    cfg.write_text("".join(out), encoding="utf-8")


def _ensure_aiopslab_venv(repo: Path) -> None:
    venv_py = repo / ".venv_aiopslab" / "bin" / "python"
    if venv_py.exists():
        return

    py312 = _find_python312()
    if not py312:
        raise RuntimeError("Full QuickStart requires python3.12 but it was not found on PATH.")

    run_cmd(f"{shlex.quote(py312)} -m venv .venv_aiopslab", repo)
    run_cmd(".venv_aiopslab/bin/python -m pip install -e .", repo)


def _ensure_kind_cluster(repo: Path) -> None:
    kind_dir = repo / "kind"
    arm_cfg = kind_dir / "kind-config-arm.yaml"
    x86_cfg = kind_dir / "kind-config-x86.yaml"

    machine = platform.machine().lower()
    if machine in ("arm64", "aarch64") and arm_cfg.exists():
        cfg = arm_cfg
    elif x86_cfg.exists():
        cfg = x86_cfg
    elif arm_cfg.exists():
        cfg = arm_cfg
    else:
        raise RuntimeError("Full QuickStart requires kind config under kind/ but none was found.")

    mirror_url = _dockerio_mirror_url()
    if mirror_url and not _dockerhub_registry_reachable(repo):
        cfg = _write_kind_config_with_dockerio_mirror(cfg, mirror_url, repo / ".aider_fsm")

    for image in _read_kind_node_images(cfg):
        if _docker_image_exists(repo, image):
            continue
        if _is_aiopslab_kind_node_image(image):
            _ensure_aiopslab_kind_node_image(repo, image, mirror_url=mirror_url)
            continue
        if _docker_pull(repo, image):
            continue
        if mirror_url:
            mirror_ref = _dockerhub_mirror_ref(image, _strip_url_scheme(mirror_url))
            if mirror_ref and _docker_pull(repo, mirror_ref):
                run_cmd(f"docker tag {shlex.quote(mirror_ref)} {shlex.quote(image)}", repo)
                continue
        raise RuntimeError(f"Failed to pull required kind node image: {image}")

    rc, out, _err = run_cmd("kind get clusters", repo)
    clusters = {line.strip() for line in out.splitlines() if line.strip()} if rc == 0 else set()
    if "kind" not in clusters:
        run_cmd(f"kind create cluster --config {shlex.quote(str(cfg))}", repo)

    run_cmd("kubectl wait --for=condition=Ready nodes --all --timeout=180s", repo)
    _ensure_kind_can_pull_image(repo, "alpine:3.19")


def _ensure_kind_cluster_generic(repo: Path, *, name: str = "kind", config: Path | None = None) -> None:
    name = (name or "kind").strip()
    rc, out, err = run_cmd("kind get clusters", repo)
    if rc != 0:
        raise RuntimeError(f"Failed to list kind clusters:\n{out}\n{err}")

    clusters = {line.strip() for line in out.splitlines() if line.strip()}
    if name not in clusters:
        cmd = ["kind", "create", "cluster", "--name", shlex.quote(name)]
        if config is not None:
            cmd.extend(["--config", shlex.quote(str(config))])
        rc2, out2, err2 = run_cmd(" ".join(cmd), repo)
        if rc2 != 0:
            raise RuntimeError(f"Failed to create kind cluster {name}:\n{out2}\n{err2}")

    # Ensure context is ready enough for kubectl commands.
    rc3, out3, err3 = run_cmd("kubectl wait --for=condition=Ready nodes --all --timeout=180s", repo)
    if rc3 != 0:
        raise RuntimeError(f"Kind cluster not ready:\n{out3}\n{err3}")


def _default_full_quickstart_test_cmd() -> str:
    # Non-interactive checks that imply QuickStart prerequisites are working.
    return (
        "kubectl wait --for=condition=Ready nodes --all --timeout=180s"
        " && test -f aiopslab/config.yml"
        " && test -x .venv_aiopslab/bin/python"
        " && .venv_aiopslab/bin/python -c 'from aiopslab.orchestrator import Orchestrator; "
        "o=Orchestrator(); "
        "print(\"PROBLEMS\", len(o.probs.get_problem_ids())); "
        "print(\"NAMESPACES\", len(o.kubectl.list_namespaces().items))'"
    )


def _probe_versions(repo: Path) -> dict[str, Any]:
    def _cmd(name: str, cmd: str) -> dict[str, Any]:
        rc, out, err = run_cmd(cmd, repo)
        return {"name": name, "cmd": cmd, "rc": rc, "out_tail": out, "err_tail": err}

    git_head = _cmd("git_head", "git rev-parse HEAD")
    if git_head["rc"] != 0:
        git_head = _cmd("git_head", "echo -n ''")

    return {
        "ts": _now_iso(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": {"version": sys.version, "executable": sys.executable},
        "git": {"head": (git_head["out_tail"] or "").strip()},
        "tools": [
            _cmd("docker", "docker version"),
            _cmd("kubectl", "kubectl version --client"),
            _cmd("kind", "kind version"),
            _cmd("helm", "helm version"),
            _cmd("colima", "colima version"),
        ],
    }


def _write_cmd_artifacts(out_dir: Path, prefix: str, res: CmdResult) -> None:
    _write_text(out_dir / f"{prefix}_cmd.txt", res.cmd + "\n")
    _write_text(out_dir / f"{prefix}_stdout.txt", res.stdout)
    _write_text(out_dir / f"{prefix}_stderr.txt", res.stderr)
    _write_json(
        out_dir / f"{prefix}_result.json",
        {"rc": res.rc, "timed_out": res.timed_out},
    )


def _read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return None, f"failed_to_read: {e}"
    try:
        data = json.loads(raw)
    except Exception as e:
        return None, f"invalid_json: {e}"
    if not isinstance(data, dict):
        return None, "metrics_json_not_object"
    return data, None


def _validate_metrics(metrics: dict[str, Any], required_keys: list[str]) -> list[str]:
    missing: list[str] = []
    for k in required_keys:
        if k not in metrics:
            missing.append(k)
    return missing


def _dump_kubectl(
    out_dir: Path,
    repo: Path,
    *,
    namespace: str | None,
    label_selector: str | None,
    include_logs: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmds: list[tuple[str, str]] = [
        ("kubectl_get_nodes", "kubectl get nodes -o wide"),
        ("kubectl_get_namespaces", "kubectl get namespaces"),
        ("kubectl_get_pods", "kubectl get pods -A -o wide"),
        ("kubectl_get_all", "kubectl get all -A -o wide"),
        ("kubectl_get_events", "kubectl get events -A --sort-by=.metadata.creationTimestamp"),
    ]
    for prefix, cmd in cmds:
        res = run_cmd_capture(cmd, repo, timeout_seconds=60)
        _write_cmd_artifacts(out_dir, prefix, res)

    if include_logs and label_selector:
        if namespace:
            cmd = (
                f"kubectl logs -n {shlex.quote(namespace)} -l {shlex.quote(label_selector)} "
                "--all-containers=true --timestamps --tail=2000"
            )
        else:
            cmd = (
                f"kubectl logs --all-namespaces -l {shlex.quote(label_selector)} "
                "--all-containers=true --timestamps --tail=2000"
            )
        res = run_cmd_capture(cmd, repo, timeout_seconds=120)
        _write_cmd_artifacts(out_dir, "kubectl_logs", res)


def _stage_rc(stage: StageResult | None) -> int | None:
    if stage is None or not stage.results:
        return None
    if stage.failed_index is None:
        return stage.results[-1].rc
    if 0 <= stage.failed_index < len(stage.results):
        return stage.results[stage.failed_index].rc
    return stage.results[-1].rc


def _stage_failed_cmd(stage: StageResult | None) -> CmdResult | None:
    if stage is None or not stage.results:
        return None
    if stage.failed_index is not None and 0 <= stage.failed_index < len(stage.results):
        return stage.results[stage.failed_index]
    return stage.results[-1]


def _run_stage(
    repo: Path,
    *,
    stage: str,
    cmds: list[str],
    workdir: Path,
    env: dict[str, str],
    timeout_seconds: int | None,
    retries: int,
    interactive: bool,
    unattended: str,
    pipeline: PipelineSpec | None,
    artifacts_dir: Path,
) -> StageResult:
    stage = stage.strip() or "stage"
    results: list[CmdResult] = []
    started = time.monotonic()

    for cmd_idx, raw_cmd in enumerate(cmds, start=1):
        cmd = raw_cmd.strip()
        if not cmd:
            continue

        for attempt in range(1, int(retries) + 2):
            if pipeline and pipeline.security_max_total_seconds:
                elapsed = time.monotonic() - started
                if elapsed > float(pipeline.security_max_total_seconds):
                    res = CmdResult(
                        cmd=cmd,
                        rc=124,
                        stdout="",
                        stderr=f"max_total_seconds_exceeded: {pipeline.security_max_total_seconds}",
                        timed_out=True,
                    )
                    results.append(res)
                    failed_index = len(results) - 1
                    _write_cmd_artifacts(artifacts_dir, f"{stage}_cmd{cmd_idx:02d}_try{attempt:02d}", res)
                    _write_cmd_artifacts(artifacts_dir, stage, res)
                    _write_json(
                        artifacts_dir / f"{stage}_summary.json",
                        {"ok": False, "failed_index": failed_index, "total_results": len(results)},
                    )
                    return StageResult(ok=False, results=results, failed_index=failed_index)

            allowed, reason = _cmd_allowed(cmd, pipeline=pipeline)
            if not allowed:
                res = CmdResult(cmd=cmd, rc=126, stdout="", stderr=reason or "blocked", timed_out=False)
            elif unattended == "strict" and _looks_interactive(cmd):
                res = CmdResult(
                    cmd=cmd,
                    rc=126,
                    stdout="",
                    stderr="likely_interactive_command_disallowed_in_strict_mode",
                    timed_out=False,
                )
            else:
                eff_timeout = timeout_seconds
                if pipeline and pipeline.security_max_cmd_seconds:
                    eff_timeout = (
                        int(pipeline.security_max_cmd_seconds)
                        if eff_timeout is None
                        else min(int(eff_timeout), int(pipeline.security_max_cmd_seconds))
                    )
                res = run_cmd_capture(
                    cmd,
                    workdir,
                    timeout_seconds=eff_timeout,
                    env=env,
                    interactive=bool(interactive and unattended == "guided"),
                )

            results.append(res)
            _write_cmd_artifacts(artifacts_dir, f"{stage}_cmd{cmd_idx:02d}_try{attempt:02d}", res)

            if res.rc == 0:
                break

        if results and results[-1].rc != 0:
            failed_index = len(results) - 1
            _write_cmd_artifacts(artifacts_dir, stage, results[-1])
            _write_json(
                artifacts_dir / f"{stage}_summary.json",
                {"ok": False, "failed_index": failed_index, "total_results": len(results)},
            )
            return StageResult(ok=False, results=results, failed_index=failed_index)

    if results:
        _write_cmd_artifacts(artifacts_dir, stage, results[-1])
    _write_json(
        artifacts_dir / f"{stage}_summary.json",
        {"ok": True, "failed_index": None, "total_results": len(results)},
    )
    return StageResult(ok=True, results=results, failed_index=None)


def run_pipeline_verification(
    repo: Path,
    *,
    pipeline: PipelineSpec | None,
    tests_cmds: list[str],
    artifacts_dir: Path,
    unattended: str = "strict",
) -> VerificationResult:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    ok = False
    failed_stage: str | None = None
    auth_res: StageResult | None = None
    tests_res: StageResult | None = None
    deploy_setup_res: StageResult | None = None
    deploy_health_res: StageResult | None = None
    bench_res: StageResult | None = None
    metrics_path: str | None = None
    metrics: dict[str, Any] | None = None
    metrics_errors: list[str] = []

    teardown_cmds = list(pipeline.deploy_teardown_cmds or []) if pipeline else []
    teardown_policy = (pipeline.deploy_teardown_policy if pipeline else "never").lower()
    kubectl_dump_enabled = bool(pipeline and pipeline.kubectl_dump_enabled)

    def _teardown_allowed(success: bool) -> bool:
        if not teardown_cmds:
            return False
        if teardown_policy == "never":
            return False
        if teardown_policy == "always":
            return True
        if teardown_policy == "on_success":
            return success
        if teardown_policy == "on_failure":
            return not success
        return False

    env_base = dict(os.environ)

    def _workdir_or_fail(stage: str, raw: str | None) -> tuple[Path, StageResult | None]:
        try:
            return _resolve_workdir(repo, raw), None
        except Exception as e:
            err = CmdResult(cmd=f"resolve_workdir {raw}", rc=2, stdout="", stderr=str(e), timed_out=False)
            _write_cmd_artifacts(artifacts_dir, f"{stage}_workdir_error", err)
            return repo, StageResult(ok=False, results=[err], failed_index=0)

    try:
        if pipeline and pipeline.auth_cmds:
            auth_env = _safe_env(env_base, pipeline.auth_env, unattended=unattended)
            auth_workdir, auth_wd_err = _workdir_or_fail("auth", pipeline.auth_workdir)
            if auth_wd_err is not None:
                failed_stage = "auth"
                return VerificationResult(
                    ok=False,
                    failed_stage=failed_stage,
                    auth=auth_wd_err,
                    metrics_errors=metrics_errors,
                )
            auth_res = _run_stage(
                repo,
                stage="auth",
                cmds=pipeline.auth_cmds,
                workdir=auth_workdir,
                env=auth_env,
                timeout_seconds=pipeline.auth_timeout_seconds,
                retries=pipeline.auth_retries,
                interactive=bool(pipeline.auth_interactive),
                unattended=unattended,
                pipeline=pipeline,
                artifacts_dir=artifacts_dir,
            )
            if not auth_res.ok:
                failed_stage = "auth"
                return VerificationResult(
                    ok=False,
                    failed_stage=failed_stage,
                    auth=auth_res,
                metrics_errors=metrics_errors,
            )

        tests_env = _safe_env(env_base, pipeline.tests_env if pipeline else {}, unattended=unattended)
        tests_workdir, tests_wd_err = _workdir_or_fail("tests", pipeline.tests_workdir if pipeline else None)
        if tests_wd_err is not None:
            failed_stage = "tests"
            return VerificationResult(
                ok=False,
                failed_stage=failed_stage,
                auth=auth_res,
                tests=tests_wd_err,
                metrics_errors=metrics_errors,
            )
        tests_timeout = pipeline.tests_timeout_seconds if pipeline else None
        tests_retries = pipeline.tests_retries if pipeline else 0
        tests_res = _run_stage(
            repo,
            stage="tests",
            cmds=tests_cmds,
            workdir=tests_workdir,
            env=tests_env,
            timeout_seconds=tests_timeout,
            retries=tests_retries,
            interactive=False,
            unattended=unattended,
            pipeline=pipeline,
            artifacts_dir=artifacts_dir,
        )
        if not tests_res.ok:
            failed_stage = "tests"
            return VerificationResult(
                ok=False,
                failed_stage=failed_stage,
                auth=auth_res,
                tests=tests_res,
                metrics_errors=metrics_errors,
            )

        if pipeline and pipeline.deploy_setup_cmds:
            deploy_env = _safe_env(env_base, pipeline.deploy_env, unattended=unattended)
            deploy_workdir, deploy_wd_err = _workdir_or_fail("deploy_setup", pipeline.deploy_workdir)
            if deploy_wd_err is not None:
                failed_stage = "deploy_setup"
                return VerificationResult(
                    ok=False,
                    failed_stage=failed_stage,
                    auth=auth_res,
                    tests=tests_res,
                    deploy_setup=deploy_wd_err,
                    metrics_errors=metrics_errors,
                )
            deploy_timeout = pipeline.deploy_timeout_seconds
            deploy_setup_res = _run_stage(
                repo,
                stage="deploy_setup",
                cmds=pipeline.deploy_setup_cmds,
                workdir=deploy_workdir,
                env=deploy_env,
                timeout_seconds=deploy_timeout,
                retries=pipeline.deploy_retries,
                interactive=False,
                unattended=unattended,
                pipeline=pipeline,
                artifacts_dir=artifacts_dir,
            )
            if not deploy_setup_res.ok:
                failed_stage = "deploy_setup"
                return VerificationResult(
                    ok=False,
                    failed_stage=failed_stage,
                    auth=auth_res,
                    tests=tests_res,
                    deploy_setup=deploy_setup_res,
                metrics_errors=metrics_errors,
            )

        if pipeline and pipeline.deploy_health_cmds:
            deploy_env = _safe_env(env_base, pipeline.deploy_env, unattended=unattended)
            deploy_workdir, deploy_wd_err = _workdir_or_fail("deploy_health", pipeline.deploy_workdir)
            if deploy_wd_err is not None:
                failed_stage = "deploy_health"
                return VerificationResult(
                    ok=False,
                    failed_stage=failed_stage,
                    auth=auth_res,
                    tests=tests_res,
                    deploy_setup=deploy_setup_res,
                    deploy_health=deploy_wd_err,
                    metrics_errors=metrics_errors,
                )
            deploy_timeout = pipeline.deploy_timeout_seconds
            deploy_health_res = _run_stage(
                repo,
                stage="deploy_health",
                cmds=pipeline.deploy_health_cmds,
                workdir=deploy_workdir,
                env=deploy_env,
                timeout_seconds=deploy_timeout,
                retries=pipeline.deploy_retries,
                interactive=False,
                unattended=unattended,
                pipeline=pipeline,
                artifacts_dir=artifacts_dir,
            )
            if not deploy_health_res.ok:
                failed_stage = "deploy_health"
                return VerificationResult(
                    ok=False,
                    failed_stage=failed_stage,
                    auth=auth_res,
                    tests=tests_res,
                    deploy_setup=deploy_setup_res,
                    deploy_health=deploy_health_res,
                metrics_errors=metrics_errors,
            )

        if pipeline and pipeline.benchmark_run_cmds:
            bench_env = _safe_env(env_base, pipeline.benchmark_env, unattended=unattended)
            bench_workdir, bench_wd_err = _workdir_or_fail("benchmark", pipeline.benchmark_workdir)
            if bench_wd_err is not None:
                failed_stage = "benchmark"
                return VerificationResult(
                    ok=False,
                    failed_stage=failed_stage,
                    auth=auth_res,
                    tests=tests_res,
                    deploy_setup=deploy_setup_res,
                    deploy_health=deploy_health_res,
                    benchmark=bench_wd_err,
                    metrics_errors=metrics_errors,
                )
            bench_timeout = pipeline.benchmark_timeout_seconds
            bench_res = _run_stage(
                repo,
                stage="benchmark",
                cmds=pipeline.benchmark_run_cmds,
                workdir=bench_workdir,
                env=bench_env,
                timeout_seconds=bench_timeout,
                retries=pipeline.benchmark_retries,
                interactive=False,
                unattended=unattended,
                pipeline=pipeline,
                artifacts_dir=artifacts_dir,
            )
            if not bench_res.ok:
                failed_stage = "benchmark"
                return VerificationResult(
                    ok=False,
                    failed_stage=failed_stage,
                    auth=auth_res,
                    tests=tests_res,
                    deploy_setup=deploy_setup_res,
                    deploy_health=deploy_health_res,
                    benchmark=bench_res,
                    metrics_errors=metrics_errors,
                )

            if pipeline.benchmark_metrics_path:
                mpath = Path(pipeline.benchmark_metrics_path).expanduser()
                if not mpath.is_absolute():
                    mpath = repo / mpath
                metrics_path = str(mpath)
                if not mpath.exists():
                    failed_stage = "metrics"
                    metrics_errors.append(f"metrics_file_missing: {mpath}")
                    return VerificationResult(
                        ok=False,
                        failed_stage=failed_stage,
                        auth=auth_res,
                        tests=tests_res,
                        deploy_setup=deploy_setup_res,
                        deploy_health=deploy_health_res,
                        benchmark=bench_res,
                        metrics_path=metrics_path,
                        metrics_errors=metrics_errors,
                    )

                # Always snapshot the produced metrics file for reproducibility, even if it is invalid JSON.
                _write_text(artifacts_dir / "metrics.json", _read_text_if_exists(mpath))

                metrics, err = _read_json(mpath)
                if err:
                    failed_stage = "metrics"
                    metrics_errors.append(err)
                    return VerificationResult(
                        ok=False,
                        failed_stage=failed_stage,
                        auth=auth_res,
                        tests=tests_res,
                        deploy_setup=deploy_setup_res,
                        deploy_health=deploy_health_res,
                        benchmark=bench_res,
                        metrics_path=metrics_path,
                        metrics=metrics,
                        metrics_errors=metrics_errors,
                    )

                missing = _validate_metrics(metrics or {}, pipeline.benchmark_required_keys)
                if missing:
                    failed_stage = "metrics"
                    metrics_errors.append("missing_keys: " + ", ".join(missing))
                    return VerificationResult(
                        ok=False,
                        failed_stage=failed_stage,
                        auth=auth_res,
                        tests=tests_res,
                        deploy_setup=deploy_setup_res,
                        deploy_health=deploy_health_res,
                        benchmark=bench_res,
                        metrics_path=metrics_path,
                        metrics=metrics,
                        metrics_errors=metrics_errors,
                    )

        ok = True
        return VerificationResult(
            ok=True,
            failed_stage=None,
            auth=auth_res,
            tests=tests_res,
            deploy_setup=deploy_setup_res,
            deploy_health=deploy_health_res,
            benchmark=bench_res,
            metrics_path=metrics_path,
            metrics=metrics,
            metrics_errors=metrics_errors,
        )
    finally:
        if kubectl_dump_enabled:
            _dump_kubectl(
                artifacts_dir / "kubectl",
                repo,
                namespace=(pipeline.kubectl_dump_namespace if pipeline else None),
                label_selector=(pipeline.kubectl_dump_label_selector if pipeline else None),
                include_logs=bool(pipeline and pipeline.kubectl_dump_include_logs),
            )

        if pipeline and _teardown_allowed(ok):
            deploy_env = _safe_env(env_base, pipeline.deploy_env, unattended=unattended)
            deploy_workdir, td_wd_err = _workdir_or_fail("deploy_teardown", pipeline.deploy_workdir)
            if td_wd_err is not None:
                _write_text(artifacts_dir / "deploy_teardown_warning.txt", "skip teardown due to invalid workdir\n")
            else:
                td = _run_stage(
                    repo,
                    stage="deploy_teardown",
                    cmds=teardown_cmds,
                    workdir=deploy_workdir,
                    env=deploy_env,
                    timeout_seconds=pipeline.deploy_timeout_seconds,
                    retries=0,
                    interactive=False,
                    unattended=unattended,
                    pipeline=pipeline,
                    artifacts_dir=artifacts_dir,
                )
                # keep the most recent command output for backwards-compatible filenames
                if td.results:
                    _write_cmd_artifacts(artifacts_dir, "deploy_teardown", td.results[-1])


def make_plan_update_prompt(snapshot_text: str, test_cmd: str, *, extra: str = "") -> str:
    extra = extra.strip()
    extra_block = f"\n[EXTRA]\n{extra}\n" if extra else ""
    return (
        "你是一个严格的计划编辑器。你的任务：只修改 PLAN.md，把计划变成机器可解析且可执行。\n"
        "\n"
        "硬约束：\n"
        "1) 只能修改 PLAN.md，禁止修改任何其他文件。\n"
        "2) `## Next (exactly ONE item)` 下面必须恰好 1 条未完成任务：`- [ ] (STEP_ID=NNN) ...`\n"
        "3) 所有 step 必须是最小原子：一次编辑 + 一次验收即可闭合。\n"
        "4) `## Acceptance` 里必须包含：`- [ ] TEST_CMD passes: `{TEST_CMD}``\n"
        "5) 不确定点写入 `## Notes`；需要人类信息则标记 Blocked（写清楚需要什么）。\n"
        "\n"
        f"TEST_CMD: {test_cmd}\n"
        "\n"
        f"{snapshot_text}"
        f"{extra_block}"
        "\n"
        "现在只改 PLAN.md。\n"
    )


def make_execute_prompt(snapshot_text: str, step: dict[str, str]) -> str:
    return (
        "你是一个严格的执行器。你的任务：只实现 `Next` 这一个 step。\n"
        "\n"
        "硬约束：\n"
        "1) 只能做这一件事，不要顺手重构，不要做额外功能。\n"
        "2) 改动越小越好。\n"
        "3) 禁止修改 PLAN.md（如果发现需要改计划，请停止改代码）。\n"
        "4) 禁止修改 pipeline 文件（部署/评测契约由人类提供，runner 会拒绝被修改）。\n"
        "\n"
        f"NEXT_STEP: (STEP_ID={step['id']}) {step['text']}\n"
        "\n"
        f"{snapshot_text}\n"
    )


def _fmt_stage_tail(prefix: str, stage: StageResult | None) -> str:
    res = _stage_failed_cmd(stage)
    if res is None:
        return ""
    return (
        f"[{prefix}_RC]\n{res.rc}\n\n"
        f"[{prefix}_STDOUT_TAIL]\n{_tail(res.stdout, STDIO_TAIL_CHARS)}\n\n"
        f"[{prefix}_STDERR_TAIL]\n{_tail(res.stderr, STDIO_TAIL_CHARS)}\n\n"
    )


def make_fix_or_replan_prompt(
    step: dict[str, str],
    verify: VerificationResult,
    *,
    tests_cmds: list[str],
    artifacts_dir: Path,
) -> str:
    metrics_errors = verify.metrics_errors or []
    metrics_block = ""
    if verify.metrics_path or metrics_errors:
        metrics_block = (
            "[METRICS]\n"
            f"metrics_path: {verify.metrics_path}\n"
            f"errors: {metrics_errors}\n"
            f"metrics_preview: {_tail(json.dumps(verify.metrics or {}, ensure_ascii=False), 2000)}\n"
            "\n"
        )
    return (
        "验收失败了。你必须二选一：\n"
        "A) 修复代码/部署清单/脚本，直到所有验收通过（优先）。\n"
        "B) 如果确实缺信息/不可闭合：只修改 PLAN.md，把该 step 拆小或标记 Blocked，并写清楚需要的人类输入。\n"
        "\n"
        f"FAILED_STEP: (STEP_ID={step['id']}) {step['text']}\n"
        "\n"
        f"FAILED_STAGE: {verify.failed_stage}\n"
        f"TEST_CMDS: {' && '.join(tests_cmds)}\n"
        f"ARTIFACTS_DIR: {artifacts_dir}\n"
        "\n"
        "如果这是环境/工具链/认证问题，你可以写入 `.aider_fsm/actions.yml` 请求 runner 执行系统修复动作。\n"
        "actions.yml 格式（YAML）：\n"
        "version: 1\n"
        "actions:\n"
        "- id: fix-001\n"
        "  kind: run_cmd\n"
        "  cmd: <shell command>\n"
        "  timeout_seconds: 300\n"
        "  retries: 0\n"
        "  risk_level: low|medium|high\n"
        "  rationale: <why>\n"
        "注意：\n"
        "- 在严格无人值守模式下避免交互式登录命令（例如 `docker login` 需要改成非交互用法）。\n"
        "- runner 会记录执行结果到 artifacts 并清空 actions.yml。\n"
        "\n"
        f"{_fmt_stage_tail('AUTH', verify.auth)}"
        f"{_fmt_stage_tail('TESTS', verify.tests)}"
        f"{_fmt_stage_tail('DEPLOY_SETUP', verify.deploy_setup)}"
        f"{_fmt_stage_tail('DEPLOY_HEALTH', verify.deploy_health)}"
        f"{_fmt_stage_tail('BENCHMARK', verify.benchmark)}"
        f"{metrics_block}"
    )


def make_mark_done_prompt(step: dict[str, str]) -> str:
    return (
        "当前 step 已通过验收。只修改 PLAN.md：\n"
        f"1) 把 `- [ ] (STEP_ID={step['id']}) ...` 从 Next 移到 Done，并改成 `- [x]`。\n"
        "2) 从 Backlog 里挑选一条最小原子任务放入 Next（保证 Next 仍然恰好 1 条）。\n"
        "3) 如果 Backlog 为空，则让 Next 为空（Next 区块保留标题但无条目）。\n"
    )


def make_block_step_prompt(step: dict[str, str], last_failure: str) -> str:
    return (
        "单步修复次数已超限。只修改 PLAN.md：\n"
        f"1) 把该 step 从 Next 移除，并在 Notes 写明 Blocked 原因与需要的人类信息。\n"
        "2) 从 Backlog 里挑选一条放入 Next（若无则 Next 置空）。\n"
        "\n"
        f"BLOCKED_STEP: (STEP_ID={step['id']}) {step['text']}\n"
        "\n"
        "[LAST_FAILURE]\n"
        f"{_tail(last_failure, STDIO_TAIL_CHARS)}\n"
    )


def _coder_run(coder, text: str, *, log_path: Path, iter_idx: int, fsm_state: str, event: str) -> None:
    append_jsonl(
        log_path,
        {
            "ts": _now_iso(),
            "iter_idx": iter_idx,
            "fsm_state": fsm_state,
            "event": event,
            "prompt_preview": _tail(text, 1200),
        },
    )
    coder.run(text)


def main() -> int:
    parser = argparse.ArgumentParser(description="Aider-FSM runner (minimal, single-process loop)")
    parser.add_argument("--repo", default=".", help="repo root path (default: .)")
    parser.add_argument("--goal", default="", help="goal for PLAN.md (used only when PLAN.md is missing)")
    parser.add_argument("--model", default="gpt-4o-mini", help="model name (default: gpt-4o-mini)")
    parser.add_argument("--test-cmd", default=None, help='acceptance command (default: pipeline/tests or "pytest -q")')
    parser.add_argument("--pipeline", default="", help="pipeline YAML path relative to repo (optional)")
    parser.add_argument(
        "--artifacts-dir",
        default="",
        help="artifacts base directory (default: pipeline.artifacts.out_dir or .aider_fsm/artifacts under repo)",
    )
    parser.add_argument("--seed", action="append", default=[], help="seed file path (repeatable)")
    parser.add_argument("--max-iters", type=int, default=200, help="max iterations (default: 200)")
    parser.add_argument("--max-fix", type=int, default=10, help="max fix attempts per step (default: 10)")
    parser.add_argument("--plan-path", default="PLAN.md", help="plan file path relative to repo (default: PLAN.md)")
    parser.add_argument("--ensure-tools", action="store_true", help="auto-install/verify docker+k8s tools on macOS")
    parser.add_argument("--ensure-kind", action="store_true", help="ensure a kind cluster exists before running (generic)")
    parser.add_argument("--kind-name", default="", help="kind cluster name (default: pipeline.tooling.kind_cluster_name or kind)")
    parser.add_argument("--kind-config", default="", help="kind config YAML path relative to repo (optional)")
    parser.add_argument(
        "--unattended",
        choices=("strict", "guided"),
        default="strict",
        help="unattended mode: strict blocks likely-interactive commands; guided allows interactive auth steps",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="run setup/checks then exit; does not invoke aider FSM loop",
    )
    parser.add_argument(
        "--full-quickstart",
        action="store_true",
        help="run AIOpsLab local kind Quick Start preflight (creates cluster/config/venv)",
    )
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    plan_rel = args.plan_path
    plan_abs = (repo / plan_rel).resolve()

    pipeline_abs: Path | None = None
    pipeline_rel: str | None = None
    pipeline: PipelineSpec | None = None
    if str(args.pipeline or "").strip():
        pipeline_abs = _resolve_config_path(repo, str(args.pipeline).strip())
        pipeline = load_pipeline_spec(pipeline_abs)
        pipeline_rel = _relpath_or_none(pipeline_abs, repo)

    tests_cmds: list[str] = []
    if args.test_cmd and str(args.test_cmd).strip():
        tests_cmds = [str(args.test_cmd).strip()]
    elif pipeline and pipeline.tests_cmds:
        tests_cmds = list(pipeline.tests_cmds)
    else:
        tests_cmds = ["pytest -q"]

    effective_test_cmd = " && ".join(tests_cmds)

    ensure_tools = bool(args.ensure_tools or (pipeline.tooling_ensure_tools if pipeline else False))
    ensure_kind = bool(args.ensure_kind or (pipeline.tooling_ensure_kind_cluster if pipeline else False))
    kind_name = (str(args.kind_name).strip() if args.kind_name else "") or (
        pipeline.tooling_kind_cluster_name if pipeline else "kind"
    )
    kind_config_raw = (str(args.kind_config).strip() if args.kind_config else "") or (
        (pipeline.tooling_kind_config or "").strip() if pipeline else ""
    )
    kind_config: Path | None = _resolve_config_path(repo, kind_config_raw) if kind_config_raw else None

    state_dir, logs_dir, state_path = _ensure_dirs(repo)
    run_id = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_path = logs_dir / f"run_{run_id}.jsonl"

    artifacts_base_raw = str(args.artifacts_dir or "").strip()
    if not artifacts_base_raw and pipeline:
        artifacts_base_raw = (pipeline.artifacts_out_dir or "").strip()
    artifacts_base = _resolve_config_path(repo, artifacts_base_raw) if artifacts_base_raw else (state_dir / "artifacts")
    artifacts_run_dir = artifacts_base / run_id
    artifacts_run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(artifacts_run_dir / "versions.json", _probe_versions(repo))
    if pipeline_abs and pipeline_abs.exists():
        _write_text(artifacts_run_dir / "pipeline.yml", _read_text_if_exists(pipeline_abs))

    if ensure_tools:
        bootstrap = Path(__file__).resolve().parent / "tools" / "bootstrap_macos.py"
        if not bootstrap.exists():
            raise RuntimeError("Missing tools/bootstrap_macos.py")

        rc, out, err = _run_py_script(bootstrap, [], cwd=repo)
        append_jsonl(
            log_path,
            {
                "ts": _now_iso(),
                "iter_idx": 0,
                "fsm_state": "S0_BOOTSTRAP",
                "event": "ensure_tools",
                "rc": rc,
                "stdout_tail": out,
                "stderr_tail": err,
            },
        )
        if rc != 0:
            if out:
                print(out, file=sys.stderr)
            if err:
                print(err, file=sys.stderr)
            return 2

    if ensure_kind:
        _ensure_kind_cluster_generic(repo, name=kind_name, config=kind_config)

    if args.full_quickstart:
        _ensure_kind_cluster(repo)
        _ensure_aiopslab_config(repo)
        _ensure_aiopslab_venv(repo)
        pipeline = None
        pipeline_abs = None
        pipeline_rel = None
        effective_test_cmd = _default_full_quickstart_test_cmd()
        tests_cmds = [effective_test_cmd]

    if args.preflight_only:
        preflight_dir = artifacts_run_dir / "preflight"
        verify = run_pipeline_verification(
            repo,
            pipeline=pipeline,
            tests_cmds=tests_cmds,
            artifacts_dir=preflight_dir,
            unattended=str(args.unattended),
        )
        _write_json(
            preflight_dir / "summary.json",
            {
                "ok": verify.ok,
                "failed_stage": verify.failed_stage,
                "auth_rc": _stage_rc(verify.auth),
                "test_rc": _stage_rc(verify.tests),
                "deploy_setup_rc": _stage_rc(verify.deploy_setup),
                "deploy_health_rc": _stage_rc(verify.deploy_health),
                "benchmark_rc": _stage_rc(verify.benchmark),
                "metrics_path": verify.metrics_path,
                "metrics_errors": verify.metrics_errors or [],
            },
        )
        append_jsonl(
            log_path,
            {
                "ts": _now_iso(),
                "iter_idx": 0,
                "fsm_state": "PREFLIGHT",
                "event": "preflight_verify",
                "ok": verify.ok,
                "failed_stage": verify.failed_stage,
            },
        )
        return 0 if verify.ok else 3

    ensure_plan_file(plan_abs, args.goal, effective_test_cmd, pipeline=pipeline)
    os.chdir(repo)

    defaults = default_state(
        repo=repo,
        plan_rel=plan_rel,
        pipeline_rel=pipeline_rel,
        model=args.model,
        test_cmd=effective_test_cmd,
    )
    state = load_state(state_path, defaults)
    save_state(state_path, state)

    fnames = [plan_rel]
    if pipeline_rel:
        fnames.append(pipeline_rel)
    fnames.extend(list(args.seed))
    coder = create_coder(model_name=args.model, fnames=fnames)

    for iter_idx in range(1, args.max_iters + 1):
        state["iter_idx"] = iter_idx
        state["fsm_state"] = "S1_SNAPSHOT"
        save_state(state_path, state)

        snapshot, snapshot_text = build_snapshot(repo, plan_abs, pipeline_abs)
        append_jsonl(
            log_path,
            {
                "ts": _now_iso(),
                "iter_idx": iter_idx,
                "fsm_state": state["fsm_state"],
                "event": "snapshot",
                "snapshot": {k: snapshot[k] for k in ("repo", "plan_path", "env_probe", "git")},
            },
        )

        # S2: plan update (only PLAN.md)
        state["fsm_state"] = "S2_PLAN_UPDATE"
        save_state(state_path, state)
        plan_text: str | None = None
        parsed: dict[str, Any] | None = None
        next_step: dict[str, str] | None = None
        backlog_open = 0

        for attempt in range(1, 4):
            extra = ""
            if attempt > 1:
                extra = "上一次计划更新越界或格式错误：请严格只改 PLAN.md，并确保 Next 恰好 1 条。"

            pipeline_before = _read_text_if_exists(pipeline_abs) if pipeline_abs else ""
            _coder_run(
                coder,
                make_plan_update_prompt(snapshot_text, effective_test_cmd, extra=extra),
                log_path=log_path,
                iter_idx=iter_idx,
                fsm_state=state["fsm_state"],
                event=f"plan_update_attempt_{attempt}",
            )
            if pipeline_abs and _read_text_if_exists(pipeline_abs) != pipeline_before:
                _write_text(pipeline_abs, pipeline_before)
                append_jsonl(
                    log_path,
                    {
                        "ts": _now_iso(),
                        "iter_idx": iter_idx,
                        "fsm_state": state["fsm_state"],
                        "event": "plan_update_touched_pipeline_reverted",
                        "pipeline_path": str(pipeline_abs),
                    },
                )

            changed = get_git_changed_files(repo)
            if changed is not None:
                illegal = non_plan_changes(changed, plan_rel)
                if illegal:
                    rc, out, err = git_checkout(repo, illegal)
                    append_jsonl(
                        log_path,
                        {
                            "ts": _now_iso(),
                            "iter_idx": iter_idx,
                            "fsm_state": state["fsm_state"],
                            "event": "plan_update_revert_non_plan_files",
                            "illegal_files": illegal,
                            "git_checkout_rc": rc,
                            "git_checkout_out": out,
                            "git_checkout_err": err,
                        },
                    )
                    if attempt == 3:
                        state["last_exit_reason"] = "PLAN_UPDATE_TOUCHED_CODE"
                        state["updated_at"] = _now_iso()
                        save_state(state_path, state)
                        return 2
                    continue

            plan_text = plan_abs.read_text(encoding="utf-8", errors="replace")
            parsed = parse_plan(plan_text)

            if parsed["errors"]:
                append_jsonl(
                    log_path,
                    {
                        "ts": _now_iso(),
                        "iter_idx": iter_idx,
                        "fsm_state": state["fsm_state"],
                        "event": "plan_parse_error",
                        "errors": parsed["errors"],
                    },
                )
                if attempt == 3:
                    state["last_exit_reason"] = "PLAN_PARSE_ERROR"
                    state["updated_at"] = _now_iso()
                    save_state(state_path, state)
                    return 2
                continue

            next_step = parsed["next_step"]
            backlog_open = int(parsed["backlog_open_count"])
            if next_step is None and backlog_open > 0:
                append_jsonl(
                    log_path,
                    {
                        "ts": _now_iso(),
                        "iter_idx": iter_idx,
                        "fsm_state": state["fsm_state"],
                        "event": "plan_inconsistent_missing_next",
                        "backlog_open_count": backlog_open,
                    },
                )
                if attempt == 3:
                    state["last_exit_reason"] = "MISSING_NEXT_STEP"
                    state["updated_at"] = _now_iso()
                    save_state(state_path, state)
                    return 2
                continue

            break

        if next_step is None:
            # no next, no backlog: verify once and decide done/needs plan
            state["fsm_state"] = "S4_VERIFY"
            save_state(state_path, state)
            _coder_run(
                coder,
                f"/test {effective_test_cmd}",
                log_path=log_path,
                iter_idx=iter_idx,
                fsm_state=state["fsm_state"],
                event="aider_test",
            )
            iter_artifacts_dir = artifacts_run_dir / f"iter_{iter_idx:04d}"
            verify = run_pipeline_verification(
                repo,
                pipeline=pipeline,
                tests_cmds=tests_cmds,
                artifacts_dir=iter_artifacts_dir,
                unattended=str(args.unattended),
            )
            state["last_test_rc"] = _stage_rc(verify.tests)
            state["last_deploy_setup_rc"] = _stage_rc(verify.deploy_setup)
            state["last_deploy_health_rc"] = _stage_rc(verify.deploy_health)
            state["last_benchmark_rc"] = _stage_rc(verify.benchmark)
            state["last_metrics_ok"] = bool(verify.ok or verify.failed_stage != "metrics")
            save_state(state_path, state)
            append_jsonl(
                log_path,
                {
                    "ts": _now_iso(),
                    "iter_idx": iter_idx,
                    "fsm_state": state["fsm_state"],
                    "event": "verify_pipeline_no_steps",
                    "ok": verify.ok,
                    "failed_stage": verify.failed_stage,
                    "artifacts_dir": str(iter_artifacts_dir),
                },
            )
            if verify.ok:
                state["last_exit_reason"] = "DONE"
                state["updated_at"] = _now_iso()
                save_state(state_path, state)
                return 0

            # failing but no tasks -> ask plan to add next step for fixing tests
            snapshot2, snapshot_text2 = build_snapshot(repo, plan_abs, pipeline_abs)
            _coder_run(
                coder,
                make_plan_update_prompt(
                    snapshot_text2,
                    effective_test_cmd,
                    extra=(
                        "当前验收失败但 Next/Backlog 为空：请添加一个最小 Next 来修复。\n"
                        f"FAILED_STAGE={verify.failed_stage}\n"
                        f"ARTIFACTS_DIR={iter_artifacts_dir}"
                    ),
                ),
                log_path=log_path,
                iter_idx=iter_idx,
                fsm_state="S2_PLAN_UPDATE",
                event="plan_update_due_to_failing_tests_no_steps",
            )
            continue

        # S3: execute
        state["fsm_state"] = "S3_EXECUTE_STEP"
        state["current_step_id"] = next_step["id"]
        state["current_step_text"] = next_step["text"]
        save_state(state_path, state)

        plan_before = plan_text
        pipeline_before = _read_text_if_exists(pipeline_abs) if pipeline_abs else ""
        _coder_run(
            coder,
            make_execute_prompt(snapshot_text, next_step),
            log_path=log_path,
            iter_idx=iter_idx,
            fsm_state=state["fsm_state"],
            event="execute_step",
        )

        plan_after = plan_abs.read_text(encoding="utf-8", errors="replace") if plan_abs.exists() else ""
        if plan_after != plan_before:
            # guard: execution must not change PLAN.md; revert using content (works even without git)
            plan_abs.write_text(plan_before, encoding="utf-8")
            append_jsonl(
                log_path,
                {
                    "ts": _now_iso(),
                    "iter_idx": iter_idx,
                    "fsm_state": state["fsm_state"],
                    "event": "execute_touched_plan_reverted",
                },
            )
            continue

        if pipeline_abs and _read_text_if_exists(pipeline_abs) != pipeline_before:
            _write_text(pipeline_abs, pipeline_before)
            append_jsonl(
                log_path,
                {
                    "ts": _now_iso(),
                    "iter_idx": iter_idx,
                    "fsm_state": state["fsm_state"],
                    "event": "execute_touched_pipeline_reverted",
                    "pipeline_path": str(pipeline_abs),
                },
            )
            continue

        # S4: verify
        state["fsm_state"] = "S4_VERIFY"
        save_state(state_path, state)

        _coder_run(
            coder,
            f"/test {effective_test_cmd}",
            log_path=log_path,
            iter_idx=iter_idx,
            fsm_state=state["fsm_state"],
            event="aider_test",
        )
        iter_artifacts_dir = artifacts_run_dir / f"iter_{iter_idx:04d}"
        verify = run_pipeline_verification(
            repo,
            pipeline=pipeline,
            tests_cmds=tests_cmds,
            artifacts_dir=iter_artifacts_dir,
            unattended=str(args.unattended),
        )
        _write_json(
            iter_artifacts_dir / "summary.json",
            {
                "ok": verify.ok,
                "failed_stage": verify.failed_stage,
                "auth_rc": _stage_rc(verify.auth),
                "test_rc": _stage_rc(verify.tests),
                "deploy_setup_rc": _stage_rc(verify.deploy_setup),
                "deploy_health_rc": _stage_rc(verify.deploy_health),
                "benchmark_rc": _stage_rc(verify.benchmark),
                "metrics_path": verify.metrics_path,
                "metrics_errors": verify.metrics_errors or [],
            },
        )
        state["last_test_rc"] = _stage_rc(verify.tests)
        state["last_deploy_setup_rc"] = _stage_rc(verify.deploy_setup)
        state["last_deploy_health_rc"] = _stage_rc(verify.deploy_health)
        state["last_benchmark_rc"] = _stage_rc(verify.benchmark)
        state["last_metrics_ok"] = bool(verify.ok or verify.failed_stage != "metrics")
        save_state(state_path, state)
        append_jsonl(
            log_path,
            {
                "ts": _now_iso(),
                "iter_idx": iter_idx,
                "fsm_state": state["fsm_state"],
                "event": "verify_pipeline",
                "ok": verify.ok,
                "failed_stage": verify.failed_stage,
                "artifacts_dir": str(iter_artifacts_dir),
            },
        )

        # S5: decide
        state["fsm_state"] = "S5_DECIDE"
        save_state(state_path, state)

        if verify.ok:
            state["fix_attempts"] = 0
            save_state(state_path, state)
            _coder_run(
                coder,
                make_mark_done_prompt(next_step),
                log_path=log_path,
                iter_idx=iter_idx,
                fsm_state=state["fsm_state"],
                event="mark_done",
            )
            continue

        state["fix_attempts"] = int(state.get("fix_attempts") or 0) + 1
        save_state(state_path, state)
        if int(state["fix_attempts"]) <= int(args.max_fix):
            _coder_run(
                coder,
                make_fix_or_replan_prompt(
                    next_step,
                    verify,
                    tests_cmds=tests_cmds,
                    artifacts_dir=iter_artifacts_dir,
                ),
                log_path=log_path,
                iter_idx=iter_idx,
                fsm_state=state["fsm_state"],
                event=f"fix_or_replan_attempt_{state['fix_attempts']}",
            )
            actions_stage = run_pending_actions(
                repo,
                pipeline=pipeline,
                unattended=str(args.unattended),
                actions_path=repo / ".aider_fsm" / "actions.yml",
                artifacts_dir=iter_artifacts_dir / "actions",
                protected_paths=[p for p in (plan_abs, pipeline_abs) if p],
            )
            if actions_stage is not None:
                append_jsonl(
                    log_path,
                    {
                        "ts": _now_iso(),
                        "iter_idx": iter_idx,
                        "fsm_state": state["fsm_state"],
                        "event": "actions_executed",
                        "ok": actions_stage.ok,
                        "actions_rc": _stage_rc(actions_stage),
                    },
                )
        else:
            state["fix_attempts"] = 0
            save_state(state_path, state)
            last_failure = (
                f"FAILED_STAGE={verify.failed_stage}\n"
                f"ARTIFACTS_DIR={iter_artifacts_dir}\n\n"
                + _fmt_stage_tail("AUTH", verify.auth)
                + _fmt_stage_tail("TESTS", verify.tests)
                + _fmt_stage_tail("DEPLOY_SETUP", verify.deploy_setup)
                + _fmt_stage_tail("DEPLOY_HEALTH", verify.deploy_health)
                + _fmt_stage_tail("BENCHMARK", verify.benchmark)
                + (f"METRICS_ERRORS={verify.metrics_errors}\n" if (verify.metrics_errors or []) else "")
            )
            _coder_run(
                coder,
                make_block_step_prompt(next_step, last_failure),
                log_path=log_path,
                iter_idx=iter_idx,
                fsm_state=state["fsm_state"],
                event="block_step",
            )

    state["last_exit_reason"] = "MAX_ITERS"
    state["updated_at"] = _now_iso()
    save_state(state_path, state)
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
