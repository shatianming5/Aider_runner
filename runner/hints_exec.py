from __future__ import annotations

import configparser
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .security import cmd_allowed
from ._util import _is_truthy, _parse_json_str_list


_FLAG_VALUE_RE = re.compile(r"(?P<flag>--[A-Za-z0-9_.-]+)\s+(?P<val>(?:\"[^\"]*\"|'[^']*'|\S+))")


def _replace_flag_value(cmd: str, *, flag: str, new_value: str) -> str:
    # 作用：内部符号：_replace_flag_value
    # 能否简略：部分
    # 原因：规模≈17 行；引用次数≈4（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
    # 证据：位置=runner/hints_exec.py:67；类型=function；引用≈4；规模≈17行
    flag = flag.strip()
    if not flag:
        return cmd
    if not new_value:
        return cmd
    v = new_value
    # Quote the value if it has whitespace.
    if any(ch.isspace() for ch in v):
        v = json.dumps(v)
    return _FLAG_VALUE_RE.sub(
        lambda m: (m.group(0) if m.group("flag") != flag else f"{flag} {v}"),
        cmd,
    )


_BRACKET_GROUP_RE = re.compile(r"\[([^\]]+)\]")
_ANGLE_GROUP_RE = re.compile(r"<[^>]+>")
_GHA_EXPR_RE = re.compile(r"\$\{\{\s*([^}]+)\s*\}\}")
_PIPE_TO_BASH_RE = re.compile(r"(?i)\b(?:curl|wget)\b[^\n]*\|[^\n]*\bbash\b")
_DOTTED_MODULE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z0-9_.]+$")
_DOCKER_LINE_RE = re.compile(r"(?im)^\s*docker\s+")
_ENV_ASSIGN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_SHELL_BUILTINS = {
    ":",
    ".",
    "alias",
    "bg",
    "break",
    "builtin",
    "cd",
    "command",
    "continue",
    "dirs",
    "echo",
    "eval",
    "exec",
    "exit",
    "export",
    "false",
    "fg",
    "hash",
    "help",
    "history",
    "jobs",
    "kill",
    "local",
    "popd",
    "printf",
    "pushd",
    "pwd",
    "read",
    "readonly",
    "return",
    "set",
    "shift",
    "source",
    "test",
    "times",
    "trap",
    "true",
    "type",
    "typeset",
    "ulimit",
    "umask",
    "unalias",
    "unset",
    "wait",
}


_PY_MAJOR_MINOR_RE = re.compile(r"(?P<major>\d+)\.(?P<minor>\d+)")


def _as_major_minor(raw: str | None) -> str:
    # 作用：内部符号：_as_major_minor
    # 能否简略：是
    # 原因：规模≈14 行；引用次数≈5（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
    # 证据：位置=runner/hints_exec.py；类型=function；引用≈5；规模≈14行
    s = str(raw or "").strip()
    if not s:
        return ""
    m = _PY_MAJOR_MINOR_RE.search(s)
    if not m:
        return ""
    try:
        major = int(m.group("major"))
        minor = int(m.group("minor"))
    except Exception:
        return ""
    if major <= 0 or minor < 0:
        return ""
    return f"{major}.{minor}"


def _infer_repo_python_pin(repo: Path) -> str:
    """Infer a repo's preferred Python major.minor from common version pin files."""
    # 作用：Infer a repo's preferred Python major.minor from common version pin files.
    # 能否简略：部分
    # 原因：多来源探测（.python-version/.tool-versions）；属于兼容性策略的一部分；规模≈33 行；引用次数≈1（静态近似，可能包含注释/字符串）
    # 证据：位置=runner/hints_exec.py；类型=function；引用≈1；规模≈33行
    repo = Path(repo).resolve()
    for rel in (".python-version", "runtime.txt"):
        p = (repo / rel).resolve()
        try:
            if not p.exists() or not p.is_file():
                continue
            for raw in p.read_text(encoding="utf-8", errors="replace").splitlines():
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                mm = _as_major_minor(line)
                if mm:
                    return mm
                break
        except Exception:
            continue

    p = (repo / ".tool-versions").resolve()
    try:
        if p.exists() and p.is_file():
            for raw in p.read_text(encoding="utf-8", errors="replace").splitlines():
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if not line.startswith("python"):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    mm = _as_major_minor(parts[1])
                    if mm:
                        return mm
    except Exception:
        pass
    return ""


def _infer_repo_requires_python(repo: Path) -> str:
    """Infer a repo's `requires-python` style spec (best-effort)."""
    # 作用：Infer a repo's `requires-python` style spec (best-effort).
    # 能否简略：部分
    # 原因：覆盖 pyproject/setup.cfg/setup.py 三类常见声明；用于选择可用解释器；规模≈63 行；引用次数≈1（静态近似，可能包含注释/字符串）
    # 证据：位置=runner/hints_exec.py；类型=function；引用≈1；规模≈63行
    repo = Path(repo).resolve()
    pyproject = (repo / "pyproject.toml").resolve()
    if pyproject.exists() and pyproject.is_file():
        try:
            data = tomllib.loads(pyproject.read_text(encoding="utf-8", errors="replace"))
            if isinstance(data, dict):
                proj = data.get("project")
                if isinstance(proj, dict):
                    rp = proj.get("requires-python")
                    if isinstance(rp, str) and rp.strip():
                        return rp.strip()
                tool = data.get("tool")
                if isinstance(tool, dict):
                    poetry = tool.get("poetry")
                    if isinstance(poetry, dict):
                        deps = poetry.get("dependencies")
                        if isinstance(deps, dict):
                            py = deps.get("python")
                            if isinstance(py, str) and py.strip():
                                return py.strip()
        except Exception:
            pass

    setup_cfg = (repo / "setup.cfg").resolve()
    if setup_cfg.exists() and setup_cfg.is_file():
        try:
            cp = configparser.ConfigParser()
            cp.read(setup_cfg, encoding="utf-8")
            if cp.has_option("options", "python_requires"):
                v = str(cp.get("options", "python_requires") or "").strip()
                if v:
                    return v
        except Exception:
            pass

    setup_py = (repo / "setup.py").resolve()
    if setup_py.exists() and setup_py.is_file():
        try:
            text = setup_py.read_text(encoding="utf-8", errors="replace")
            m = re.search(r"(?i)python_requires\\s*=\\s*['\\\"]([^'\\\"]+)['\\\"]", text)
            if m:
                v = str(m.group(1) or "").strip()
                if v:
                    return v
        except Exception:
            pass
    return ""


def _best_python_minor_from_spec(spec: str, *, candidates: list[str]) -> str:
    """Pick the highest major.minor in candidates that satisfies a python spec string."""
    # 作用：Pick the highest major.minor in candidates that satisfies a python spec string.
    # 能否简略：部分
    # 原因：优先用 packaging 精确匹配；缺失时回退到启发式；用于 uv/venv 选择；规模≈46 行；引用次数≈1（静态近似，可能包含注释/字符串）
    # 证据：位置=runner/hints_exec.py；类型=function；引用≈1；规模≈46行
    s = str(spec or "").strip()
    if not s:
        return ""
    try:
        from packaging.specifiers import SpecifierSet  # type: ignore
        from packaging.version import Version  # type: ignore

        ss = SpecifierSet(s)
        for mm in candidates:
            try:
                if Version(f"{mm}.0") in ss:
                    return mm
            except Exception:
                continue
    except Exception:
        pass

    # Heuristic fallback when packaging isn't available or the spec isn't PEP 440 (e.g. Poetry's ^3.11).
    # Try explicit major.minor mentions first.
    for mm in candidates:
        if mm in s:
            return mm
    # Try simple upper-bound patterns like "<3.13" -> choose 3.12, etc.
    m = re.search(r"<\\s*(\\d+)\\.(\\d+)", s)
    if m:
        try:
            major = int(m.group(1))
            minor = int(m.group(2))
        except Exception:
            major = 0
            minor = 0
        if major > 0:
            want = f"{major}.{max(0, minor - 1)}"
            if want in candidates:
                return want
    return ""


def _infer_uv_python_candidates(repo: Path, *, env: dict[str, str]) -> list[str]:
    """Infer a list of uv `--python` requests to try (most preferred first)."""
    # 作用：Infer a list of uv `--python` requests to try (most preferred first).
    # 能否简略：否
    # 原因：把“显式配置 + repo 元数据 + 安全默认值”整合为统一策略；避免写死 py311；规模≈48 行；引用次数≈1（静态近似，可能包含注释/字符串）
    # 证据：位置=runner/hints_exec.py；类型=function；引用≈1；规模≈48行
    env2 = dict(env or {})
    out: list[str] = []

    raw_candidates = _parse_json_str_list(env2.get("AIDER_FSM_HINT_UV_PYTHON_CANDIDATES_JSON"))
    if raw_candidates:
        out.extend([c.strip() for c in raw_candidates if isinstance(c, str) and c.strip()])
    else:
        single = str(env2.get("AIDER_FSM_HINT_UV_PYTHON") or env2.get("UV_PYTHON") or "").strip()
        if single:
            out.append(single)

    if not out:
        pinned = _infer_repo_python_pin(repo)
        if pinned:
            out.append(pinned)

    if not out:
        spec = _infer_repo_requires_python(repo)
        if spec:
            # Prefer newer stable minors first when we need to choose.
            prefer = ["3.12", "3.11", "3.10", "3.9", "3.8"]
            picked = _best_python_minor_from_spec(spec, candidates=prefer)
            if picked:
                out.append(picked)
            else:
                # As a last resort, pick any explicit X.Y mention.
                mm = _as_major_minor(spec)
                if mm:
                    out.append(mm)

    if not out and sys.version_info >= (3, 13):
        # Safe defaults when running under very new Python versions: prefer a stable minor
        # with broad wheel availability.
        out.extend(["3.12", "3.11"])

    # Deduplicate while preserving order.
    seen: set[str] = set()
    cleaned: list[str] = []
    for v in out:
        s = str(v or "").strip()
        if not s or s in seen:
            continue
        seen.add(s)
        cleaned.append(s)
    return cleaned


def _canonical_base_url(url: str | None) -> str:
    # 作用：内部符号：_canonical_base_url
    # 能否简略：是
    # 原因：规模≈8 行；引用次数≈3（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
    # 证据：位置=runner/hints_exec.py:141；类型=function；引用≈3；规模≈8行
    s = str(url or "").strip()
    if not s:
        return ""
    s = s.rstrip("/")
    if s.endswith("/v1"):
        return s[: -len("/v1")]
    return s


def _first_command_line(cmd: str) -> str:
    # 作用：内部符号：_first_command_line
    # 能否简略：否
    # 原因：规模≈6 行；引用次数≈6（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=runner/hints_exec.py:151；类型=function；引用≈6；规模≈6行
    for raw in str(cmd or "").splitlines():
        line = raw.strip()
        if line:
            return line
    return ""


def _extract_cli_flag_value(cmd: str, flag: str) -> str:
    # 作用：内部符号：_extract_cli_flag_value
    # 能否简略：部分
    # 原因：规模≈17 行；引用次数≈4（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
    # 证据：位置=runner/hints_exec.py:159；类型=function；引用≈4；规模≈17行
    line = _first_command_line(cmd)
    if not line:
        return ""
    try:
        parts = shlex.split(line, posix=True)
    except Exception:
        return ""
    i = 0
    while i < len(parts):
        tok = str(parts[i] or "")
        if tok == flag and i + 1 < len(parts):
            return str(parts[i + 1] or "").strip()
        if tok.startswith(flag + "="):
            return str(tok.split("=", 1)[1] or "").strip()
        i += 1
    return ""


def _extract_cli_flag_value_any(cmd: str, flags: list[str]) -> str:
    # 作用：内部符号：_extract_cli_flag_value_any
    # 能否简略：部分
    # 原因：规模≈6 行；引用次数≈4（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
    # 证据：位置=runner/hints_exec.py:178；类型=function；引用≈4；规模≈6行
    for flag in list(flags or []):
        v = _extract_cli_flag_value(cmd, str(flag))
        if v:
            return v
    return ""


def _hint_backend(cmd: str) -> str:
    # 作用：内部符号：_hint_backend
    # 能否简略：是
    # 原因：规模≈9 行；引用次数≈3（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
    # 证据：位置=runner/hints_exec.py:186；类型=function；引用≈3；规模≈9行
    backend = _extract_cli_flag_value(cmd, "--backend").strip().lower()
    if backend:
        return backend
    line = _first_command_line(cmd).lower()
    # Generic heuristic: evaluator-style CLIs with model+dataset often default to OpenAI backend.
    if ".evaluate" in line and "--dataset" in line and "--model" in line:
        return "openai"
    return ""


def _is_remote_openai_hint(cmd: str) -> bool:
    # 作用：内部符号：_is_remote_openai_hint
    # 能否简略：是
    # 原因：规模≈2 行；引用次数≈3（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
    # 证据：位置=runner/hints_exec.py:197；类型=function；引用≈3；规模≈2行
    return _hint_backend(cmd) == "openai"


_SCORE_TOKEN_RE = re.compile(
    r"(?i)\b(?P<key>pass@1\+?|accuracy|score)\b[^0-9%]{0,12}(?P<val>\d+(?:\.\d+)?)(?P<pct>\s*%)?"
)


def _normalize_score(value: float, *, had_percent: bool) -> float | None:
    # 作用：内部符号：_normalize_score
    # 能否简略：是
    # 原因：规模≈10 行；引用次数≈3（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
    # 证据：位置=runner/hints_exec.py:219；类型=function；引用≈3；规模≈10行
    v = float(value)
    if had_percent:
        v = v / 100.0
    # Heuristic: if a tool prints accuracy like "75" (meaning 75%), normalize to [0,1].
    if v > 1.0 and v <= 100.0:
        v = v / 100.0
    if v < 0.0 or v > 1.0:
        return None
    return float(v)


def _extract_score_from_text(text: str) -> tuple[float | None, str]:
    """Best-effort score extraction from stdout/stderr (generic, benchmark-agnostic)."""
    # 作用：Best-effort score extraction from stdout/stderr (generic, benchmark-agnostic).
    # 能否简略：部分
    # 原因：规模≈22 行；引用次数≈2（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
    # 证据：位置=runner/hints_exec.py:232；类型=function；引用≈2；规模≈22行
    t = str(text or "")
    # Strip ANSI sequences so regexes match more reliably.
    t = re.sub(r"\x1b\[[0-9;]*m", "", t)
    last: dict[str, tuple[float, bool]] = {}
    for m in _SCORE_TOKEN_RE.finditer(t):
        key = str(m.group("key") or "").strip().lower()
        raw = str(m.group("val") or "").strip()
        had_pct = bool((m.group("pct") or "").strip())
        try:
            val = float(raw)
        except Exception:
            continue
        last[key] = (val, had_pct)
    for key in ("pass@1", "pass@1+", "accuracy", "score"):
        if key in last:
            val, had_pct = last[key]
            norm = _normalize_score(val, had_percent=had_pct)
            if norm is not None:
                return norm, f"text:{key}"
    return None, "no_score_in_text"


def _extract_score_from_json_obj(obj: object) -> tuple[float | None, str]:
    """Best-effort score extraction from JSON-like objects."""
    # 作用：Best-effort score extraction from JSON-like objects.
    # 能否简略：部分
    # 原因：规模≈25 行；引用次数≈2（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
    # 证据：位置=runner/hints_exec.py:256；类型=function；引用≈2；规模≈25行
    pairs: list[tuple[str, float]] = []
    # Manual DFS that preserves the original recursive order:
    # for each dict item (in order): emit numeric pair, then descend into the value.
    _item = object()
    stack: list[object] = [obj]
    while stack:
        x = stack.pop()
        if isinstance(x, tuple) and len(x) == 3 and x[0] is _item:
            _, k, v = x
            kk = str(k or "").strip().lower()
            if isinstance(v, (int, float)):
                pairs.append((kk, float(v)))
            stack.append(v)
            continue
        if isinstance(x, dict):
            for k, v in reversed(list(x.items())):
                stack.append((_item, k, v))
        elif isinstance(x, list):
            for it in reversed(x):
                stack.append(it)
    # Prefer pass@1, then accuracy, then score.
    for needle in ("pass@1", "pass_at_1", "pass@1+", "pass_at_1_plus", "accuracy", "score"):
        for k, v in reversed(pairs):
            if needle in k:
                norm = _normalize_score(v, had_percent=False)
                if norm is not None:
                    return norm, f"json:{needle}"
    return None, "no_score_in_json"


def _candidate_metrics_paths(cmd: str, *, repo: Path, workdir: Path | None = None) -> list[Path]:
    """Infer likely output paths for evaluation metrics from a hint command.

    NOTE: some hint commands are executed from an artifacts workdir (not the repo root)
    to avoid polluting the repo and to sidestep permission issues caused by docker-created
    root-owned output directories. For relative paths, prefer resolving against that
    execution workdir when provided.
    """
    # 作用：Infer likely output paths for evaluation metrics from a hint command.
    # 能否简略：部分
    # 原因：规模≈37 行；引用次数≈2（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
    # 证据：位置=runner/hints_exec.py:297；类型=function；引用≈2；规模≈37行
    repo = Path(repo).resolve()
    base = Path(workdir).resolve() if workdir is not None else repo
    out: list[Path] = []
    out_dir = _extract_cli_flag_value_any(
        cmd,
        [
            "--output-dir",
            "--output_dir",
            "--out-dir",
            "--out_dir",
            "--outdir",
            "--results-dir",
            "--results_dir",
        ],
    )
    if out_dir:
        p = Path(out_dir.strip())
        if not p.is_absolute():
            p = (base / p).resolve()
        for name in ("metrics.json", "results.json", "summary.json"):
            out.append((p / name).resolve())
    # Also allow commands that directly write a metrics file.
    out_path = _extract_cli_flag_value_any(cmd, ["--metrics", "--metrics-path", "--metrics_path"])
    if out_path:
        p = Path(out_path.strip())
        if not p.is_absolute():
            p = (base / p).resolve()
        out.append(p.resolve())
    return out


def _hint_runtime_compatible(*, cmd: str, env: dict[str, str], strict_compat: bool) -> tuple[bool, str]:
    # 作用：内部符号：_hint_runtime_compatible
    # 能否简略：是
    # 原因：规模≈19 行；引用次数≈2（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
    # 证据：位置=runner/hints_exec.py:329；类型=function；引用≈2；规模≈19行
    if not strict_compat:
        return True, "ok"
    low = _first_command_line(cmd).lower()
    if not low:
        return False, "empty"

    backend = _hint_backend(cmd)
    llm_kind = str(env.get("AIDER_LLM_KIND") or "").strip().lower()
    if llm_kind == "remote" and backend and backend != "openai":
        if "--samples" not in low:
            return False, f"backend_mismatch:{backend}"

    runtime_base = _canonical_base_url(env.get("OPENAI_BASE_URL") or env.get("OPENAI_API_BASE"))
    hinted_base = _canonical_base_url(_extract_cli_flag_value_any(cmd, ["--base-url", "--base_url"]))
    if runtime_base and hinted_base and runtime_base != hinted_base:
        return False, "base_url_mismatch"

    return True, "ok"


def normalize_hint_command(cmd: str, *, env: dict[str, str]) -> tuple[str, str | None]:
    """Normalize a doc-derived command hint into something runnable.

    Returns (sanitized_cmd, skip_reason). If skip_reason is not None, callers should skip it.
    """
    # 作用：Normalize a doc-derived command hint into something runnable.
    # 能否简略：部分
    # 原因：规模≈184 行；引用次数≈6（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
    # 证据：位置=runner/hints_exec.py:354；类型=function；引用≈6；规模≈184行
    s = str(cmd or "").strip()
    if not s:
        return "", "empty"

    # Strip common prompt prefixes that appear in docs (best-effort).
    # Only strip `$` when followed by whitespace to avoid breaking `$HOME/foo` style paths.
    cleaned: list[str] = []
    for raw in s.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("> "):
            line = line[2:].lstrip()
        if line.startswith("$") and len(line) >= 2 and line[1].isspace():
            line = line[2:].lstrip()
        if line.startswith(">>> "):
            line = line[4:].lstrip()
        if line.startswith("... "):
            line = line[4:].lstrip()
        cleaned.append(line)
    s = "\n".join(cleaned).strip()
    if not s:
        return "", "empty_after_sanitize"

    # Replace bracketed option groups like [a|b|c] -> a (first option).
    s2 = _BRACKET_GROUP_RE.sub(
        lambda m: (
            str(m.group(1) or "").split("|", 1)[0].strip()
            if "|" in str(m.group(1) or "")
            else m.group(0)
        ),
        s,
    )
    # Remove angle placeholders like <TENSOR_PARALLEL_SIZE>
    s2 = _ANGLE_GROUP_RE.sub("", s2)

    # Replace common GitHub Actions expressions (e.g., matrix python versions) with local defaults.
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    s2 = _GHA_EXPR_RE.sub(
        lambda m: (
            py_ver
            if (
                "matrix.python-version" in (inner := str(m.group(1) or "").strip().lower())
                or "matrix.python_version" in inner
                or "python-version" in inner
            )
            else ""
        ),
        s2,
    )

    # Replace model/base-url flags with env-provided values when available.
    model = (env.get("AIDER_LLM_MODEL") or env.get("OPENAI_MODEL") or "").strip()
    base_url = (env.get("OPENAI_API_BASE") or env.get("OPENAI_BASE_URL") or "").strip()
    if model:
        s2 = _replace_flag_value(s2, flag="--model", new_value=model)
    if base_url:
        s2 = _replace_flag_value(s2, flag="--base-url", new_value=base_url)
        s2 = _replace_flag_value(s2, flag="--base_url", new_value=base_url)

    # Normalize whitespace but preserve newlines for multi-command scripts.
    s2 = re.sub(r"[ \t]+", " ", s2)
    lines: list[str] = []
    for raw in s2.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(line)
    s2 = "\n".join(lines).strip()
    if not s2:
        return "", "empty_after_sanitize"

    # Best-effort: rewrite console-script style invocations like `pkg.module ...`
    # into `python -m pkg.module ...` when the entrypoint isn't available.
    #
    # This is generic and avoids benchmark-specific hardcoding (common in README/CI).
    py = (env.get("AIDER_FSM_PYTHON") or env.get("PYTHON") or "python3").strip() or "python3"
    # When the pipeline provides a repo-relative python path (e.g. `.aider_fsm/venv/bin/python`),
    # make it absolute so hints can run from any working directory.
    repo_root = str(env.get("AIDER_FSM_REPO_ROOT") or "").strip()
    if repo_root and ("/" in py or py.startswith((".", "~"))):
        try:
            p = Path(py).expanduser()
            if not p.is_absolute():
                cand = (Path(repo_root).expanduser().resolve() / p).resolve()
                if cand.exists():
                    py = str(cand)
        except Exception:
            pass

    rewritten: list[str] = []
    for line in s2.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            parts = shlex.split(line, posix=True)
        except Exception:
            rewritten.append(line)
            continue
        if not parts:
            rewritten.append(line)
            continue
        first = str(parts[0] or "").strip()
        if not first or "/" in first or first.startswith((".", "~")):
            rewritten.append(line)
            continue
        if _DOTTED_MODULE_RE.fullmatch(first) and shutil.which(first) is None:
            rest = " ".join(shlex.quote(str(p)) for p in parts[1:])
            base = f"{shlex.quote(py)} -m {shlex.quote(first)}"
            rewritten.append(f"{base} {rest}".strip() if rest else base)
        else:
            rewritten.append(line)
    s2 = "\n".join(rewritten).strip()

    py_is_path = ("/" in py) or py.startswith((".", "~"))
    py_first = shlex.split(shlex.quote(py))[0]
    rewritten_tools: list[str] = []
    for raw in s2.splitlines():
        line = raw.strip()
        if not line:
            continue
        if py_is_path:
            try:
                parts = shlex.split(line, posix=True)
            except Exception:
                rewritten_tools.append(line)
                continue
            if not parts:
                rewritten_tools.append(line)
                continue

            prefix: list[str] = []
            i = 0
            while i < len(parts) and _ENV_ASSIGN_RE.match(str(parts[i] or "").strip()):
                prefix.append(str(parts[i] or ""))
                i += 1
            if i < len(parts) and str(parts[i] or "") == "env":
                prefix.append("env")
                i += 1
                while i < len(parts) and _ENV_ASSIGN_RE.match(str(parts[i] or "").strip()):
                    prefix.append(str(parts[i] or ""))
                    i += 1
            if i >= len(parts):
                rewritten_tools.append(line)
                continue

            cmd0 = str(parts[i] or "")
            rest = [str(x or "") for x in parts[i + 1 :]]

            tok = cmd0.strip()
            is_py = tok in ("python", "python3") or bool(re.fullmatch(r"python\\d+(?:\\.\\d+)?", tok))
            is_pip = tok in ("pip", "pip3") or bool(re.fullmatch(r"pip\\d+(?:\\.\\d+)?", tok))

            if cmd0 != py and is_py:
                cmd0 = py
            elif is_pip:
                cmd0 = py
                rest = ["-m", "pip"] + rest
            elif cmd0 == "pytest":
                cmd0 = py
                rest = ["-m", "pytest"] + rest

            line = " ".join(shlex.quote(x) for x in (prefix + [cmd0] + rest)).strip()
        rewritten_tools.append(line)
    s2 = "\n".join(rewritten_tools).strip()

    fire_aliases = {
        "--base-url": "--base_url",
        "--n-samples": "--n_samples",
        "--id-range": "--id_range",
        "--i-just-wanna-run": "--i_just_wanna_run",
        "--test-details": "--test_details",
        "--base-only": "--base_only",
        "--output-file": "--output_file",
        "--min-time-limit": "--min_time_limit",
        "--gt-time-limit-factor": "--gt_time_limit_factor",
    }
    bounded: list[str] = []
    for line in s2.splitlines():
        # Heuristic: console-script entrypoints like `pkg.subcmd` are often python-fire CLIs.
        looks_fire = False
        try:
            parts = shlex.split(line, posix=True)
        except Exception:
            parts = []
        if parts:
            # Skip leading env assignments, e.g. `FOO=1 cmd ...`.
            i = 0
            while i < len(parts) and _ENV_ASSIGN_RE.match(str(parts[i] or "").strip()):
                i += 1
            if i < len(parts):
                first = str(parts[i] or "").strip()
                if first:
                    if _DOTTED_MODULE_RE.fullmatch(first):
                        looks_fire = True
                    # Also accept `python -m pkg.mod ...`.
                    elif first in ("python", "python3", py_first):
                        if i + 2 < len(parts) and str(parts[i + 1]) == "-m":
                            mod = str(parts[i + 2] or "").strip()
                            if _DOTTED_MODULE_RE.fullmatch(mod):
                                looks_fire = True
        if looks_fire:
            for old, new in fire_aliases.items():
                line = line.replace(old, new)

        # Best-effort: bound expensive "codegen + evaluate" hint commands using AIDER_EVAL_LIMIT.
        low = line.lower()
        if ("--samples" not in low) and (" -s " not in f" {low} "):
            if ("--backend openai" in low) or ("--backend=openai" in low):
                if ("--model" in low) and ("--dataset" in low) and ((".evaluate" in low) or (".codegen" in low)):
                    parts2 = line.split()
                    has_n_samples = any(p.startswith("--n_samples") for p in parts2)
                    if not has_n_samples:
                        line = (line + " --n_samples 1").strip()

        bounded.append(line)
    s2 = "\n".join(bounded)

    # Some repos recommend `pytest -n ...` (xdist), but the plugin is often missing in
    # minimal environments. Strip xdist-only flags by default to improve compatibility.
    strip_pytest_n = _is_truthy(env.get("AIDER_FSM_HINT_STRIP_PYTEST_N", "1"))
    if strip_pytest_n:
        stripped: list[str] = []
        for line in s2.splitlines():
            try:
                parts = shlex.split(line, posix=True)
            except Exception:
                stripped.append(line)
                continue
            if not parts:
                stripped.append(line)
                continue
            if "pytest" not in parts:
                stripped.append(line)
                continue
            out: list[str] = []
            i = 0
            while i < len(parts):
                tok = str(parts[i] or "")
                if tok == "-n":
                    # Drop `-n <N|auto>`; keep the next token if it looks like another flag.
                    if i + 1 < len(parts) and not str(parts[i + 1] or "").startswith("-"):
                        i += 2
                    else:
                        i += 1
                    continue
                if tok.startswith("-n="):
                    i += 1
                    continue
                if tok == "--dist":
                    if i + 1 < len(parts) and not str(parts[i + 1] or "").startswith("-"):
                        i += 2
                    else:
                        i += 1
                    continue
                if tok.startswith("--dist="):
                    i += 1
                    continue
                out.append(tok)
                i += 1
            stripped.append(" ".join(shlex.quote(x) for x in out))
        s2 = "\n".join(stripped)

    # If the command still contains bracket placeholders, it's likely not directly runnable.
    tmp = re.sub(r"\[\s*\d+\s*,\s*\d+\s*\]", "", s2)
    if "[" in tmp and "]" in tmp:
        return s2, "unresolved_brackets"
    if "<" in s2 and ">" in s2:
        return s2, "unresolved_angle_placeholders"

    # Block common unsafe installer patterns from docs.
    if _PIPE_TO_BASH_RE.search(s2):
        return s2, "blocked_pipe_to_bash"

    # Apply the runner's safe-mode denylist as a generic guardrail.
    allowed, reason = cmd_allowed(s2, pipeline=None)
    if not allowed:
        return s2, reason or "blocked_by_policy"
    return s2, None


@dataclass(frozen=True)
class HintAttempt:
    # 作用：内部符号：HintAttempt
    # 能否简略：否
    # 原因：规模≈9 行；引用次数≈9（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=runner/hints_exec.py:537；类型=class；引用≈9；规模≈9行
    raw: str
    sanitized: str
    rc: int
    seconds: float
    timed_out: bool
    stdout_tail: str
    stderr_tail: str
    skip_reason: str | None = None


@dataclass(frozen=True)
class HintProbe:
    # 作用：内部符号：HintProbe
    # 能否简略：否
    # 原因：规模≈6 行；引用次数≈7（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=runner/hints_exec.py:549；类型=class；引用≈7；规模≈6行
    raw: str
    sanitized: str
    ok: bool | None
    reason: str
    priority: int


def _tail(text: str, n: int) -> str:
    # 作用：内部符号：_tail
    # 能否简略：否
    # 原因：规模≈5 行；引用次数≈11（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=runner/hints_exec.py:557；类型=function；引用≈11；规模≈5行
    t = str(text or "")
    if len(t) <= n:
        return t
    return t[-n:]


def _docker_available(*, env: dict[str, str]) -> tuple[bool, str]:
    """Best-effort check for a usable local Docker daemon.

    This is intentionally generic and only used to avoid spending hint attempts on
    guaranteed-failing docker commands (e.g. Docker Desktop / Colima not running).
    """
    # 作用：Best-effort check for a usable local Docker daemon.
    # 能否简略：部分
    # 原因：规模≈25 行；引用次数≈2（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
    # 证据：位置=runner/hints_exec.py:574；类型=function；引用≈2；规模≈25行
    if shutil.which("docker") is None:
        return False, "docker_not_found"
    try:
        res = subprocess.run(
            ["docker", "info"],
            check=False,
            capture_output=True,
            text=True,
            timeout=6,
            env=env,
        )
    except Exception as e:
        return False, f"docker_info_failed: {e}"
    if int(res.returncode) != 0:
        tail = (res.stderr or res.stdout or "").strip()
        if len(tail) > 500:
            tail = tail[-500:]
        return False, tail or f"docker_info_rc={res.returncode}"
    return True, "ok"


def _extract_invoked_command(parts: list[str]) -> tuple[str, list[str]]:
    # 作用：内部符号：_extract_invoked_command
    # 能否简略：是
    # 原因：规模≈18 行；引用次数≈2（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
    # 证据：位置=runner/hints_exec.py:596；类型=function；引用≈2；规模≈18行
    i = 0
    n = len(parts)
    while i < n:
        tok = str(parts[i] or "").strip()
        if not tok:
            i += 1
            continue
        if _ENV_ASSIGN_RE.match(tok):
            i += 1
            continue
        if tok == "env":
            i += 1
            while i < n and _ENV_ASSIGN_RE.match(str(parts[i] or "").strip()):
                i += 1
            continue
        return tok, parts[i:]
    return "", []


def _probe_hint_command(
    *,
    cmd: str,
    repo: Path,
    env: dict[str, str],
    timeout_seconds: int,
) -> tuple[bool | None, str]:
    """Best-effort non-mutating probe for hint runnability."""
    # 作用：Best-effort non-mutating probe for hint runnability.
    # 能否简略：否
    # 原因：规模≈110 行；引用次数≈3（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=runner/hints_exec.py:623；类型=function；引用≈3；规模≈110行
    text = str(cmd or "").strip()
    if not text:
        return False, "empty"

    first_line = ""
    for raw in text.splitlines():
        s = raw.strip()
        if s:
            first_line = s
            break
    if not first_line:
        return False, "empty"

    try:
        parts = shlex.split(first_line, posix=True)
    except Exception:
        return None, "probe_shlex_failed"
    if not parts:
        return False, "empty"

    invoked, tail_parts = _extract_invoked_command(parts)
    if not invoked:
        return None, "probe_no_invoked_command"

    tok = str(invoked).strip()
    if not tok:
        return None, "probe_no_invoked_command"

    # Shell wrappers and scripts are hard to probe cheaply without side effects.
    if tok in ("bash", "sh", "zsh", "fish"):
        return None, "probe_shell_wrapper"
    if tok in _SHELL_BUILTINS:
        return None, "probe_shell_builtin"
    if "/" in tok or tok.startswith((".", "~")):
        return True, "ok"

    # Best-effort: if a hint references an explicit samples file, skip it early when missing.
    # This avoids spending attempts on obviously failing commands (common in README snippets).
    if tok != "docker":
        samples = ""
        i = 0
        while i < len(parts):
            t = str(parts[i] or "").strip()
            if t in ("--samples", "-s") and i + 1 < len(parts):
                samples = str(parts[i + 1] or "").strip()
                break
            if t.startswith("--samples="):
                samples = str(t.split("=", 1)[1] or "").strip()
                break
            i += 1
        if samples and not samples.startswith("-"):
            sp = Path(samples)
            if not sp.is_absolute():
                sp = (repo / sp).resolve()
            try:
                if not sp.exists():
                    return False, f"samples_not_found:{sp}"
            except Exception:
                pass

    # Validate python module entrypoints (`python -m xxx`) up front.
    py_names = {
        "python",
        "python3",
        Path(str(env.get("AIDER_FSM_PYTHON") or "")).name.strip(),
        Path(str(env.get("PYTHON") or "")).name.strip(),
    }
    py_names = {x for x in py_names if x}
    if tok in py_names:
        if len(tail_parts) >= 3 and str(tail_parts[1]) == "-m":
            module = str(tail_parts[2] or "").strip()
            if module:
                probe_py = (
                    str(env.get("AIDER_FSM_PYTHON") or "").strip()
                    or str(env.get("PYTHON") or "").strip()
                    or tok
                    or "python3"
                )
                code = (
                    "import importlib.util, sys; "
                    "m = (sys.argv[1] if len(sys.argv) > 1 else '').strip(); "
                    "sys.exit(0 if (m and importlib.util.find_spec(m) is not None) else 3)"
                )
                try:
                    res = subprocess.run(
                        [probe_py, "-c", code, module],
                        check=False,
                        capture_output=True,
                        text=True,
                        timeout=min(max(2, int(timeout_seconds)), 8),
                        cwd=str(repo),
                        env=env,
                    )
                except Exception as e:
                    return None, f"probe_module_check_failed:{e}"
                if int(res.returncode) != 0:
                    return False, f"module_not_found:{module}"
                return True, "ok"

    if shutil.which(tok) is None:
        return False, f"binary_not_found:{tok}"
    return True, "ok"


def _matched_anchors(text: str, *, anchors: list[str]) -> list[str]:
    # 作用：内部符号：_matched_anchors
    # 能否简略：部分
    # 原因：规模≈16 行；引用次数≈4（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
    # 证据：位置=runner/hints_exec.py:728；类型=function；引用≈4；规模≈16行
    if not anchors:
        return []
    low = str(text or "").lower()
    seen: set[str] = set()
    out: list[str] = []
    for raw in anchors:
        a = str(raw or "").strip()
        if not a:
            continue
        if a in seen:
            continue
        if a.lower() in low:
            seen.add(a)
            out.append(a)
    return out


def run_hints(
    *,
    repo: Path,
    max_attempts: int = 3,
    timeout_seconds: int = 600,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    # 作用：内部符号：run_hints
    # 能否简略：否
    # 原因：规模≈617 行；引用次数≈16（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=runner/hints_exec.py:752；类型=function；引用≈16；规模≈617行
    repo = Path(repo).resolve()
    env2 = dict(env or os.environ)

    raw_hints = _parse_json_str_list(env2.get("AIDER_FSM_HINTS_JSON"))
    if not raw_hints:
        hints_file: Path | None = None
        artifacts_root = (repo / ".aider_fsm" / "artifacts").resolve()
        if artifacts_root.exists():
            candidates = list(artifacts_root.glob("*/scaffold/scaffold_command_hints.txt"))
            if candidates:
                candidates.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
                hints_file = candidates[0]
        if hints_file is not None:
            try:
                text = hints_file.read_text(encoding="utf-8", errors="replace")
            except Exception:
                text = ""
            raw_hints = []
            for raw in text.splitlines():
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                raw_hints.append(line)

    anchors = _parse_json_str_list(env2.get("AIDER_FSM_HINT_ANCHORS_JSON"))
    used_anchors: list[str] = []

    kind = (env2.get("AIDER_LLM_KIND") or "").strip().lower()
    prefer_offline = _is_truthy(env2.get("AIDER_FSM_PREFER_OFFLINE_HINTS"))
    login_shell = _is_truthy(env2.get("AIDER_FSM_HINT_LOGIN_SHELL"))
    strict_compat = _is_truthy(env2.get("AIDER_FSM_HINT_STRICT_COMPAT", "1"))
    require_real_score = _is_truthy(env2.get("AIDER_FSM_REQUIRE_REAL_SCORE"))
    auto_uv_venv = _is_truthy(env2.get("AIDER_FSM_HINT_AUTO_UV", env2.get("AIDER_FSM_HINT_AUTO_UV_PY311", "1")))
    artifacts_dir_s = str(env2.get("AIDER_FSM_ARTIFACTS_DIR") or "").strip()
    artifacts_dir: Path | None = None
    if artifacts_dir_s:
        try:
            p = Path(artifacts_dir_s).expanduser()
            artifacts_dir = p.resolve() if p.is_absolute() else (repo / p).resolve()
        except Exception:
            artifacts_dir = None

    uv_py_candidates = _infer_uv_python_candidates(repo, env=env2)
    uv_py_env_default = str(env2.get("AIDER_FSM_HINT_UV_PYTHON") or env2.get("UV_PYTHON") or "").strip()
    if not uv_py_env_default and sys.version_info >= (3, 13) and uv_py_candidates:
        # When the runner itself is executed under a very new Python, steer uv toward a
        # stable minor with broad wheel availability, unless the caller already pinned it.
        uv_py_env_default = uv_py_candidates[0]

    uv_hint_env: dict[str, str] | None = None
    uv_hint_env_py: str = ""

    ranked_hints: list[tuple[int, str]] = []
    for raw in raw_hints:
        s = str(raw or "").lower()
        p = 0
        # Prefer commands that actually *run* evaluations/tests over setup/build steps.
        if "pytest" in s:
            p += 50
        if "evaluate" in s or "evaluation" in s:
            p += 20
        if "benchmark" in s:
            p += 10
        has_eval = ("pytest" in s) or ("evaluate" in s) or ("evaluation" in s) or ("benchmark" in s)
        if "docker build" in s and not has_eval:
            p -= 5
        if ("pip install" in s or "poetry install" in s or "conda install" in s) and not has_eval:
            p -= 10
        if prefer_offline:
            if " --samples" in f" {s} ":
                p += 15
            if " --backend openai" in s or " backend openai" in s:
                p -= 30
        else:
            if " --backend openai" in s or " backend openai" in s:
                p += 20
            if "openai" in s and kind == "remote":
                p += 5
        if s.startswith("docker "):
            p += 3
        if "vllm" in s and kind == "remote":
            p -= 5
        if "anthropic" in s or "bedrock" in s or "google" in s:
            p -= 5
        if "ollama" in s:
            p -= 3
        ranked_hints.append((p, raw))
    ranked_hints.sort(key=lambda t: t[0], reverse=True)

    attempts: list[HintAttempt] = []
    probes: list[HintProbe] = []
    chosen: str | None = None
    ok = False
    score = 0.0
    reason = ""
    rc0_no_score = False
    docker_status: tuple[bool, str] | None = None
    hint_work_root: Path | None = None

    candidates: list[dict[str, Any]] = []
    probe_timeout = min(max(3, int(timeout_seconds)), 20)
    seen_sanitized: set[str] = set()

    # Phase 1: normalize + probe runnability without executing hinted workloads.
    for priority, raw in ranked_hints:
        sanitized, skip_reason = normalize_hint_command(raw, env=env2)
        if skip_reason is not None:
            attempts.append(
                HintAttempt(
                    raw=raw,
                    sanitized=sanitized,
                    rc=0,
                    seconds=0.0,
                    timed_out=False,
                    stdout_tail="",
                    stderr_tail="",
                    skip_reason=skip_reason,
                )
            )
            probes.append(HintProbe(raw=raw, sanitized=sanitized, ok=False, reason=skip_reason, priority=priority))
            continue

        key = str(sanitized or "").strip()
        if key in seen_sanitized:
            attempts.append(
                HintAttempt(
                    raw=raw,
                    sanitized=sanitized,
                    rc=0,
                    seconds=0.0,
                    timed_out=False,
                    stdout_tail="",
                    stderr_tail="",
                    skip_reason="duplicate_sanitized_hint",
                )
            )
            probes.append(
                HintProbe(
                    raw=raw,
                    sanitized=sanitized,
                    ok=False,
                    reason="duplicate_sanitized_hint",
                    priority=priority,
                )
            )
            continue
        seen_sanitized.add(key)

        compat_ok, compat_reason = _hint_runtime_compatible(cmd=sanitized, env=env2, strict_compat=strict_compat)
        if not compat_ok:
            skip = f"incompatible_hint: {compat_reason}"
            attempts.append(
                HintAttempt(
                    raw=raw,
                    sanitized=sanitized,
                    rc=0,
                    seconds=0.0,
                    timed_out=False,
                    stdout_tail="",
                    stderr_tail="",
                    skip_reason=skip,
                )
            )
            probes.append(HintProbe(raw=raw, sanitized=sanitized, ok=False, reason=skip, priority=priority))
            continue

        if _DOCKER_LINE_RE.search(sanitized):
            if docker_status is None:
                docker_status = _docker_available(env=env2)
            if not docker_status[0]:
                skip = f"docker_unavailable: {docker_status[1]}"
                attempts.append(
                    HintAttempt(
                        raw=raw,
                        sanitized=sanitized,
                        rc=0,
                        seconds=0.0,
                        timed_out=False,
                        stdout_tail="",
                        stderr_tail="",
                        skip_reason=skip,
                    )
                )
                probes.append(HintProbe(raw=raw, sanitized=sanitized, ok=False, reason=skip, priority=priority))
                continue

        probe_ok, probe_reason = _probe_hint_command(
            cmd=sanitized,
            repo=repo,
            env=env2,
            timeout_seconds=probe_timeout,
        )
        probes.append(
            HintProbe(
                raw=raw,
                sanitized=sanitized,
                ok=probe_ok,
                reason=str(probe_reason or ""),
                priority=priority,
            )
        )
        candidates.append(
            {
                "raw": raw,
                "sanitized": sanitized,
                "priority": int(priority),
                "probe_ok": probe_ok,
                "probe_reason": str(probe_reason or ""),
            }
        )

    candidates.sort(
        key=lambda x: (
            (2 if x.get("probe_ok") is True else 1 if x.get("probe_ok") is None else 0),
            int(x.get("priority") or 0),
        ),
        reverse=True,
    )

    # Prefer a diverse first-attempt set so we don't burn the whole attempt budget
    # on a single category (e.g. many pytest hints failing for missing deps) while
    # skipping simpler setup hints that might succeed (e.g. `pip install pkg`).
    picked: set[int] = set()
    ordered: list[dict[str, Any]] = []
    for want in ("pytest", "install", "docker"):
        for i, cand in enumerate(candidates):
            if i in picked:
                continue
            low = str(cand.get("sanitized") or "").lower()
            match = False
            if want == "pytest":
                match = "pytest" in low
            elif want == "install":
                match = ("pip install" in low) or ("poetry install" in low) or ("conda install" in low)
            elif want == "docker":
                match = low.lstrip().startswith("docker ")
            if match:
                ordered.append(cand)
                picked.add(i)
                break
    for i, cand in enumerate(candidates):
        if i in picked:
            continue
        ordered.append(cand)

    # Phase 2: execute best candidates only (bounded by max_attempts).
    executed = 0
    openai_auth_failed = False
    for cand in ordered:
        if executed >= int(max(0, max_attempts)):
            break

        probe_ok = cand.get("probe_ok")
        probe_reason = str(cand.get("probe_reason") or "")
        raw = str(cand.get("raw") or "")
        sanitized = str(cand.get("sanitized") or "")
        first_low = _first_command_line(sanitized).lower()
        looks_openai_codegen = False
        if first_low:
            if ("--samples" not in first_low) and (" -s " not in f" {first_low} "):
                if ("--backend openai" in first_low) or ("--backend=openai" in first_low):
                    if ("--model" in first_low) and ("--dataset" in first_low):
                        if (".evaluate" in first_low) or (".codegen" in first_low):
                            looks_openai_codegen = True

        if openai_auth_failed and _is_remote_openai_hint(sanitized):
            attempts.append(
                HintAttempt(
                    raw=raw,
                    sanitized=sanitized,
                    rc=0,
                    seconds=0.0,
                    timed_out=False,
                    stdout_tail="",
                    stderr_tail="",
                    skip_reason="skipped_after_openai_auth_failure",
                )
            )
            continue

        if probe_ok is False:
            attempts.append(
                HintAttempt(
                    raw=raw,
                    sanitized=sanitized,
                    rc=0,
                    seconds=0.0,
                    timed_out=False,
                    stdout_tail="",
                    stderr_tail="",
                    skip_reason=f"probe_failed: {probe_reason or 'unrunnable'}",
                )
            )
            continue

        attempt_no = int(executed) + 1
        if artifacts_dir is None:
            workdir = repo
        elif _DOCKER_LINE_RE.search(sanitized) or looks_openai_codegen:
            if hint_work_root is None:
                hint_work_root = (artifacts_dir / "hints_workdir").resolve()
                hint_work_root.mkdir(parents=True, exist_ok=True)
            workdir = (hint_work_root / f"attempt_{attempt_no:02d}").resolve()
            workdir.mkdir(parents=True, exist_ok=True)
        else:
            workdir = repo
        metrics_paths = _candidate_metrics_paths(sanitized, repo=repo, workdir=workdir if workdir != repo else None)
        if require_real_score:
            metrics_paths.append((repo / ".aider_fsm" / "metrics.json").resolve())
        pre_mtimes: dict[Path, float] = {}
        for p in metrics_paths:
            try:
                if p.exists():
                    pre_mtimes[p] = float(p.stat().st_mtime)
            except Exception:
                continue

        t0 = time.monotonic()
        timed_out = False

        # IMPORTANT: prefer a non-login shell by default so bootstrap PATH overrides
        # (e.g. `.aider_fsm/venv/bin:$PATH`) are preserved. Login shells frequently
        # reset PATH via /etc/profile and can accidentally select global tools.
        env3: dict[str, str] = env2
        if uv_py_env_default:
            env3 = dict(env3)
            env3.setdefault("UV_PYTHON", uv_py_env_default)
        if workdir != repo:
            # When running from an artifacts workdir, keep the repo root importable
            # for `python -m pkg.mod` style hints that rely on local modules.
            if env3 is env2:
                env3 = dict(env3)
            repo_s = str(repo)
            pp = str(env3.get("PYTHONPATH") or "")
            parts = [p for p in pp.split(os.pathsep) if p]
            if repo_s not in parts:
                env3["PYTHONPATH"] = pp + (os.pathsep if pp else "") + repo_s
        if workdir != repo and looks_openai_codegen:
            # Best-effort: create a small dataset override file for smoke/full-lite runs.
            try:
                override_lim = int(str(env3.get("AIDER_EVAL_LIMIT") or "").strip() or 0)
            except Exception:
                override_lim = 0
            if override_lim > 0:
                override_dataset = _extract_cli_flag_value(sanitized, "--dataset").strip().lower()
                if override_dataset in ("humaneval", "mbpp"):
                    override_var = (
                        "HUMANEVAL_OVERRIDE_PATH" if override_dataset == "humaneval" else "MBPP_OVERRIDE_PATH"
                    )
                    if not str(env3.get(override_var) or "").strip():
                        # Identify the evaluator package from either:
                        # - `python -m pkg.mod ...` (module execution)
                        # - `pkg.mod ...` / `/path/to/pkg.mod ...` (console-script style)
                        override_line = _first_command_line(sanitized)
                        override_module = ""
                        try:
                            override_parts = shlex.split(override_line, posix=True) if override_line else []
                        except Exception:
                            override_parts = []
                        if override_parts:
                            if "-m" in override_parts:
                                try:
                                    override_module = str(override_parts[override_parts.index("-m") + 1] or "").strip()
                                except Exception:
                                    override_module = ""
                            if not override_module:
                                # Skip leading env assignments, e.g. `FOO=1 cmd ...`.
                                j = 0
                                while j < len(override_parts) and _ENV_ASSIGN_RE.match(str(override_parts[j] or "").strip()):
                                    j += 1
                                if j < len(override_parts):
                                    override_first = str(override_parts[j] or "").strip()
                                    # Allow either `pkg.mod` or `/abs/path/pkg.mod`.
                                    override_mod_cand = os.path.basename(override_first)
                                    if _DOTTED_MODULE_RE.fullmatch(override_mod_cand):
                                        override_module = override_mod_cand
                                    elif _DOTTED_MODULE_RE.fullmatch(override_first):
                                        override_module = override_first
                        if override_module and "." in override_module:
                            override_pkg = override_module.split(".", 1)[0].strip()
                            if override_pkg:
                                override_out_path = (
                                    (Path(workdir) / f"{override_dataset}_override_{override_lim}.jsonl").resolve()
                                )

                                reuse_ok = False
                                try:
                                    reuse_ok = (
                                        override_out_path.exists()
                                        and override_out_path.is_file()
                                        and override_out_path.stat().st_size > 0
                                    )
                                except Exception:
                                    reuse_ok = False
                                if reuse_ok:
                                    env3 = dict(env3)
                                    env3[override_var] = str(override_out_path)
                                else:
                                    # Prefer the runner-provided python when available.
                                    override_py_exec = str(
                                        env3.get("AIDER_FSM_PYTHON") or env3.get("PYTHON") or sys.executable
                                    ).strip() or sys.executable
                                    try:
                                        p2 = Path(override_py_exec).expanduser()
                                        if not p2.is_absolute() and (
                                            "/" in override_py_exec or override_py_exec.startswith((".", "~"))
                                        ):
                                            py_exec_cand = (repo / p2).resolve()
                                            if py_exec_cand.exists():
                                                override_py_exec = str(py_exec_cand)
                                    except Exception:
                                        pass

                                    override_code = r"""
import importlib
import json
import sys
from pathlib import Path

pkg = (sys.argv[1] if len(sys.argv) > 1 else "").strip()
dataset = (sys.argv[2] if len(sys.argv) > 2 else "").strip().lower()
out = Path(sys.argv[3] if len(sys.argv) > 3 else "").expanduser().resolve()
limit = int(sys.argv[4] if len(sys.argv) > 4 else "0")
if not pkg or not out or limit <= 0:
    raise SystemExit(2)

if dataset == "humaneval":
    dm = importlib.import_module(pkg + ".data.humaneval")
    src = dm._ready_human_eval_plus_path()
elif dataset == "mbpp":
    dm = importlib.import_module(pkg + ".data.mbpp")
    src = dm._ready_mbpp_plus_path()
else:
    raise SystemExit(3)

seen = set()
out.parent.mkdir(parents=True, exist_ok=True)
with open(src, "r", encoding="utf-8", errors="replace") as f, open(out, "w", encoding="utf-8") as g:
    for line in f:
        s = line.strip()
        if not s:
            continue
        obj = json.loads(s)
        tid = obj.get("task_id")
        if not isinstance(tid, str) or not tid.strip():
            continue
        if tid in seen:
            continue
        seen.add(tid)
        g.write(s + "\n")
        if len(seen) >= limit:
            break
if len(seen) <= 0:
    raise SystemExit(4)
"""
                                    try:
                                        res = subprocess.run(
                                            [
                                                override_py_exec,
                                                "-c",
                                                override_code,
                                                override_pkg,
                                                override_dataset,
                                                str(override_out_path),
                                                str(override_lim),
                                            ],
                                            check=False,
                                            capture_output=True,
                                            text=True,
                                            timeout=60,
                                            cwd=str(repo),
                                            env=env3,
                                        )
                                        if int(res.returncode) == 0 and override_out_path.exists():
                                            env3 = dict(env3)
                                            env3[override_var] = str(override_out_path)
                                    except Exception:
                                        pass

        try:
            bash_args = ["bash", "-lc", sanitized] if login_shell else ["bash", "-c", sanitized]
            res = subprocess.run(
                bash_args,
                check=False,
                capture_output=True,
                text=True,
                timeout=float(timeout_seconds),
                cwd=str(workdir),
                env=env3,
            )
            rc = int(res.returncode)
            out = res.stdout or ""
            err = res.stderr or ""
            this_timed_out = False
        except subprocess.TimeoutExpired as e:
            timed_out = True
            out_t = getattr(e, "stdout", "") or ""
            err_t = getattr(e, "stderr", "") or ""
            if isinstance(out_t, bytes):
                out_t = out_t.decode("utf-8", errors="replace")
            if isinstance(err_t, bytes):
                err_t = err_t.decode("utf-8", errors="replace")
            rc = 124
            out = str(out_t)
            err = str(err_t)
            this_timed_out = True

        # If we see a Python/C-extension build failure (common on Py3.13), retry once
        # inside a uv-managed venv with a more compatible Python.
        if rc != 0 and not this_timed_out and auto_uv_venv:
            tail_text = (_tail(out, 20000) + "\n" + _tail(err, 20000)).lower()
            looks_incompat = False
            if tail_text:
                if "greenlet" in tail_text and (
                    ("cframe" in tail_text) or ("_pycframe" in tail_text) or ("failed to build" in tail_text)
                ):
                    looks_incompat = True
                elif ("failed building wheel for" in tail_text) or ("could not build wheels for" in tail_text):
                    looks_incompat = True
                elif ("subprocess-exited-with-error" in tail_text) and (("error:" in tail_text) or ("failed" in tail_text)):
                    looks_incompat = True
                elif ("failed to build installable wheels" in tail_text) and (
                    ("pyproject.toml" in tail_text) or ("greenlet" in tail_text)
                ):
                    looks_incompat = True

            if looks_incompat:
                if sys.version_info >= (3, 13) or ("cp313" in tail_text) or ("python 3.13" in tail_text) or ("py3.13" in tail_text):
                    env_uv: dict[str, str] | None = None
                    prep_reason = ""
                    prep_py = ""
                    if uv_hint_env is not None:
                        env_uv = uv_hint_env
                        prep_reason = "cached"
                        prep_py = uv_hint_env_py
                    elif not auto_uv_venv:
                        prep_reason = "disabled"
                    elif shutil.which("uv") is None:
                        prep_reason = "uv_not_found"
                    elif not uv_py_candidates:
                        prep_reason = "no_uv_python_candidates"
                    else:
                        raw_venv_dir = str(env2.get("AIDER_FSM_HINT_UV_VENV_DIR") or "").strip()
                        uv_try = [uv_py_candidates[0]] if raw_venv_dir else list(uv_py_candidates)

                        last_err = ""
                        for py_req in uv_try:
                            try:
                                m = re.match(r"^\\s*(\\d+)\\.(\\d+)\\s*$", py_req)
                                tag = f"py{m.group(1)}{m.group(2)}" if m else "py"
                            except Exception:
                                tag = "py"

                            if raw_venv_dir:
                                venv_dir = Path(raw_venv_dir).expanduser()
                                if not venv_dir.is_absolute():
                                    venv_dir = (repo / venv_dir).resolve()
                            else:
                                venv_dir = (repo / ".aider_fsm" / f"venv_hints_{tag}").resolve()
                            py_bin = (venv_dir / "bin" / "python").absolute()
                            try:
                                venv_dir.parent.mkdir(parents=True, exist_ok=True)
                            except Exception:
                                pass
                            try:
                                uv_res = subprocess.run(
                                    ["uv", "venv", "--allow-existing", "--seed", "pip", "--python", py_req, str(venv_dir)],
                                    check=False,
                                    capture_output=True,
                                    text=True,
                                    timeout=600,
                                    cwd=str(repo),
                                    env=env2,
                                )
                            except Exception as e:
                                last_err = f"uv_venv_failed:{e}"
                                continue
                            try:
                                rc_i = int(getattr(uv_res, "returncode", 1))
                            except Exception:
                                rc_i = 1
                            if rc_i != 0:
                                tail = _tail(
                                    str(getattr(uv_res, "stderr", "") or "") + "\n" + str(getattr(uv_res, "stdout", "") or ""),
                                    2500,
                                )
                                last_err = f"uv_venv_failed_rc={getattr(uv_res, 'returncode', None)}:{tail}"
                                continue

                            envx = dict(env2)
                            old_path = str(envx.get("PATH") or "")
                            envx["PATH"] = str((venv_dir / "bin").absolute()) + (os.pathsep + old_path if old_path else "")
                            envx["VIRTUAL_ENV"] = str(venv_dir.absolute())
                            envx["AIDER_FSM_PYTHON"] = str(py_bin)
                            envx["PYTHON"] = str(py_bin)
                            envx.setdefault("UV_PYTHON", str(py_req).strip())
                            uv_hint_env = envx
                            uv_hint_env_py = str(py_req).strip()
                            env_uv = uv_hint_env
                            prep_reason = "ok"
                            prep_py = uv_hint_env_py
                            break
                        if env_uv is None:
                            prep_reason = last_err or "uv_venv_failed"
                    if env_uv is not None:
                        try:
                            bash_args = ["bash", "-lc", sanitized] if login_shell else ["bash", "-c", sanitized]
                            res = subprocess.run(
                                bash_args,
                                check=False,
                                capture_output=True,
                                text=True,
                                timeout=float(timeout_seconds),
                                cwd=str(workdir),
                                env=env_uv,
                            )
                            rc2 = int(res.returncode)
                            out2 = res.stdout or ""
                            err2 = res.stderr or ""
                        except subprocess.TimeoutExpired as e:
                            timed_out = True
                            out_t = getattr(e, "stdout", "") or ""
                            err_t = getattr(e, "stderr", "") or ""
                            if isinstance(out_t, bytes):
                                out_t = out_t.decode("utf-8", errors="replace")
                            if isinstance(err_t, bytes):
                                err_t = err_t.decode("utf-8", errors="replace")
                            rc2 = 124
                            out2 = str(out_t)
                            err2 = str(err_t)
                        # Keep the retry result, but also surface that a retry happened.
                        out = out2
                        extra = f"{prep_reason}" + (f" python={prep_py}" if prep_py else "")
                        err = f"(retry_uv_venv: {extra})\n{err2}"
                        rc = rc2

        executed += 1
        dt = time.monotonic() - t0
        attempts.append(
            HintAttempt(
                raw=raw,
                sanitized=sanitized,
                rc=rc,
                seconds=float(dt),
                timed_out=timed_out,
                stdout_tail=_tail(out, 4000),
                stderr_tail=_tail(err, 4000),
                skip_reason=None,
            )
        )

        low_cmd = sanitized.lower()

        if rc == 0 and require_real_score:
            # Prefer deterministic file outputs when available, otherwise parse stdout/stderr.
            extracted: float | None = None
            source = ""

            # Pytest: use passed/total as a concrete score (even when rc==0).
            if "pytest" in low_cmd:
                tail_text = _tail(out, 20000) + "\n" + _tail(err, 20000)
                counts = None
                t = str(tail_text or "")
                # Strip ANSI color codes so regexes can match reliably.
                t = re.sub(r"\x1b\[[0-9;]*m", "", t)
                passed_ms = list(re.finditer(r"(?i)\b(\d+)\s+passed\b", t))
                failed_ms = list(re.finditer(r"(?i)\b(\d+)\s+failed\b", t))
                errors_ms = list(re.finditer(r"(?i)\b(\d+)\s+error(?:s)?\b", t))
                try:
                    passed = int(passed_ms[-1].group(1)) if passed_ms else 0
                except Exception:
                    passed = 0
                try:
                    failed = int(failed_ms[-1].group(1)) if failed_ms else 0
                except Exception:
                    failed = 0
                try:
                    errors = int(errors_ms[-1].group(1)) if errors_ms else 0
                except Exception:
                    errors = 0
                total = passed + failed + errors
                if total > 0:
                    counts = (passed, failed, errors)
                if counts is not None:
                    passed, failed, errors = counts
                    total = max(1, passed + failed + errors)
                    extracted = float(passed) / float(total)
                    source = f"pytest_counts: passed={passed} failed={failed} errors={errors}"

            if extracted is None:
                for p in metrics_paths:
                    try:
                        if not p.exists():
                            continue
                        mt = float(p.stat().st_mtime)
                        if p in pre_mtimes and mt <= pre_mtimes[p] + 1e-6:
                            continue
                    except Exception:
                        continue
                    try:
                        data = json.loads(p.read_text(encoding="utf-8", errors="replace"))
                    except Exception as e:
                        val, src = None, f"metrics_json_parse_failed:{e}"
                    else:
                        val, src = _extract_score_from_json_obj(data)
                    if val is not None:
                        extracted = float(val)
                        source = f"file:{p.name}:{src}"
                        break

            if extracted is None:
                val, src = _extract_score_from_text(_tail(out, 20000) + "\n" + _tail(err, 20000))
                if val is not None:
                    extracted = float(val)
                    source = src

            if extracted is not None:
                chosen = sanitized
                ok = True
                score = float(extracted)
                reason = str(source or "ok")
                used_anchors = _matched_anchors(sanitized, anchors=anchors)
                break

            rc0_no_score = True
            # Continue trying other hints to find one that yields a parseable score.
            continue

        if rc == 0:
            chosen = sanitized
            ok = True
            score = 1.0
            reason = ""
            used_anchors = _matched_anchors(sanitized, anchors=anchors)
            break

        # Some evaluation commands (notably `pytest`) intentionally return non-zero
        # when checks fail, but still produce useful numeric metrics. Treat these as
        # a "successful run" and derive score from the outputs.
        if "pytest" in low_cmd:
            tail_text = _tail(out, 20000) + "\n" + _tail(err, 20000)
            counts = None
            t = str(tail_text or "")
            # Strip ANSI color codes so regexes can match reliably.
            t = re.sub(r"\x1b\[[0-9;]*m", "", t)
            passed_ms = list(re.finditer(r"(?i)\b(\d+)\s+passed\b", t))
            failed_ms = list(re.finditer(r"(?i)\b(\d+)\s+failed\b", t))
            errors_ms = list(re.finditer(r"(?i)\b(\d+)\s+error(?:s)?\b", t))
            try:
                passed = int(passed_ms[-1].group(1)) if passed_ms else 0
            except Exception:
                passed = 0
            try:
                failed = int(failed_ms[-1].group(1)) if failed_ms else 0
            except Exception:
                failed = 0
            try:
                errors = int(errors_ms[-1].group(1)) if errors_ms else 0
            except Exception:
                errors = 0
            total = passed + failed + errors
            if total > 0:
                counts = (passed, failed, errors)
            if counts is not None:
                passed, failed, errors = counts
                total = max(1, passed + failed + errors)
                chosen = sanitized
                ok = True
                score = float(passed) / float(total)
                reason = f"pytest_nonzero_exit: passed={passed} failed={failed} errors={errors}"
                used_anchors = _matched_anchors(sanitized, anchors=anchors)
                break

        if _is_remote_openai_hint(sanitized):
            low = (_tail(out, 12000) + "\n" + _tail(err, 12000)).lower()
            needles = (
                "invalid_api_key",
                "incorrect api key provided",
                "authenticationerror",
                "error code: 401",
                "status': 401",
                "status: 401",
            )
            if any(n in low for n in needles):
                openai_auth_failed = True

    if not ok:
        if not raw_hints:
            reason = "no_hints"
        elif require_real_score and rc0_no_score:
            reason = "all_hints_no_real_score"
        elif openai_auth_failed:
            reason = "all_hints_auth_failed_or_unrunnable"
        elif candidates and not any((c.get("probe_ok") is not False) for c in candidates):
            reason = "all_hints_unrunnable"
        elif any(a.skip_reason == "unresolved_brackets" for a in attempts):
            reason = "all_hints_unresolved_or_failed"
        else:
            reason = "all_hints_failed"

    return {
        "ok": bool(ok),
        "score": float(score) if ok else 0.0,
        "chosen_command": chosen,
        "used_anchors": used_anchors,
        "executed_attempts": int(executed),
        "probes": [
            {
                "raw": p.raw,
                "sanitized": p.sanitized,
                "ok": p.ok,
                "reason": p.reason,
                "priority": p.priority,
            }
            for p in probes
        ],
        "attempts": [
            {
                "raw": a.raw,
                "sanitized": a.sanitized,
                "rc": a.rc,
                "seconds": a.seconds,
                "timed_out": a.timed_out,
                "stdout_tail": a.stdout_tail,
                "stderr_tail": a.stderr_tail,
                "skip_reason": a.skip_reason,
            }
            for a in attempts
        ],
        "reason": reason,
    }


def main() -> int:
    # 作用：内部符号：main
    # 能否简略：否
    # 原因：规模≈12 行；引用次数≈25（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=runner/hints_exec.py:1365；类型=function；引用≈25；规模≈12行
    import argparse

    ap = argparse.ArgumentParser(description="Run doc-derived hint commands with best-effort sanitization.")
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--timeout-seconds", type=int, default=600)
    args = ap.parse_args()

    repo_root = Path(os.environ.get("AIDER_FSM_REPO_ROOT") or ".").resolve()
    res = run_hints(repo=repo_root, max_attempts=int(args.max_attempts), timeout_seconds=int(args.timeout_seconds))
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0 if res.get("ok") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
