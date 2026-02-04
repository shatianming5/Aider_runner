from __future__ import annotations

import re

from .pipeline_spec import PipelineSpec


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


def compile_patterns(patterns: list[str] | tuple[str, ...]) -> list[re.Pattern[str]]:
    """中文说明：
    - 含义：把字符串规则（正则/普通字符串）编译为 regex 列表。
    - 内容：对非法正则会降级为 `re.escape` 以保证“配置写错也能安全运行”；用于 deny/allow 策略匹配。
    - 可简略：可能（工具函数；但对错误正则的降级策略很实用）。
    """
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


def matches_any(patterns: list[re.Pattern[str]], text: str) -> str | None:
    """中文说明：
    - 含义：判断文本是否命中任意正则；命中则返回命中的 pattern 字符串。
    - 内容：用于给出“为什么被阻止”的可解释原因（返回命中的规则）。
    - 可简略：可能（小工具；但可解释性对审计很关键）。
    """
    for p in patterns:
        if p.search(text):
            return p.pattern
    return None


def looks_interactive(cmd: str) -> bool:
    """中文说明：
    - 含义：粗略判断命令是否可能需要交互输入（strict unattended 下要阻止）。
    - 内容：目前只覆盖少量高风险/高概率卡住的命令（如 `docker login`、`gh auth login`）。
    - 可简略：可能（启发式可以扩展/收缩；但保留此检查能显著降低卡死风险）。
    """
    s = cmd.strip().lower()
    if not s:
        return False

    # Heuristics to avoid hanging in strict unattended runs.
    if s.startswith("docker login") and "--password-stdin" not in s and " -p " not in s and " --password " not in s:
        return True
    if " gh auth login" in f" {s}" and "--with-token" not in s:
        return True
    return False


def safe_env(base: dict[str, str], extra: dict[str, str], *, unattended: str) -> dict[str, str]:
    """中文说明：
    - 含义：构造给 stage/actions/bootstrap 命令使用的环境变量。
    - 内容：合并 base+extra；在 strict 模式下设置 `CI=1`、`GIT_TERMINAL_PROMPT=0` 等以减少交互/噪音。
    - 可简略：否（安全/可复现运行的重要一环）。
    """
    env = dict(base)
    env.update({k: str(v) for k, v in extra.items()})
    if unattended == "strict":
        env.setdefault("CI", "1")
        env.setdefault("GIT_TERMINAL_PROMPT", "0")
        env.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
        env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def cmd_allowed(cmd: str, *, pipeline: PipelineSpec | None) -> tuple[bool, str | None]:
    """中文说明：
    - 含义：判断某条命令是否允许执行，并返回 (允许?, 原因)。
    - 内容：始终应用 hard deny（如 rm -rf /、fork bomb）；若无 pipeline 合同则用默认 safe denylist；若有 pipeline 则按 `security.mode/allowlist/denylist` 决定。
    - 可简略：否（这是 runner 的核心安全边界；任何简化都可能引入破坏性风险）。
    """
    cmd = cmd.strip()
    if not cmd:
        return False, "empty_command"

    hard_deny = compile_patterns(_HARD_DENY_PATTERNS)
    hit = matches_any(hard_deny, cmd)
    if hit:
        return False, f"blocked_by_hard_deny: {hit}"

    # If no pipeline is provided, default to safe mode deny patterns.
    if pipeline is None:
        deny = compile_patterns(_SAFE_DEFAULT_DENY_PATTERNS)
        hit = matches_any(deny, cmd)
        if hit:
            return False, f"blocked_by_default_safe_deny: {hit}"
        return True, None

    mode = (pipeline.security_mode or "safe").strip().lower()
    if mode not in ("safe", "system"):
        return False, f"invalid_security_mode: {mode}"

    deny_patterns = list(pipeline.security_denylist or [])
    if mode == "safe":
        deny_patterns.extend(list(_SAFE_DEFAULT_DENY_PATTERNS))
    deny = compile_patterns(deny_patterns)
    hit = matches_any(deny, cmd)
    if hit:
        return False, f"blocked_by_denylist: {hit}"

    allow_patterns = list(pipeline.security_allowlist or [])
    if allow_patterns:
        allow = compile_patterns(allow_patterns)
        if matches_any(allow, cmd) is None:
            return False, "blocked_by_allowlist"

    return True, None
