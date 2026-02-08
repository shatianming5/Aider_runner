from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from .types import CmdResult


STDIO_TAIL_CHARS = 8000
ARTIFACT_TEXT_LIMIT_CHARS = 2_000_000


def tail(text: str, n: int) -> str:
    """中文说明：
    - 含义：截取字符串末尾 `n` 个字符（用于日志/输出截断）。
    - 内容：如果文本超过 n，则只保留尾部，避免 artifacts 体积过大。
    - 可简略：可能（非常小的工具函数；但集中使用可统一截断策略）。
    """
    # 作用：中文说明：
    # 能否简略：否
    # 原因：规模≈9 行；引用次数≈48（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=runner/subprocess_utils.py:21；类型=function；引用≈48；规模≈9行
    if len(text) <= n:
        return text
    return text[-n:]


def run_cmd(cmd: str, cwd: Path) -> tuple[int, str, str]:
    """中文说明：
    - 含义：在指定目录执行 shell 命令，并返回 (rc, stdout_tail, stderr_tail)。
    - 内容：使用 `subprocess.run(..., shell=True, capture_output=True)`，并对 stdout/stderr 做尾部截断。
    - 可简略：可能（与 `run_cmd_capture` 有部分功能重叠，但返回结构不同）。
    """
    # 作用：中文说明：
    # 能否简略：否
    # 原因：规模≈14 行；引用次数≈5（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=runner/subprocess_utils.py:32；类型=function；引用≈5；规模≈14行
    p = subprocess.run(
        cmd,
        cwd=str(cwd),
        shell=True,
        text=True,
        capture_output=True,
    )
    return p.returncode, tail(p.stdout, STDIO_TAIL_CHARS), tail(p.stderr, STDIO_TAIL_CHARS)


def run_cmd_capture(
    cmd: str,
    cwd: Path,
    *,
    timeout_seconds: int | None = None,
    env: dict[str, str] | None = None,
    interactive: bool = False,
) -> CmdResult:
    """中文说明：
    - 含义：执行命令并捕获完整 stdout/stderr（或交互模式不捕获）。
    - 内容：支持 timeout；超时返回 rc=124 并标记 timed_out；交互模式用于 `--unattended guided` 的少量场景。
    - 可简略：否（runner 的命令执行/审计核心之一，调用广泛）。
    """
    # 作用：中文说明：
    # 能否简略：否
    # 原因：规模≈39 行；引用次数≈10（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=runner/subprocess_utils.py:55；类型=function；引用≈10；规模≈39行
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


def limit_text(text: str, limit: int) -> str:
    """中文说明：
    - 含义：将文本裁剪到最大长度（用于写文件 artifacts 时的硬上限）。
    - 内容：超过限制时保留前缀并追加 `...[truncated]...` 标记。
    - 可简略：可能（小工具；但统一上限很重要）。
    """
    # 作用：中文说明：
    # 能否简略：否
    # 原因：规模≈9 行；引用次数≈2（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=runner/subprocess_utils.py:89；类型=function；引用≈2；规模≈9行
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]...\n"


def write_text(path: Path, text: str) -> None:
    """中文说明：
    - 含义：写入文本文件，并限制最大内容长度。
    - 内容：自动创建父目录；对内容做 `ARTIFACT_TEXT_LIMIT_CHARS` 截断；用于记录日志与 stage 输出。
    - 可简略：否（集中处理目录创建与截断，避免调用点重复/遗漏）。
    """
    # 作用：中文说明：
    # 能否简略：否
    # 原因：规模≈8 行；引用次数≈97（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=runner/subprocess_utils.py:100；类型=function；引用≈97；规模≈8行
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(limit_text(text, ARTIFACT_TEXT_LIMIT_CHARS), encoding="utf-8", errors="replace")


def write_json(path: Path, data: Any) -> None:
    """中文说明：
    - 含义：以 UTF-8 写入 pretty JSON（带缩进 + 末尾换行）。
    - 内容：用于写 summary/state/metrics 等结构化 artifacts。
    - 可简略：可能（小工具；也可用 `json.dump`，但此处统一格式更利于 diff/审计）。
    """
    # 作用：中文说明：
    # 能否简略：否
    # 原因：规模≈8 行；引用次数≈12（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=runner/subprocess_utils.py:110；类型=function；引用≈12；规模≈8行
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_text_if_exists(path: Path) -> str:
    """中文说明：
    - 含义：读取文件文本；若不存在则返回空字符串。
    - 内容：避免调用点重复写 `exists()` 判断，常用于读取可选配置/历史 artifacts。
    - 可简略：可能（薄封装；但可提升可读性与一致性）。
    """
    # 作用：中文说明：
    # 能否简略：否
    # 原因：规模≈9 行；引用次数≈10（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=runner/subprocess_utils.py:120；类型=function；引用≈10；规模≈9行
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def write_cmd_artifacts(out_dir: Path, prefix: str, res: CmdResult) -> None:
    """中文说明：
    - 含义：把一次命令执行结果按固定文件名落盘到 artifacts 目录。
    - 内容：写入 `<prefix>_cmd/stdout/stderr/result.json`，便于离线审计与定位失败。
    - 可简略：否（契约化 artifacts 命名与结构；删改会影响上层读取与测试）。
    """
    # 作用：中文说明：
    # 能否简略：否
    # 原因：规模≈13 行；引用次数≈17（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=runner/subprocess_utils.py:131；类型=function；引用≈17；规模≈13行
    write_text(out_dir / f"{prefix}_cmd.txt", res.cmd + "\n")
    write_text(out_dir / f"{prefix}_stdout.txt", res.stdout)
    write_text(out_dir / f"{prefix}_stderr.txt", res.stderr)
    write_json(
        out_dir / f"{prefix}_result.json",
        {"rc": res.rc, "timed_out": res.timed_out},
    )
