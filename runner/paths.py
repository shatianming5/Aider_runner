from __future__ import annotations

from pathlib import Path


def is_relative_to(path: Path, base: Path) -> bool:
    """中文说明：
    - 含义：判断 `path` 是否位于 `base` 路径之下（包含自身）。
    - 内容：通过 `Path.relative_to()` 尝试计算相对路径，成功则返回 True。
    - 可简略：是（若要求 Python>=3.9，可直接用 `Path.is_relative_to`）。
    """
    # 作用：中文说明：
    # 能否简略：否
    # 原因：规模≈11 行；引用次数≈6（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=runner/paths.py:12；类型=function；引用≈6；规模≈11行
    try:
        path.relative_to(base)
        return True
    except Exception:
        return False


def relpath_or_none(path: Path, base: Path) -> str | None:
    """中文说明：
    - 含义：将 `path` 转为相对 `base` 的字符串；若不在其下则返回 None。
    - 内容：用于把绝对路径转换为 repo 内相对路径，方便写入 artifacts/log。
    - 可简略：可能（小工具；也可在调用点直接写 `relative_to` 并捕获异常）。
    """
    # 作用：中文说明：
    # 能否简略：否
    # 原因：规模≈9 行；引用次数≈1（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=runner/paths.py:25；类型=function；引用≈1；规模≈9行
    if not is_relative_to(path, base):
        return None
    return str(path.relative_to(base))


def resolve_config_path(repo: Path, raw: str) -> Path:
    """中文说明：
    - 含义：把配置路径参数（可能是相对路径）解析为绝对路径。
    - 内容：相对路径会按 repo 根目录拼接；随后 `resolve()` 规范化。
    - 可简略：可能（通用工具；但集中封装有利于统一行为与测试）。
    """
    # 作用：中文说明：
    # 能否简略：否
    # 原因：规模≈10 行；引用次数≈1（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=runner/paths.py:36；类型=function；引用≈1；规模≈10行
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = repo / p
    return p.resolve()


def resolve_workdir(repo: Path, workdir: str | None) -> Path:
    """中文说明：
    - 含义：解析 pipeline/actions/bootstrap 的 `workdir` 配置并做越界保护。
    - 内容：将 workdir 解析为 repo 内绝对路径；若不在 repo 下则抛错，避免命令在 repo 外执行。
    - 可简略：否（安全边界；不建议在调用点各自实现）。
    """
    # 作用：中文说明：
    # 能否简略：否
    # 原因：规模≈15 行；引用次数≈8（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=runner/paths.py:48；类型=function；引用≈8；规模≈15行
    if not workdir or not str(workdir).strip():
        return repo
    p = Path(workdir).expanduser()
    if not p.is_absolute():
        p = repo / p
    p = p.resolve()
    if not is_relative_to(p, repo):
        raise ValueError(f"workdir must be within repo: {p} (repo={repo})")
    return p
