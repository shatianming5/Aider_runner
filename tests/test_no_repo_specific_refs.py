from __future__ import annotations

from pathlib import Path


def test_repo_has_no_rd_agent_specific_refs():
    """中文说明：
    - 含义：保证本仓库不再残留特定项目（rd_agent 等）硬编码字样。
    - 内容：扫描 README/docs/examples（存在即扫描），若发现包含 banned 关键字则失败并列出文件。
    - 可简略：否（用于守住“无特定项目硬编码”的核心约束；建议保留）。
    """
    # 作用：中文说明：
    # 能否简略：否
    # 原因：测试代码（优先可读性）；规模≈25 行；引用次数≈1（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=tests/test_no_repo_specific_refs.py:12；类型=function；引用≈1；规模≈25行
    root = Path(__file__).resolve().parents[1]
    banned = ("rd_agent", "rd-agent")

    files: list[Path] = [root / "README.md"]
    for base in (root / "docs", root / "examples"):
        if base.exists():
            files.extend([p for p in base.rglob("*") if p.is_file()])

    offenders: list[str] = []
    for p in files:
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        lower = text.lower()
        if any(b in lower for b in banned):
            offenders.append(str(p.relative_to(root)))

    assert offenders == []


def test_runner_code_has_no_benchmark_identity_hardcoding():
    """Ensure orchestration code does not branch on concrete benchmark identities."""
    # 作用：Ensure orchestration code does not branch on concrete benchmark identities.
    # 能否简略：否
    # 原因：测试代码（优先可读性）；规模≈21 行；引用次数≈1（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=tests/test_no_repo_specific_refs.py:35；类型=function；引用≈1；规模≈21行
    root = Path(__file__).resolve().parents[1]
    banned = ("gsm8k", "evalplus", "miniwob", "miniwob-plusplus")

    code_files: list[Path] = []
    runner_dir = root / "runner"
    if runner_dir.exists():
        code_files.extend([p for p in runner_dir.rglob("*.py") if p.is_file()])

    offenders: list[str] = []
    for p in code_files:
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        lower = text.lower()
        if any(token in lower for token in banned):
            offenders.append(str(p.relative_to(root)))

    assert offenders == []
