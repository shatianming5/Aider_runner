from __future__ import annotations

from pathlib import Path


def test_repo_has_no_rd_agent_specific_refs():
    """中文说明：
    - 含义：保证本仓库不再残留特定项目（rd_agent 等）硬编码字样。
    - 内容：扫描 README/docs/examples（存在即扫描），若发现包含 banned 关键字则失败并列出文件。
    - 可简略：否（用于守住“无特定项目硬编码”的核心约束；建议保留）。
    """
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
