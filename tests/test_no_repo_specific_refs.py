from __future__ import annotations

from pathlib import Path


def test_repo_has_no_rd_agent_specific_refs():
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

