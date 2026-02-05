from __future__ import annotations

from pathlib import Path

from runner.eval_audit import audit_eval_script_for_hardcoded_nonzero_score


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_audit_eval_script_detects_hardcoded_decimal_score(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _write(
        repo / ".aider_fsm" / "stages" / "evaluation.sh",
        "#!/usr/bin/env bash\n"
        "set -Eeuo pipefail\n"
        "SCORE=0.8\n"
        "echo ok\n",
    )
    msg = audit_eval_script_for_hardcoded_nonzero_score(repo)
    assert msg is not None
    assert "hardcoded_nonzero_score" in msg


def test_audit_eval_script_allows_zero_score(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _write(
        repo / ".aider_fsm" / "stages" / "evaluation.sh",
        "#!/usr/bin/env bash\n"
        "set -Eeuo pipefail\n"
        "SCORE=0.0\n"
        "echo ok\n",
    )
    msg = audit_eval_script_for_hardcoded_nonzero_score(repo)
    assert msg is None

