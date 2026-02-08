from __future__ import annotations

from pathlib import Path

from runner.eval_audit import audit_eval_script_for_hardcoded_nonzero_score


def _write(path: Path, content: str) -> None:
    # 作用：内部符号：_write
    # 能否简略：否
    # 原因：测试代码（优先可读性）；规模≈3 行；引用次数≈24（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=tests/test_eval_audit.py:9；类型=function；引用≈24；规模≈3行
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_audit_eval_script_detects_hardcoded_decimal_score(tmp_path: Path) -> None:
    # 作用：pytest 测试用例：验证行为契约
    # 能否简略：否
    # 原因：测试代码（优先可读性）；规模≈12 行；引用次数≈1（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=tests/test_eval_audit.py:14；类型=function；引用≈1；规模≈12行
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
    # 作用：pytest 测试用例：验证行为契约
    # 能否简略：否
    # 原因：测试代码（优先可读性）；规模≈11 行；引用次数≈1（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=tests/test_eval_audit.py:28；类型=function；引用≈1；规模≈11行
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

