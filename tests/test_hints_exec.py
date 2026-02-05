from __future__ import annotations

from runner.hints_exec import normalize_hint_command


def test_normalize_hint_command_rewrites_dotted_entrypoint_to_python_module() -> None:
    cmd, reason = normalize_hint_command("foo.bar --x 1", env={"AIDER_FSM_PYTHON": "python3"})
    assert reason is None
    assert cmd.startswith("python3 -m foo.bar --x 1")


def test_normalize_hint_command_keeps_existing_python_module_invocations() -> None:
    cmd, reason = normalize_hint_command("python3 -m foo.bar --x 1", env={"AIDER_FSM_PYTHON": "python3"})
    assert reason is None
    assert cmd == "python3 -m foo.bar --x 1"

