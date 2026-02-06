from __future__ import annotations

from pathlib import Path

import pytest

from runner.hints_exec import normalize_hint_command, run_hints


def test_normalize_hint_command_rewrites_dotted_entrypoint_to_python_module() -> None:
    cmd, reason = normalize_hint_command("foo.bar --x 1", env={"AIDER_FSM_PYTHON": "python3"})
    assert reason is None
    assert cmd.startswith("python3 -m foo.bar --x 1")


def test_normalize_hint_command_keeps_existing_python_module_invocations() -> None:
    cmd, reason = normalize_hint_command("python3 -m foo.bar --x 1", env={"AIDER_FSM_PYTHON": "python3"})
    assert reason is None
    assert cmd == "python3 -m foo.bar --x 1"


def test_run_hints_skips_docker_commands_when_docker_unavailable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)

    # One docker-based hint; if Docker isn't reachable we should skip it without attempting execution.
    env = {"AIDER_FSM_HINTS_JSON": '["docker run --rm hello-world"]'}

    monkeypatch.setattr("runner.hints_exec.shutil.which", lambda name: "/usr/bin/docker" if name == "docker" else None)

    def fake_run(cmd, *args, **kwargs):
        if cmd == ["docker", "info"]:
            class R:
                returncode = 1
                stdout = ""
                stderr = "daemon not running"

            return R()
        raise AssertionError(f"unexpected subprocess.run: {cmd!r}")

    monkeypatch.setattr("runner.hints_exec.subprocess.run", fake_run)

    res = run_hints(repo=repo, max_attempts=3, timeout_seconds=5, env=env)
    assert res.get("ok") is False
    attempts = res.get("attempts") or []
    assert attempts and attempts[0].get("skip_reason", "").startswith("docker_unavailable:")
