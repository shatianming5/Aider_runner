import shlex
import sys
from pathlib import Path

import pytest

from runner.bootstrap import load_bootstrap_spec, run_bootstrap


def _py_cmd(code: str) -> str:
    py = shlex.quote(sys.executable)
    return f'{py} -c "{code}"'


def test_load_bootstrap_spec_ok(tmp_path: Path):
    p = tmp_path / "bootstrap.yml"
    p.write_text(
        "\n".join(
            [
                "version: 1",
                "env: {FOO: bar}",
                "cmds: [echo ok]",
                "workdir: .",
                "timeout_seconds: 10",
                "retries: 2",
                "",
            ]
        ),
        encoding="utf-8",
    )
    spec, raw = load_bootstrap_spec(p)
    assert raw.strip()
    assert spec.version == 1
    assert spec.env["FOO"] == "bar"
    assert spec.cmds == ["echo ok"]
    assert spec.workdir == "."
    assert spec.timeout_seconds == 10
    assert spec.retries == 2


def test_load_bootstrap_spec_invalid_version(tmp_path: Path):
    p = tmp_path / "bootstrap.yml"
    p.write_text("version: 2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_bootstrap_spec(p)


def test_run_bootstrap_applies_env_and_runs_cmd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    repo = tmp_path
    (repo / ".aider_fsm").mkdir(parents=True, exist_ok=True)
    bootstrap_path = repo / ".aider_fsm" / "bootstrap.yml"
    code = "import os,sys; sys.exit(0 if os.environ.get('BAR')=='bar-baz' else 3)"
    bootstrap_path.write_text(
        "\n".join(
            [
                "version: 1",
                "env:",
                "  FOO: bar",
                "  BAR: ${FOO}-baz",
                "cmds:",
                f"  - {_py_cmd(code)}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    # Ensure a stable PATH for expansion tests (even though this spec doesn't change it).
    monkeypatch.setenv("PATH", "/usr/bin")

    stage, applied_env = run_bootstrap(
        repo,
        bootstrap_path=bootstrap_path,
        pipeline=None,
        unattended="strict",
        artifacts_dir=repo / "artifacts",
    )
    assert stage.ok is True
    assert applied_env["FOO"] == "bar"
    assert applied_env["BAR"] == "bar-baz"
