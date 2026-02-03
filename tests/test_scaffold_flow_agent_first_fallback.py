from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path

from runner.agent_client import AgentResult
from runner.runner import RunnerConfig, run


def _ok_cmd() -> str:
    py = shlex.quote(sys.executable)
    return f'{py} -c "import sys; sys.exit(0)"'


def _latest_run_dir(artifacts_base: Path) -> Path:
    runs = sorted([p for p in artifacts_base.iterdir() if p.is_dir()])
    assert runs, f"no run dirs under {artifacts_base}"
    return runs[-1]


class _AgentWritesValidPipeline:
    def __init__(self, repo: Path):
        self._repo = repo

    def run(self, _text: str, *, fsm_state: str, iter_idx: int, purpose: str) -> AgentResult:
        if purpose == "scaffold_contract":
            py = shlex.quote(sys.executable)
            ok_cmd = _ok_cmd()
            write_metrics = (
                f"{py} -c \"import json, pathlib; "
                "pathlib.Path('.aider_fsm').mkdir(parents=True, exist_ok=True); "
                "pathlib.Path('.aider_fsm/metrics.json').write_text(json.dumps({'score': 0})+'\\\\n')\""
            )
            (self._repo / "pipeline.yml").write_text(
                "\n".join(
                    [
                        "version: 1",
                        "security:",
                        "  mode: safe",
                        "  max_cmd_seconds: 60",
                        "tests:",
                        "  cmds:",
                        f"    - {json.dumps(ok_cmd)}",
                        "benchmark:",
                        "  run_cmds:",
                        f"    - {json.dumps(write_metrics)}",
                        "  metrics_path: .aider_fsm/metrics.json",
                        "  required_keys: [score]",
                        "artifacts:",
                        "  out_dir: .aider_fsm/artifacts",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
        return AgentResult(assistant_text=f"ok ({purpose})")

    def close(self) -> None:
        return


class _AgentWritesInvalidPipeline:
    def __init__(self, repo: Path):
        self._repo = repo

    def run(self, _text: str, *, fsm_state: str, iter_idx: int, purpose: str) -> AgentResult:
        if purpose == "scaffold_contract":
            (self._repo / "pipeline.yml").write_text("version: 2\n", encoding="utf-8")
        return AgentResult(assistant_text=f"ok ({purpose})")

    def close(self) -> None:
        return


class _AgentWritesParseableButMissingMetricsContract:
    def __init__(self, repo: Path):
        self._repo = repo

    def run(self, _text: str, *, fsm_state: str, iter_idx: int, purpose: str) -> AgentResult:
        if purpose == "scaffold_contract":
            (self._repo / "pipeline.yml").write_text(
                "\n".join(
                    [
                        "version: 1",
                        "artifacts:",
                        "  out_dir: .aider_fsm/artifacts",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
        return AgentResult(assistant_text=f"ok ({purpose})")

    def close(self) -> None:
        return


def test_opencode_scaffold_agent_first_success(tmp_path: Path):
    repo = tmp_path / "repo_ok"
    repo.mkdir()

    artifacts_base = repo / ".aider_fsm" / "artifacts"
    ok_cmd = _ok_cmd()
    cfg = RunnerConfig(
        repo=repo,
        goal="g",
        model="openai/gpt-4o-mini",
        plan_rel="PLAN.md",
        pipeline_abs=None,
        pipeline_rel=None,
        pipeline=None,
        tests_cmds=[ok_cmd],
        effective_test_cmd=ok_cmd,
        artifacts_base=artifacts_base,
        seed_files=[],
        max_iters=1,
        max_fix=1,
        unattended="strict",
        preflight_only=True,
        opencode_url="",
        opencode_timeout_seconds=1,
        opencode_bash="restricted",
        require_pipeline=True,
        scaffold_contract="opencode",
    )

    rc = run(cfg, agent=_AgentWritesValidPipeline(repo))
    assert rc == 0

    assert (repo / "pipeline.yml").exists()
    assert (repo / ".aider_fsm" / "metrics.json").exists()

    run_dir = _latest_run_dir(artifacts_base)
    assert (run_dir / "scaffold_agent_result.txt").exists()
    assert (run_dir / "preflight" / "summary.json").exists()


def test_opencode_scaffold_invalid_pipeline_fails_fast(tmp_path: Path):
    repo = tmp_path / "repo_bad"
    repo.mkdir()

    artifacts_base = repo / ".aider_fsm" / "artifacts"
    ok_cmd = _ok_cmd()
    cfg = RunnerConfig(
        repo=repo,
        goal="g",
        model="openai/gpt-4o-mini",
        plan_rel="PLAN.md",
        pipeline_abs=None,
        pipeline_rel=None,
        pipeline=None,
        tests_cmds=[ok_cmd],
        effective_test_cmd=ok_cmd,
        artifacts_base=artifacts_base,
        seed_files=[],
        max_iters=1,
        max_fix=1,
        unattended="strict",
        preflight_only=True,
        opencode_url="",
        opencode_timeout_seconds=1,
        opencode_bash="restricted",
        require_pipeline=True,
        scaffold_contract="opencode",
    )

    rc = run(cfg, agent=_AgentWritesInvalidPipeline(repo))
    assert rc == 2

    run_dir = _latest_run_dir(artifacts_base)
    assert (run_dir / "scaffold_agent_pipeline_parse_error.txt").exists()
    assert (run_dir / "scaffold_error.txt").exists()
    assert not (run_dir / "preflight").exists()


def test_opencode_scaffold_incomplete_pipeline_fails_fast(tmp_path: Path):
    repo = tmp_path / "repo_incomplete"
    repo.mkdir()

    artifacts_base = repo / ".aider_fsm" / "artifacts"
    ok_cmd = _ok_cmd()
    cfg = RunnerConfig(
        repo=repo,
        goal="g",
        model="openai/gpt-4o-mini",
        plan_rel="PLAN.md",
        pipeline_abs=None,
        pipeline_rel=None,
        pipeline=None,
        tests_cmds=[ok_cmd],
        effective_test_cmd=ok_cmd,
        artifacts_base=artifacts_base,
        seed_files=[],
        max_iters=1,
        max_fix=1,
        unattended="strict",
        preflight_only=True,
        opencode_url="",
        opencode_timeout_seconds=1,
        opencode_bash="restricted",
        require_pipeline=True,
        scaffold_contract="opencode",
    )

    rc = run(cfg, agent=_AgentWritesParseableButMissingMetricsContract(repo))
    assert rc == 2

    run_dir = _latest_run_dir(artifacts_base)
    assert (run_dir / "scaffold_agent_pipeline_validation_error.txt").exists()
    assert (run_dir / "scaffold_error.txt").exists()
    assert not (run_dir / "preflight").exists()
