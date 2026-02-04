from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path

from runner.opencode_run import main


def _py() -> str:
    return shlex.quote(sys.executable)


def test_opencode_run_local_repo_deploy_rollout_evaluation(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".aider_fsm").mkdir()
    trained = tmp_path / "trained_model"
    trained.mkdir()

    # Deploy writes runtime_env.json.
    deploy_setup_cmd = (
        f"{_py()} -c "
        + shlex.quote(
            "import json, pathlib; "
            "pathlib.Path('.aider_fsm').mkdir(parents=True, exist_ok=True); "
            "pathlib.Path('.aider_fsm/runtime_env.json').write_text("
            "json.dumps({'ok': True, 'service': {'base_url': 'http://example.invalid'}})+'\\n'"
            ")"
        )
    )
    deploy_health_cmd = f"{_py()} -c " + shlex.quote("print('ok')")

    # Rollout/eval MUST read AIDER_RUNTIME_ENV_PATH (no hardcoded default path).
    rollout_cmd = (
        f"{_py()} -c "
        + shlex.quote(
            "import json, os, pathlib; "
            "p=os.environ['AIDER_RUNTIME_ENV_PATH']; "
            "env=json.loads(pathlib.Path(p).read_text()); "
            "pathlib.Path('.aider_fsm/rollout.json').write_text(json.dumps({'ok': True, 'base_url': env['service']['base_url']})+'\\n')"
        )
    )
    eval_cmd = (
        f"{_py()} -c "
        + shlex.quote(
            "import json, os, pathlib; "
            "r=json.loads(pathlib.Path('.aider_fsm/rollout.json').read_text()); "
            "pathlib.Path('.aider_fsm/metrics.json').write_text("
            "json.dumps({'ok': bool(r.get('ok')), 'score': 1 if r.get('ok') else 0, 'trained_model_dir': os.getenv('AIDER_TRAINED_MODEL_DIR','')})+'\\n'"
            ")"
        )
    )

    (repo / "pipeline.yml").write_text(
        "\n".join(
            [
                "version: 1",
                "security:",
                "  mode: safe",
                "  max_cmd_seconds: 60",
                "tests:",
                "  cmds: [echo ok]",
                "deploy:",
                "  setup_cmds:",
                f"    - {json.dumps(deploy_setup_cmd)}",
                "  health_cmds:",
                f"    - {json.dumps(deploy_health_cmd)}",
                "rollout:",
                "  run_cmds:",
                f"    - {json.dumps(rollout_cmd)}",
                "evaluation:",
                "  run_cmds:",
                f"    - {json.dumps(eval_cmd)}",
                "  metrics_path: .aider_fsm/metrics.json",
                "  required_keys: [score, ok]",
                "",
            ]
        ),
        encoding="utf-8",
    )

    out_dir = tmp_path / "artifacts_out"
    rc = main(
        [
            "--repo",
            str(repo),
            "--trained-model-dir",
            str(trained),
            "--artifacts-dir",
            str(out_dir),
            "--env-file",
            "",
        ]
    )
    assert rc == 0

    assert (repo / ".aider_fsm" / "runtime_env.json").exists()
    assert (repo / ".aider_fsm" / "rollout.json").exists()
    assert (repo / ".aider_fsm" / "metrics.json").exists()
    runtime_env = json.loads((repo / ".aider_fsm" / "runtime_env.json").read_text(encoding="utf-8"))
    assert runtime_env.get("inference", {}).get("model_dir") == str(trained)
    metrics = json.loads((repo / ".aider_fsm" / "metrics.json").read_text(encoding="utf-8"))
    assert metrics.get("score") == 1
    assert metrics.get("trained_model_dir") == str(trained)

    # Runner writes stage artifacts into the specified artifacts dir.
    assert (out_dir / "deploy" / "deploy_setup_summary.json").exists()
    assert (out_dir / "deploy" / "deploy_health_summary.json").exists()
    assert (out_dir / "rollout_evaluation" / "evaluation_summary.json").exists()
