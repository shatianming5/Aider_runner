from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path

from runner.cli import main


def test_pipeline_injects_runner_root_and_pythonpath(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()

    py = shlex.quote(sys.executable)
    cmd = (
        f"{py} -c "
        + shlex.quote(
            "import os, pathlib, runner; "
            "root=os.environ.get('AIDER_FSM_RUNNER_ROOT',''); "
            "assert root, 'missing AIDER_FSM_RUNNER_ROOT'; "
            "rp=pathlib.Path(root).resolve(); "
            "assert rp.exists(), 'runner_root_not_exists'; "
            "pp=os.environ.get('PYTHONPATH',''); "
            "assert str(rp) in pp.split(os.pathsep), 'runner_root_not_in_PYTHONPATH'; "
            "rf=pathlib.Path(runner.__file__).resolve(); "
            "rf.relative_to(rp); "
            "assert os.environ.get('AIDER_FSM_PYTHON',''), 'missing AIDER_FSM_PYTHON'; "
            "print('ok')"
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
                "  cmds:",
                f"    - {json.dumps(cmd)}",
                "artifacts:",
                "  out_dir: .aider_fsm/artifacts",
                "",
            ]
        ),
        encoding="utf-8",
    )

    rc = main(
        [
            "--repo",
            str(repo),
            "--preflight-only",
            "--require-pipeline",
            "--env-file",
            "",
        ]
    )
    assert rc == 0

