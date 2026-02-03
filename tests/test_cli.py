import shlex
import sys
from pathlib import Path

from runner.cli import main


def _py_cmd(exit_code: int) -> str:
    py = shlex.quote(sys.executable)
    return f'{py} -c "import sys; sys.exit({exit_code})"'


def test_cli_auto_discovers_root_pipeline_for_local_repo(tmp_path: Path):
    repo = tmp_path
    (repo / "pipeline.yml").write_text(
        "\n".join(
            [
                "version: 1",
                "tests:",
                f"  cmds: [{_py_cmd(0)}]",
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

