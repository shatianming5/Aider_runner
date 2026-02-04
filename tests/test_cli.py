import shlex
import sys
from pathlib import Path

from runner.cli import main


def _py_cmd(exit_code: int) -> str:
    """中文说明：
    - 含义：构造一个会以指定 exit code 退出的 Python 命令。
    - 内容：用于把 tests_cmds 写进 pipeline.yml，确保 CLI 流程可控地返回 0/非 0。
    - 可简略：是（测试样板代码；可直接内联）。
    """
    py = shlex.quote(sys.executable)
    return f'{py} -c "import sys; sys.exit({exit_code})"'


def test_cli_auto_discovers_root_pipeline_for_local_repo(tmp_path: Path):
    """中文说明：
    - 含义：验证 CLI 在本地 repo 场景能自动发现根目录的 pipeline.yml。
    - 内容：写入最小 pipeline.yml（tests + artifacts），调用 `main([... --preflight-only])`，断言返回码为 0。
    - 可简略：可能（也可覆盖多层目录/显式路径；当前覆盖“默认发现”主路径）。
    """
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
