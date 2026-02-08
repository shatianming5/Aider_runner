from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path

from runner.env_local import open_env, rollout_and_evaluate


def _py() -> str:
    """中文说明：
    - 含义：返回当前 Python 解释器的 shell-quoted 路径。
    - 内容：用于拼接 pipeline.yml 里的 rollout/evaluation 命令，避免空格等导致的 shell 解析问题。
    - 可简略：是（测试 helper；也可在各处直接 `shlex.quote(sys.executable)`）。
    """
    # 作用：中文说明：
    # 能否简略：是
    # 原因：测试代码（优先可读性）；规模≈7 行；引用次数≈3（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
    # 证据：位置=tests/test_env_local.py:17；类型=function；引用≈3；规模≈7行
    return shlex.quote(sys.executable)


def test_env_local_rollout_and_evaluate(tmp_path: Path):
    """中文说明：
    - 含义：验证本地 env 的 rollout + evaluation 能端到端执行并产生可读 metrics。
    - 内容：构造 pipeline.yml，其中 rollout 写 rollout.json，evaluation 读之并写 metrics.json；调用 `rollout_and_evaluate` 并检查产物与 artifacts。
    - 可简略：否（同机调用接口是核心能力；该测试覆盖最小可用闭环）。
    """
    # 作用：中文说明：
    # 能否简略：否
    # 原因：测试代码（优先可读性）；规模≈81 行；引用次数≈1（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=tests/test_env_local.py:26；类型=function；引用≈1；规模≈81行
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".aider_fsm").mkdir()

    rollout_cmd = (
        f"{_py()} -c "
        + shlex.quote(
            "import json, os, pathlib; "
            "pathlib.Path('.aider_fsm').mkdir(parents=True, exist_ok=True); "
            "pathlib.Path('.aider_fsm/rollout.json').write_text("
            "json.dumps({'ok': True, 'model': os.getenv('OPENCODE_MODEL','')})+'\\n'"
            ")"
        )
    )
    eval_cmd = (
        f"{_py()} -c "
        + shlex.quote(
            "import json, os, pathlib; "
            "rollout=json.loads(pathlib.Path('.aider_fsm/rollout.json').read_text()); "
            "score=1 if rollout.get('ok') else 0; "
            "pathlib.Path('.aider_fsm').mkdir(parents=True, exist_ok=True); "
            "pathlib.Path('.aider_fsm/metrics.json').write_text("
            "json.dumps({'score': score, 'model': os.getenv('OPENCODE_MODEL','')})+'\\n'"
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
                "rollout:",
                "  run_cmds:",
                f"    - {json.dumps(rollout_cmd)}",
                "evaluation:",
                "  run_cmds:",
                f"    - {json.dumps(eval_cmd)}",
                "  metrics_path: .aider_fsm/metrics.json",
                "  required_keys: [score]",
                "artifacts:",
                "  out_dir: .aider_fsm/artifacts",
                "",
            ]
        ),
        encoding="utf-8",
    )

    env = open_env(repo)
    out_dir = tmp_path / "artifacts_out"
    rollout_res, eval_res = rollout_and_evaluate(
        env,
        artifacts_dir=out_dir,
        env_overrides={"OPENCODE_MODEL": "opencode/gpt-5-nano"},
    )

    assert rollout_res.ok is True
    assert eval_res.ok is True

    assert rollout_res.rollout_path is not None
    assert rollout_res.rollout_path.exists()

    assert eval_res.metrics_path is not None
    assert eval_res.metrics_path.exists()

    assert eval_res.metrics is not None
    assert eval_res.metrics.get("score") == 1
    assert eval_res.metrics.get("model") == "opencode/gpt-5-nano"

    # Runner writes stage artifacts into the specified artifacts dir.
    assert (out_dir / "evaluation_summary.json").exists()
