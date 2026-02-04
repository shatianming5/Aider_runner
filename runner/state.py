from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


STATE_VERSION = 1


def now_iso() -> str:
    """中文说明：
    - 含义：生成本地时区的 ISO-like 时间戳字符串（用于日志/状态）。
    - 内容：基于 `time.strftime`，格式形如 `YYYY-MM-DDTHH:MM:SS±ZZZZ`。
    - 可简略：可能（可换成 `datetime` 更标准；但当前实现足够且无额外依赖）。
    """
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())


def ensure_dirs(repo: Path) -> tuple[Path, Path, Path]:
    """中文说明：
    - 含义：确保 `.aider_fsm/` 与 `.aider_fsm/logs/` 目录存在，并返回常用路径。
    - 内容：创建目录（best-effort）；返回 (state_dir, logs_dir, state_path)。
    - 可简略：否（统一目录结构是 artifacts/状态可追溯的基础）。
    """
    state_dir = repo / ".aider_fsm"
    logs_dir = state_dir / "logs"
    state_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)
    return state_dir, logs_dir, state_dir / "state.json"


def default_state(
    *, repo: Path, plan_rel: str, model: str, test_cmd: str, pipeline_rel: str | None = None
) -> dict[str, Any]:
    """中文说明：
    - 含义：生成 `.aider_fsm/state.json` 的默认状态结构（含版本与关键运行参数）。
    - 内容：记录当前迭代、FSM 状态、当前 step、各 stage 最近一次 rc 等；用于断点恢复与审计。
    - 可简略：否（状态字段是 runner 行为的“解释器”；缺字段会降低可观测性）。
    """
    return {
        "version": STATE_VERSION,
        "repo": str(repo),
        "plan_path": plan_rel,
        "pipeline_path": pipeline_rel or "",
        "model": model,
        "test_cmd": test_cmd,
        "iter_idx": 0,
        "fsm_state": "S0_BOOTSTRAP",
        "current_step_id": None,
        "current_step_text": None,
        "fix_attempts": 0,
        "last_bootstrap_rc": None,
        "last_rollout_rc": None,
        "last_test_rc": None,
        "last_deploy_setup_rc": None,
        "last_deploy_health_rc": None,
        "last_benchmark_rc": None,
        "last_evaluation_rc": None,
        "last_metrics_ok": None,
        "last_exit_reason": None,
        "updated_at": now_iso(),
    }


def load_state(path: Path, defaults: dict[str, Any]) -> dict[str, Any]:
    """中文说明：
    - 含义：从 state.json 读取状态，并与 defaults 合并（容忍文件缺失/损坏）。
    - 内容：读取失败或格式不对时回退到 defaults；每次加载都会更新 `updated_at`。
    - 可简略：可能（薄封装；但对坏文件的容错在无人值守场景很关键）。
    """
    if not path.exists():
        return dict(defaults)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return dict(defaults)
    if not isinstance(data, dict):
        return dict(defaults)
    merged = dict(defaults)
    merged.update(data)
    merged["updated_at"] = now_iso()
    return merged


def save_state(path: Path, state: dict[str, Any]) -> None:
    """中文说明：
    - 含义：把 state dict 以 pretty JSON 写入到 `.aider_fsm/state.json`。
    - 内容：统一缩进与编码，方便 diff/排查。
    - 可简略：可能（与 `subprocess_utils.write_json` 功能类似；可考虑合并）。
    """
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """中文说明：
    - 含义：向 jsonl 日志文件追加一条记录。
    - 内容：每行一个 JSON object，便于流式记录与后续机器解析。
    - 可简略：可能（很小的 IO 封装；但统一 jsonl 格式很有价值）。
    """
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
