import json
from pathlib import Path

from runner.state import append_jsonl, default_state, load_state, save_state


def test_state_roundtrip(tmp_path: Path):
    """中文说明：
    - 含义：验证 state 的 save/load 基本回环可用。
    - 内容：生成 defaults，写入 state.json，再加载并断言关键字段存在且值与 defaults 一致。
    - 可简略：可能（可断言更多字段；当前覆盖主路径）。
    """
    state_path = tmp_path / "state.json"
    defaults = default_state(
        repo=tmp_path, plan_rel="PLAN.md", model="gpt-4o-mini", test_cmd="pytest -q"
    )
    save_state(state_path, defaults)
    loaded = load_state(state_path, defaults)
    assert loaded["version"] == defaults["version"]
    assert loaded["plan_path"] == "PLAN.md"


def test_state_merge_defaults(tmp_path: Path):
    """中文说明：
    - 含义：验证 `load_state` 会把旧 state 与 defaults 合并（缺失字段从 defaults 补齐）。
    - 内容：state.json 只写入 iter_idx，再加载并断言 iter_idx 保留且 plan_path 从 defaults 补齐。
    - 可简略：可能（可增加更多字段组合；当前覆盖合并语义主路径）。
    """
    state_path = tmp_path / "state.json"
    defaults = default_state(
        repo=tmp_path, plan_rel="PLAN.md", model="gpt-4o-mini", test_cmd="pytest -q"
    )
    state_path.write_text(json.dumps({"iter_idx": 7}) + "\n", encoding="utf-8")
    loaded = load_state(state_path, defaults)
    assert loaded["iter_idx"] == 7
    assert loaded["plan_path"] == "PLAN.md"


def test_append_jsonl(tmp_path: Path):
    """中文说明：
    - 含义：验证 `append_jsonl` 会以 JSONL 方式追加多行记录。
    - 内容：向同一文件追加两条 dict，读取并断言行数为 2。
    - 可简略：是（简单 IO 覆盖）。
    """
    log = tmp_path / "run.jsonl"
    append_jsonl(log, {"a": 1})
    append_jsonl(log, {"b": 2})
    lines = log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
