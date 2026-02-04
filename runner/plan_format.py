from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .pipeline_spec import PipelineSpec


_STEP_RE = re.compile(r"^\s*-\s*\[\s*([xX ])\s*\]\s*\(STEP_ID=([0-9]+)\)\s*(.*?)\s*$")


def plan_template(goal: str, test_cmd: str, *, pipeline: PipelineSpec | None = None) -> str:
    """中文说明：
    - 含义：生成一个最小可解析的 `PLAN.md` 初始模板。
    - 内容：包含 Goal/Acceptance/Next/Backlog/Done/Notes；Acceptance 至少包含 `TEST_CMD passes`；可按 pipeline 是否启用 deploy/rollout/eval/benchmark 追加条目。
    - 可简略：否（PLAN 格式是 runner 的协议；变化会影响解析与闭环行为）。
    """
    goal = goal.strip() or "<fill goal>"
    acceptance: list[str] = [f"- [ ] TEST_CMD passes: `{test_cmd}`"]
    if pipeline:
        if pipeline.deploy_setup_cmds or pipeline.deploy_health_cmds:
            acceptance.append("- [ ] Deploy succeeds (see pipeline.yml)")
        if getattr(pipeline, "rollout_run_cmds", None):
            acceptance.append("- [ ] Rollout succeeds (see pipeline.yml)")
        if getattr(pipeline, "evaluation_run_cmds", None):
            acceptance.append("- [ ] Evaluation succeeds (see pipeline.yml)")
        if pipeline.benchmark_run_cmds:
            acceptance.append("- [ ] Benchmark succeeds (see pipeline.yml)")
        if (
            getattr(pipeline, "evaluation_metrics_path", None)
            or getattr(pipeline, "evaluation_required_keys", None)
            or pipeline.benchmark_metrics_path
            or pipeline.benchmark_required_keys
        ):
            acceptance.append("- [ ] Metrics file/keys present (see pipeline.yml)")
    return (
        "# PLAN\n"
        "\n"
        "## Goal\n"
        f"- {goal}\n"
        "\n"
        "## Acceptance\n"
        + "\n".join(acceptance)
        + "\n"
        "\n"
        "## Next (exactly ONE item)\n"
        "- [ ] (STEP_ID=001) Build Backlog: break the goal into smallest steps (each step = one edit + one verify)\n"
        "\n"
        "## Backlog\n"
        "\n"
        "## Done\n"
        "- [x] (STEP_ID=000) Initialized plan file\n"
        "\n"
        "## Notes\n"
        "- \n"
    )


def ensure_plan_file(plan_abs: Path, goal: str, test_cmd: str, *, pipeline: PipelineSpec | None = None) -> None:
    """中文说明：
    - 含义：确保 `PLAN.md` 存在；不存在则按模板创建。
    - 内容：只在文件缺失时写入；避免覆盖用户已有计划。
    - 可简略：可能（薄封装；但让 runner 主流程更清晰）。
    """
    if plan_abs.exists():
        return
    plan_abs.parent.mkdir(parents=True, exist_ok=True)
    plan_abs.write_text(plan_template(goal, test_cmd, pipeline=pipeline), encoding="utf-8")


def _extract_section_lines(lines: list[str], heading_prefix: str) -> list[str] | None:
    """中文说明：
    - 含义：从 PLAN.md 行列表中提取某个 `## ...` 小节的内容行。
    - 内容：找到以 heading_prefix 开头的行作为起点，直到下一个 `## ` 标题或文件结束。
    - 可简略：否（解析器的核心小工具，集中处理边界情况）。
    """
    start = None
    for i, line in enumerate(lines):
        if line.strip().startswith(heading_prefix):
            start = i + 1
            break
    if start is None:
        return None
    end = len(lines)
    for j in range(start, len(lines)):
        if lines[j].strip().startswith("## "):
            end = j
            break
    return lines[start:end]


def _parse_step_lines(section_lines: list[str]) -> tuple[list[dict[str, Any]], list[str]]:
    """中文说明：
    - 含义：解析某个小节内的 step 列表行（`- [ ] (STEP_ID=NNN) ...`）。
    - 内容：返回 (steps, bad_lines)；bad_lines 用于指出格式不符合协议的行。
    - 可简略：否（PLAN 协议解析关键；错误行需要被显式报告）。
    """
    steps: list[dict[str, Any]] = []
    bad: list[str] = []
    for line in section_lines:
        if not re.match(r"^\s*-\s*\[", line):
            continue
        m = _STEP_RE.match(line)
        if not m:
            bad.append(line.strip())
            continue
        checked = m.group(1).lower() == "x"
        steps.append({"id": m.group(2), "text": m.group(3).strip(), "checked": checked})
    return steps, bad


def find_duplicate_step_ids(plan_text: str) -> list[str]:
    """中文说明：
    - 含义：检测 PLAN.md 中是否存在重复的 STEP_ID（跨 Next/Backlog/Done）。
    - 内容：返回重复的 id 列表；用于拒绝不一致的计划状态。
    - 可简略：否（避免 FSM 误选/误标记 step）。
    """
    lines = plan_text.splitlines()
    ids: list[str] = []
    for heading in ("## Next", "## Backlog", "## Done"):
        section = _extract_section_lines(lines, heading)
        if section is None:
            continue
        steps, _bad = _parse_step_lines(section)
        ids.extend([s["id"] for s in steps])
    counts: dict[str, int] = {}
    for sid in ids:
        counts[sid] = counts.get(sid, 0) + 1
    return [sid for sid, c in counts.items() if c > 1]


def parse_next_step(plan_text: str) -> tuple[dict[str, str] | None, str | None]:
    """中文说明：
    - 含义：解析 `## Next (exactly ONE item)` 中的唯一未完成 step。
    - 内容：要求 Next 区域恰好 1 条未勾选 step；否则返回 (None, error_code)。
    - 可简略：否（这是 runner 决定“下一步只做一件事”的核心约束）。
    """
    lines = plan_text.splitlines()
    section = _extract_section_lines(lines, "## Next")
    if section is None:
        return None, "missing_next_section"
    steps, bad = _parse_step_lines(section)
    if bad:
        return None, "bad_next_line"
    if len(steps) == 0:
        return None, None
    if len(steps) != 1:
        return None, "next_count_not_one"
    if steps[0]["checked"]:
        return None, "next_is_checked"
    dups = find_duplicate_step_ids(plan_text)
    if dups:
        return None, "duplicate_step_id"
    return {"id": steps[0]["id"], "text": steps[0]["text"]}, None


def parse_backlog_open_count(plan_text: str) -> tuple[int, str | None]:
    """中文说明：
    - 含义：统计 Backlog 中未完成 step 的数量。
    - 内容：用于展示进度或作为停止条件的输入（例如 Done/Stop 逻辑）。
    - 可简略：可能（小功能；但保持与解析规则一致更可靠）。
    """
    lines = plan_text.splitlines()
    section = _extract_section_lines(lines, "## Backlog")
    if section is None:
        return 0, "missing_backlog_section"
    steps, bad = _parse_step_lines(section)
    if bad:
        return 0, "bad_backlog_line"
    return sum(1 for s in steps if not s["checked"]), None


def parse_plan(plan_text: str) -> dict[str, Any]:
    """中文说明：
    - 含义：对 PLAN.md 做一次整体解析（Next + Backlog 统计 + 错误列表）。
    - 内容：聚合 parse_next_step/parse_backlog_open_count 的结果，统一返回结构，便于 runner 日志与决策。
    - 可简略：可能（聚合函数；但提升调用点可读性）。
    """
    next_step, next_err = parse_next_step(plan_text)
    backlog_open, backlog_err = parse_backlog_open_count(plan_text)
    errors: list[str] = []
    if next_err:
        errors.append(next_err)
    if backlog_err:
        errors.append(backlog_err)
    return {
        "next_step": next_step,
        "backlog_open_count": backlog_open,
        "errors": errors,
    }
