from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CmdResult:
    """中文说明：
    - 含义：单条命令（shell）执行的结构化结果。
    - 内容：保存命令字符串、返回码、stdout/stderr 与是否超时；用于写入 artifacts 与诊断失败原因。
    - 可简略：否（基础数据结构，被多处依赖）。
    """

    cmd: str
    rc: int
    stdout: str
    stderr: str
    timed_out: bool = False


@dataclass(frozen=True)
class StageResult:
    """中文说明：
    - 含义：一个 stage（如 tests/deploy/rollout）整体执行结果。
    - 内容：包含该阶段所有命令（含重试）的 CmdResult 列表，并用 failed_index 标记首个失败/最终失败位置。
    - 可简略：否（核心验收结构；简化会影响日志与回溯能力）。
    """

    ok: bool
    results: list[CmdResult]
    failed_index: int | None = None


@dataclass(frozen=True)
class VerificationResult:
    """中文说明：
    - 含义：一次 pipeline 验收（verification）的汇总结果。
    - 内容：聚合 auth/tests/deploy/rollout/evaluation/benchmark 等阶段的 StageResult，并附带 metrics 读取与 required_keys 校验结果。
    - 可简略：否（runner 的核心输出；对外 API 与日志都依赖）。
    """

    ok: bool
    failed_stage: str | None
    bootstrap: StageResult | None = None
    auth: StageResult | None = None
    tests: StageResult | None = None
    deploy_setup: StageResult | None = None
    deploy_health: StageResult | None = None
    rollout: StageResult | None = None
    evaluation: StageResult | None = None
    benchmark: StageResult | None = None
    metrics_path: str | None = None
    metrics: dict[str, Any] | None = None
    metrics_errors: list[str] | None = None
