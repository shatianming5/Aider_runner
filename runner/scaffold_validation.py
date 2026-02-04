from __future__ import annotations

from pathlib import Path

from .pipeline_spec import PipelineSpec


def validate_scaffolded_pipeline(
    pipeline: PipelineSpec,
    *,
    require_metrics: bool,
) -> list[str]:
    """中文说明：
    - 含义：校验 OpenCode scaffold 生成的 pipeline.yml 是否满足“最小可跑闭环”要求。
    - 内容：
      - 要求 `security.max_cmd_seconds` 合理设置（避免无限跑）。
      - 要求 deploy+rollout+evaluation 三个阶段都具备可执行命令。
      - 若 require_metrics=True，则要求 evaluation 产出 metrics JSON（含 score 与 ok）。
    - 可简略：否（这是“只给 URL 也能跑”的合同底线；若缺失会让后续流程不可用或不可审计）。
    """
    missing: list[str] = []

    if pipeline.security_max_cmd_seconds is None or int(pipeline.security_max_cmd_seconds) <= 0:
        missing.append("security.max_cmd_seconds")

    if not list(pipeline.deploy_setup_cmds or []):
        missing.append("deploy.setup_cmds")

    # health_cmds 允许为空（某些 repo 无健康检查），但 deploy 必须至少有 setup。

    if not list(pipeline.rollout_run_cmds or []):
        missing.append("rollout.run_cmds")

    if not list(pipeline.evaluation_run_cmds or []):
        missing.append("evaluation.run_cmds")

    if require_metrics:
        required_keys = {"score", "ok"}
        if not str(pipeline.evaluation_metrics_path or "").strip():
            missing.append("evaluation.metrics_path")
        if not required_keys.issubset(set(pipeline.evaluation_required_keys or [])):
            missing.append("evaluation.required_keys (missing: score, ok)")

    return missing


def validate_scaffolded_files(repo_root: Path) -> list[str]:
    """中文说明：
    - 含义：校验 scaffold 生成的关键脚本/文件是否存在。
    - 内容：要求 `.aider_fsm/stages/{deploy_setup,deploy_health,rollout,evaluation}.sh` 存在（可为 no-op，但必须可执行）。
    - 可简略：可能（属于更严格的契约校验；但能显著减少“pipeline 可解析但缺脚本”的空跑情况）。
    """
    repo_root = Path(repo_root).resolve()
    required = [
        repo_root / ".aider_fsm" / "stages" / "deploy_setup.sh",
        repo_root / ".aider_fsm" / "stages" / "deploy_health.sh",
        repo_root / ".aider_fsm" / "stages" / "rollout.sh",
        repo_root / ".aider_fsm" / "stages" / "evaluation.sh",
    ]
    missing: list[str] = []
    for p in required:
        if not p.exists():
            missing.append(str(p.relative_to(repo_root)))
    return missing
