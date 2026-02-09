# 文档索引（中文）

本文档：`docs/README.md`

本目录下的文档均以“目标仓库自带合同（`pipeline.yml` + `.aider_fsm/`）”为核心，说明如何让 runner 在不写 benchmark-specific 逻辑的前提下完成部署、rollout、评测与审计留痕。

## 入门与集成

- `docs/overview.md`: 项目概览，核心约束与目标仓库侧需要提供的文件。
- `docs/integration.md`: 如何把一个目标仓库接入 runner（从 `examples/` 模板开始）。
- `docs/env_api.md`: 库 API（`setup()` → `rollout()` → `evaluate()`）的约定、输入输出与关键环境变量。

## 合同与产物规范

- `docs/pipeline_spec.md`: `pipeline.yml`（v1）规范，定义各 stage 的字段与执行顺序。
- `docs/bootstrap_spec.md`: `.aider_fsm/bootstrap.yml`（v1）规范，用于可复现的环境准备（venv/依赖/缓存）。
- `docs/metrics_schema.md`: 指标 JSON 的推荐 schema（让不同 repo 的结果可对齐比较）。

## 运行与验证

- `docs/verification.md`: smoke/full-lite 的验证脚本与“证据清单”（审计与复现用）。

## 安全与审计

- `docs/security_model.md`: 命令执行的安全模型（denylist/allowlist、安全/系统模式、unattended 模式）。

## 代码精简报告（静态）

- `docs/simplify_report.md`: 静态扫描生成的“可简化/可内联/可合并”候选清单（可读版）。
- `docs/simplify_report.csv`: 同一份报告的机器可读明细（CSV）。

说明：

- `docs/simplify_report.csv` 的列名目前保持英文（便于脚本稳定生成/解析），但 `reason` 字段已是中文；可直接用表格工具查看。
- 报告由脚本生成：`scripts/simplify_report.py`（依赖 `scripts/annotate_symbols.py`）。

