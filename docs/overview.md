# OpenCode-FSM Runner 概览

本文档：`docs/overview.md`

本仓库提供一个小而可审计的 **library**，用于执行并验收“目标仓库自带合同（repo-owned contract）”：

1. 读取目标仓库的 `pipeline.yml`（若缺失，可通过 OpenCode 生成 scaffold 合同）
2. 按合同选择性执行 stages（deploy → rollout → evaluation → benchmark）
3. 校验合同要求的 JSON 产物与 metrics，并写入可追溯的 artifacts 证据

设计目标：让你可以在 agent/训练脚本里，用统一的 `setup/rollout/evaluate` API 驱动任意仓库或 benchmark，runner 本身不写 benchmark-specific 的硬编码逻辑。

## 关键约束（为什么要这样设计）

- **Pipeline 是人类拥有（human-owned）**：runner 不接受模型对 `pipeline.yml` 的改写，避免合同被“悄悄篡改”导致不可审计。
- **验收是确定性的（deterministic）**：runner 只负责执行命令、收集 stdout/stderr、落盘证据、做结构化校验，不做“智能猜测”。

## 目标仓库侧需要/可能出现的文件

- `pipeline.yml`（可选）：验证合同（见 `docs/pipeline_spec.md`）。若缺失，`setup()` 可触发 scaffold。
- `.aider_fsm/bootstrap.yml`（可选）：repo-owned 环境准备（见 `docs/bootstrap_spec.md`）。
- `.aider_fsm/stages/*.sh`：各 stage 脚本（deploy/rollout/evaluation/benchmark 等），通常由 `pipeline.yml` 调用。
- `.aider_fsm/runtime_env.json`：deploy 输出的运行时连接信息（可选但推荐）。
- `.aider_fsm/rollout.json`：rollout 输出（可选，若启用 rollout）。
- `.aider_fsm/metrics.json`：evaluation/benchmark 输出的指标 JSON（可选，若启用 metrics 验证）。
- `.aider_fsm/artifacts/<run_id>/...`：runner 写入的可审计证据（每次运行一个 run_id）。

## 相关实现文件（代码指向）

- `runner_env.py`: 对外最小入口（`setup/EnvSession` 的 import alias）
- `runner/env.py`: `setup()` + `EnvSession`（对外 API 编排与重试/repair）
- `runner/env_local.py`: 本地执行模式的 env 打开、stage-only 调用与 bootstrap 运行
- `runner/pipeline_spec.py`: `pipeline.yml` 解析/规范化（v1）
- `runner/pipeline_verify.py`: 执行 pipeline stages、收集 artifacts、做结构化验收
- `runner/bootstrap.py`: `.aider_fsm/bootstrap.yml` 解析与执行
- `runner/security.py`: 命令安全策略（denylist/allowlist、hard deny 等）
- `runner/contract_provenance.py`: scaffold/repair 的 provenance（哪些合同文件被改动、来源是谁）
- `runner/scaffold_validation.py`: scaffold 合同最小可运行性校验

## 相关文档（路径）

- `docs/integration.md`
- `docs/env_api.md`
- `docs/pipeline_spec.md`
- `docs/bootstrap_spec.md`
- `docs/metrics_schema.md`
- `docs/security_model.md`
- `docs/verification.md`
