# `pipeline.yml` 规范（v1）

本文档：`docs/pipeline_spec.md`

`pipeline.yml` 定义一个 **人类拥有（human-owned）的验证合同**。runner 会按固定顺序执行 stages：

1. `auth`（可选）
2. `tests`
3. `deploy.setup`（可选）
4. `deploy.health`（可选）
5. `rollout`（可选）
6. `evaluation`（可选）
7. `benchmark`（可选）
8. `evaluation.metrics` / `benchmark.metrics` 校验（可选）

示例文件：

- `examples/pipeline.example.yml`
- `examples/pipeline.benchmark_skeleton.yml`

## 顶层字段（Top-level）

- `version`: 必须为 `1`
- `security`: 命令安全策略与超时
- `tests`: 测试命令（除非你在 CLI 侧显式传入 `--test-cmd` 覆盖，否则为必需）
- `auth`: 登录/鉴权步骤（可选）
- `deploy`: 部署相关步骤（可选）
- `rollout`: post-training/RL rollout 步骤（可选）
- `evaluation`: 评测步骤 + metrics 校验（推荐）
- `benchmark`: 额外 benchmark 步骤 + metrics 校验（可选）
- `artifacts`: 本次运行 artifacts 输出目录（runner 证据落盘）

## `security`

- `mode`: `safe` 或 `system`
- `allowlist`: 可选的 regex 白名单；若设置，命令必须至少匹配一条 allowlist 才允许执行
- `denylist`: 可选的 regex 黑名单；命中即阻止
- `max_cmd_seconds`: 单条命令超时（可选）
- `max_total_seconds`: 一个 stage 的总超时（可选）

更详细说明：`docs/security_model.md`。

## `tests`

- `cmds`: shell 命令列表
- `timeout_seconds`, `retries`
- `env`: 环境变量映射
- `workdir`: 工作目录（必须在 repo 内）

## `auth`（可选）

- `steps` 或 `cmds`: shell 命令列表
- `interactive`: 若为 `true`，你必须用 `--unattended guided` 运行（允许交互）
- `timeout_seconds`, `retries`, `env`, `workdir`

## `deploy`（可选）

- `setup_cmds`, `health_cmds`, `teardown_cmds`
- `teardown_policy`: `always|on_success|on_failure|never`
- `timeout_seconds`, `retries`, `env`, `workdir`
- `kubectl_dump`: 可选，验收后做额外调试 dump

## `rollout`（可选）

- `run_cmds`
- `timeout_seconds`, `retries`, `env`, `workdir`

## `evaluation`（可选）

- `run_cmds`
- `metrics_path`: evaluation 产出的指标 JSON 文件路径（相对 repo）
- `required_keys`: metrics JSON 必须存在的 key 列表
- `timeout_seconds`, `retries`, `env`, `workdir`

约定（Convention）：

- 若 `required_keys` 包含 `ok`，runner 额外要求 `metrics.ok === true`（避免“占位成功” metrics）。
- 若设置了 `AIDER_FSM_REQUIRE_HINTS=1`，runner 额外要求 evaluation 后存在 `.aider_fsm/hints_used.json`：
  - 必须是 JSON object，且包含 `ok: true` 与非空的 `used_anchors` list
  - 若提供了 `AIDER_FSM_HINT_ANCHORS_JSON`（JSON array），`used_anchors` 必须至少包含其中一个 token
  - 若没有任何提示/官方命令能成功运行：应写 `ok: false` + 清晰的 `reason`，并以非零退出码结束

## `benchmark`（可选）

- `run_cmds`
- `metrics_path`: benchmark 产出的指标 JSON 文件路径（相对 repo）
- `required_keys`: metrics JSON 必须存在的 key 列表
- `timeout_seconds`, `retries`, `env`, `workdir`

约定（Convention）：

- 若 `required_keys` 包含 `ok`，runner 额外要求 `metrics.ok === true`。

推荐的 metrics schema：`docs/metrics_schema.md`。

## Tooling bootstrap（环境准备的放置位置）

runner 故意不内置平台相关的工具安装/集群创建逻辑。

- 若你需要“运行前环境准备/写配置/预热”等步骤：使用 `.aider_fsm/actions.yml`（见 `examples/actions.example.yml`）
- 若你需要 repo-owned、每次都要执行的环境准备：使用 `.aider_fsm/bootstrap.yml`（见 `docs/bootstrap_spec.md`）

## runner 注入的环境变量（每条 stage 命令都会有）

- `AIDER_FSM_STAGE`: stage 名称（例如 `tests`、`deploy_setup`、`rollout`、`evaluation`、`benchmark`）
- `AIDER_FSM_ARTIFACTS_DIR`: 当前 stage 的 artifacts 目录（绝对路径）
- `AIDER_FSM_REPO_ROOT`: repo 根目录（绝对路径）

## 相关实现文件（代码指向）

- `runner/pipeline_spec.py`: `pipeline.yml` 解析与规范化（v1 schema）
- `runner/pipeline_verify.py`: stage 执行、超时/重试、安全策略、metrics 验收与 artifacts 落盘
