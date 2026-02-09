# 指标（metrics）schema（推荐）

本文档：`docs/metrics_schema.md`

runner 会对 `evaluation.metrics_path`（推荐）或 `benchmark.metrics_path` 指向的指标 JSON 做结构化校验：

- 当配置了 `evaluation.required_keys` 时：要求这些 key 必须存在于 `evaluation.metrics_path` 的 JSON object 中
- 当配置了 `benchmark.required_keys` 时：要求这些 key 必须存在于 `benchmark.metrics_path` 的 JSON object 中

为了让不同目标仓库的结果更易比较（evaluation + rollout + post-training），本文推荐一个尽量稳定的 metrics schema。

## 相关文件（目标仓库侧）

- `evaluation.metrics_path` 或 `benchmark.metrics_path`：指标 JSON（若你要开启 metrics 验证，这是必须的）
- Rollout 产物（用于 post-training/RL，强烈建议）：
  - `.aider_fsm/rollout.json`
  - `rollout.json.paths.samples_jsonl`：样本 JSONL 文件路径（通常在 `$AIDER_FSM_ARTIFACTS_DIR` 或 `.aider_fsm/` 下）

## 最小可用 metrics JSON（建议）

建议最小字段：

- `ok`: boolean（只有当“真实跑出了分数”才为 true；不要把“测试全过”等价为 ok）
- `score`: number（或你也可以用 `eval.score`，但建议同时保留顶层 `score` 便于通用处理）
- `ts`: ISO 时间戳字符串
- `run_id`: string（可直接复用 `AIDER_FSM_RUN_ID`）

## 按场景推荐字段

### Evaluation（评测）

- `eval.score`: 总分（主 KPI）
- `eval.details`: 可选，object/array，记录分项/子任务细节
- （可选但强烈建议，当使用 doc/CI hints 时）
  - `.aider_fsm/hints_used.json`：证明“确实运行过官方/提示命令”
  - `.aider_fsm/hints_run.json`：hints 执行的调试追踪

### Rollout（post-training / RL）

rollout 至少应产出：

- `.aider_fsm/rollout.json`（JSON object），建议包含：
  - `ok`: boolean
  - `paths.samples_jsonl`: string（指向一个 JSONL 文件）

Samples JSONL schema（每行一个 JSON object，benchmark-agnostic）：

- `prompt`: string
- `completion`: string
- `reward`: number
- `meta`: object（可选）

rollout 额外推荐字段（可选）：

- `n_episodes`: integer
- `success_rate`: number（范围 `[0, 1]`）
- `failures_by_type`: object（失败类型 -> 次数）
- `avg_latency_ms`: number

### Training（post-training）

- `train.ok`: boolean
- `train.steps`: integer
- `train.wall_time_s`: number
- `train.loss`: number（可选）

## 备注

- 建议保持 metrics JSON 小而稳定；完整日志写入 `.aider_fsm/artifacts/<run_id>/...`。
- 可用 `AIDER_FSM_*` 环境变量把 provenance（run_id、stage、artifacts_dir 等）写回到 metrics/rollout 里，便于审计与复现。

## 相关实现文件（代码指向）

- `runner/pipeline_verify.py`: metrics_path/required_keys 的校验逻辑
- `runner/pipeline_spec.py`: `pipeline.yml` 中 evaluation/benchmark 字段的解析与规范化
