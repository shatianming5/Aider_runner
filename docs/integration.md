# 目标仓库接入指南（deploy + rollout + evaluation + benchmark）

本文档：`docs/integration.md`

runner 被刻意设计为 **generic**（不写 benchmark-specific 逻辑）。要让一个目标仓库可被 runner 驱动，需要通过目标仓库提供合同与产物来“接入”：

1. `pipeline.yml`：验证/执行合同（哪些 stage、怎么跑）
2. metrics JSON：由你的评测/基准产出（runner 只做结构化校验）
3. （可选）`.aider_fsm/bootstrap.yml`：repo-owned 环境准备（可复现、非交互）

## 推荐起步方式（从模板开始）

1. 把 `examples/pipeline.benchmark_skeleton.yml` 复制到目标仓库根目录，命名为 `pipeline.yml`
2. 按需补全（最常用字段）：
   - `deploy.setup_cmds` / `deploy.health_cmds`：如果你的 benchmark 需要先起服务/容器/进程
   - `rollout.run_cmds`：可选，用于产出轨迹/样本（写出 `.aider_fsm/rollout.json`）
   - `evaluation.run_cmds`：推荐，跑评测并写出 `evaluation.metrics_path`
   - `evaluation.required_keys`：你的 KPI 字段（runner 会校验这些 key 必须存在）
   - `benchmark.run_cmds`：可选，额外 benchmark；若配置 `benchmark.metrics_path` 同样会校验

相关规范：`docs/pipeline_spec.md`，示例：`examples/pipeline.example.yml`、`examples/pipeline.benchmark_skeleton.yml`。

## 环境准备（强烈建议）

若目标仓库依赖较多/环境复杂，建议提供 `.aider_fsm/bootstrap.yml` 来保证“单命令可复现”：

- 规范：`docs/bootstrap_spec.md`
- 示例：`examples/bootstrap.example.yml`

常见模式：在 `.aider_fsm/venv` 下创建虚拟环境并把其 `bin/` 目录 prepend 到 `PATH`，让后续 stages 不需要硬编码解释器路径。

## Metrics JSON 合同

runner 期望在 `evaluation.metrics_path`（推荐）或 `benchmark.metrics_path` 找到一个 JSON **object**。

- 若配置了 `evaluation.required_keys` / `benchmark.required_keys`，runner 会校验这些字段必须存在。
- 建议保持 metrics 小而稳定；完整原始日志写到 `.aider_fsm/artifacts/<run_id>/...`，便于审计与追溯。

推荐 schema：`docs/metrics_schema.md`。

## 运行方式（库 API）

```python
import runner_env

sess = runner_env.setup("/abs/path/to/target_repo")  # 也可以是 repo URL
sess.rollout(llm="deepseek-v3.2", mode="smoke", require_samples=True, repair_iters=0)
sess.evaluate(mode="smoke", repair_iters=0)
```

Artifacts 默认写入：`.aider_fsm/artifacts/<run_id>/`。

## 相关实现文件（代码指向）

- `runner_env.py`: 对外入口（`setup/EnvSession`）
- `runner/env.py`: `setup()`、`EnvSession.rollout()`、`EnvSession.evaluate()`
- `runner/pipeline_spec.py`: `pipeline.yml` schema（v1）
- `runner/pipeline_verify.py`: 执行 stages + 验收 metrics/产物 + 写入 artifacts
- `runner/bootstrap.py`: `.aider_fsm/bootstrap.yml` 的解析与执行

## 进一步阅读

- `docs/env_api.md`（如果你要在单个训练脚本里编排 `setup/rollout/evaluate`）
