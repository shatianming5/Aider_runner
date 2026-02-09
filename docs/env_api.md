# 库 API：`setup()` → `rollout()` → `evaluate()`

本文档：`docs/env_api.md`

本仓库对外暴露一个很小的 **programmatic API**，用于在不写 benchmark-specific runner 代码的前提下，把目标仓库从 `setup` 驱动到 `deploy/rollout/evaluate` 的闭环。

推荐（也是唯一支持）的 library 用法是导入 `runner_env`：

```python
import runner_env

sess = runner_env.setup("https://github.com/<owner>/<repo>")  # 也可以是 HF dataset URL
sess.rollout(llm="my-remote-model-name", mode="smoke")        # 或本地 HF 模型目录路径
sess.evaluate(mode="smoke")                                   # 复用 session 的 llm；结束时 best-effort teardown
```

目标（中文）：

- 在 **不写 benchmark-specific 硬编码** 的前提下，让你可以在单个训练脚本里用 `setup/rollout/evaluate` 形式驱动任意 repo / benchmark / dataset。
- 最大化 OpenCode 的自主性：缺少 `pipeline.yml` 时由 OpenCode scaffold 合同；评测/测试命令尽量来自目标仓库的 README / docs / CI workflows（而不是 runner 手写启动逻辑）。
- 作为库（library）使用（硬约束）：只支持 `import runner_env`（推荐）或 `from runner import env as runner_env`（等价别名），并通过
  `sess = runner_env.setup(url)` → `sess.rollout(llm=...)` → `sess.evaluate()` 驱动闭环（`runner_env` 只是 import alias）。

---

## 核心概念：repo-owned contract（不硬编码怎么跑）

runner 自身保持 benchmark-agnostic。如何运行一个目标仓库由目标仓库通过合同声明：

- `pipeline.yml`（v1）：要跑哪些 stages（tests/deploy/rollout/evaluation/benchmark）
- `.aider_fsm/stages/*.sh`：各 stage 的脚本，产出 runner 期望的 JSON 文件：
  - `.aider_fsm/runtime_env.json`
  - `.aider_fsm/rollout.json` + `rollout.json.paths.samples_jsonl`
  - `.aider_fsm/metrics.json`（当 pipeline 配置 `required_keys` 时，必须包含 `ok` 与 `score`）

如果目标仓库缺少 `pipeline.yml`，`runner_env.setup()` 会通过 OpenCode 生成一个最小可运行的 scaffold 合同：

- `strict_opencode=True`（默认）：runner **不会** 预写/补丁合同文件；成功依赖 OpenCode（或 repo 预先存在文件）产出有效合同。
- `strict_opencode=False`（已弃用）：只为兼容保留。runner 仍然 **不会** prewrite/seed/fallback-write 合同文件。

scaffold/repair 的 provenance 会记录在：

- `<artifacts>/scaffold/scaffold_provenance.json`
- `<artifacts>/repair_*/repair_provenance.json`（发生 repair 时）

规范参考：`docs/pipeline_spec.md`、`docs/metrics_schema.md`。

---

## 公共 API（Public API）

只支持以下调用：

- `sess = runner_env.setup(target, ...) -> EnvSession`
- `sess.rollout(llm=..., ...)`
- `sess.evaluate(...)`

说明：

- `rollout()` 必须显式传入 `llm=...`。
- `evaluate()` 可以复用 session 在 `rollout()` 时配置的 llm；也允许为了单次调用方便传入 `llm=...`。
- `evaluate()` 结束时会 best-effort 做 teardown（对外不暴露 teardown API）。

### `llm` 参数（本地 vs 远端）

`llm` 支持两种形式：

- **本地 HF 模型目录**：传入一个存在的目录路径。
  - runner 设置：`AIDER_LLM_KIND=local_hf` 与 `AIDER_TRAINED_MODEL_DIR=/abs/path`
- **远端 model id/name**：任何其他非空字符串。
  - runner 设置：`AIDER_LLM_KIND=remote`、`AIDER_LLM_MODEL=<string>`，并同时设置 `OPENAI_MODEL=<string>`
  - endpoint/auth 从环境变量读取（OpenAI-compatible）：`OPENAI_API_KEY`，以及可选的 `OPENAI_API_BASE` / `OPENAI_BASE_URL`

runner 不会硬编码 endpoints、tokens、ports 或 benchmark flags。

---

## Rollout 合同（用于 RL/post-training）

为了支持通用的 post-training，rollout stage 需要产出：

- `.aider_fsm/rollout.json`（JSON object）
- `rollout.json.paths.samples_jsonl`：指向一个 JSONL 文件，每行应包含：
  - `prompt`（string）
  - `completion`（string）
  - `reward`（number）

### Rollout 验证（可选）

当你启用 rollout 合同验证（代码里 `require_samples=True`，或验证套件里 `--require-samples`），runner 会校验：

- `rollout.json.paths.samples_jsonl` 必须存在
- JSONL 每行必须能解析出合法的 `(prompt, completion, reward)`

额外的通用 sanity checks（所有 targets）：

- 若存在 `rollout.json.counts.errors` 且 `errors >= samples`，rollout 视为无效
- 若 **所有** sample 的 `completion` 都为空/空白，rollout 视为无效

对于 Hugging Face dataset snapshot（通过 `data/hf_manifest.json` + 含 `question/answer` 的测试 parquet 自动识别），还会额外约束：

- **最小样本数**：`min(AIDER_EVAL_LIMIT, test_rows)`（默认 smoke=8，full=64；除非你设置了 `AIDER_EVAL_LIMIT`）
- **Prompt 多样性**：防止单 prompt 机械重复
- **Prompt 锚定**：prompt 必须包含真实数据集 question 文本（防止合成无关任务）

---

## Evaluation：doc/CI hints（最大化自主性）

当目标仓库的 README/docs/CI 中存在明显“官方命令”（例如 `pytest -q`、`make test`、`npm test`），runner 会提取并注入：

- `AIDER_FSM_HINTS_JSON`：候选命令的 JSON list
- `AIDER_FSM_HINT_ANCHORS_JSON`：高信号 token 的 JSON list（用于证明命令确实被运行过）

此时 evaluation 期望至少运行一个 hinted/official 命令，并写出：

- `.aider_fsm/hints_used.json`（JSON object），包含：
  - `ok`: boolean（只有当某个 hinted/official 命令成功运行才为 true）
  - `used_anchors`: list（用于证明命令被使用的 anchors）
  - `commands`: 尝试过的命令（建议写，便于审计）
  - `reason`: 当 `ok=false` 时必须给出原因

scaffold 合同常用的通用 helper：

- `runner.generic_evaluation` 会通过 `runner.hints_exec.run_hints()` 执行 hints，并写出：
  - `.aider_fsm/hints_run.json`（调试追踪）
  - `.aider_fsm/hints_used.json`
  - `.aider_fsm/metrics.json`

补充说明：

- 对 `pytest` 风格 hints：即使退出码非 0，只要能解析出测试摘要，runner 仍将其视为“有效 evaluation 运行”，并用
  `score = passed / (passed + failed + errors)` 计算数值指标（保持 runner 通用但仍能产出 numeric score）。
- 若 **没有任何 hints** 且未设置 `AIDER_FSM_REQUIRE_HINTS`，`runner.generic_evaluation` 会 fallback：
  从 `rollout.json.paths.samples_jsonl` 计算 `score = average(reward)`。

离线偏好（可选）：

- 设置 `AIDER_FSM_PREFER_OFFLINE_HINTS=1`，倾向选择不依赖远端推理的命令（例如 `--samples ...`），并降低包含
  `--backend openai` 的 hints 优先级。

Shell 执行说明（可选）：

- 默认用 **非 login shell** 执行 hints（`bash -c ...`），这样 bootstrap 注入的 PATH 覆盖（例如 `.aider_fsm/venv/bin:$PATH`）更容易被保留。
- 若你明确需要 login shell：设置 `AIDER_FSM_HINT_LOGIN_SHELL=1`（使用 `bash -lc ...`）。

运行时超时覆盖（可选）：

- `AIDER_FSM_MAX_CMD_SECONDS=<int>`：运行时覆盖 `pipeline.security.max_cmd_seconds`（适合 full eval 很长的仓库）
- `AIDER_FSM_MAX_TOTAL_SECONDS=<int>`：运行时覆盖 `pipeline.security.max_total_seconds`

Token 上限（可选）：

- `AIDER_FSM_MAX_TOKENS=<int>`：设置内置 `runner.generic_rollout` 的 OpenAI-compatible request `max_tokens`

---

## 验证套件（单文档）

smoke/full-lite 的验证命令与证据清单见：`docs/verification.md`。

## 相关实现文件（代码指向）

- `runner_env.py`: 对外入口别名（`setup/EnvSession`）
- `runner/env.py`: `setup()` 与 `EnvSession`（对外编排与 repair 重试）
- `runner/env_local.py`: 本地执行模式的 open_env / deploy / rollout / evaluation
- `runner/generic_rollout.py`: 通用 rollout 脚本（常用于 scaffold 合同）
- `runner/generic_evaluation.py`: 通用 evaluation 脚本（hints 执行 + metrics 落盘）
- `runner/hints_exec.py`: hints 执行器（非交互 shell、超时、日志与 hints_used/hints_run）
- `runner/pipeline_verify.py`: pipeline stage 执行 + 产物/metrics 校验
