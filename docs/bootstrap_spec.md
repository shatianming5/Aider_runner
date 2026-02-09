# `.aider_fsm/bootstrap.yml` 规范（v1）

本文档：`docs/bootstrap_spec.md`

`bootstrap.yml` 定义 **repo-owned 的环境准备（environment setup）** 步骤，会在 pipeline 验证前运行。
其目标是让“单命令合同运行”尽量可复现，例如：创建 venv、安装依赖、预热缓存等。

注意事项：

- runner 会把 bootstrap 的证据写在 `.aider_fsm/artifacts/<run_id>/...` 下。
- bootstrap 命令与 pipeline/actions 命令使用同一套安全策略（denylist/allowlist、unattended 模式等）。
- 在 `--unattended strict` 模式下，疑似交互式命令会被阻止（避免无人值守卡死）。
- bootstrap 的定位是 **环境准备**（venv/依赖/构建缓存）。不要把 evaluation/test/benchmark 的实际运行（例如 `pytest`、benchmark CLI）
  放进 `bootstrap.yml`；这些应该放到 `pipeline.yml` 的对应 stage（尤其是 `evaluation`）中。
- 如果你在 `.aider_fsm/venv` 下创建 venv，建议所有安装都用 venv 解释器执行（例如 `.aider_fsm/venv/bin/python -m pip ...`），
  避免污染全局/用户 site-packages。

## 顶层字段（Top-level）

- `version`: 必须为 `1`
- `cmds`: shell 命令列表（可选；只做 env 注入时允许为空）
  - 别名：`steps`
- `env`: 作用于 bootstrap/pipeline/actions 的环境变量映射（可选）
  - 支持最小变量展开：`${VAR}` 与 `$VAR` 会从当前进程环境读取并替换
  - 使用 `$$` 表示字面量 `$`
- `workdir`: 工作目录（必须在 repo 内；默认 repo root）
- `timeout_seconds`: 单条命令超时（可选）
- `retries`: 每条命令的重试次数（可选；默认 0）

## Artifacts（bootstrap 会写哪些证据）

Bootstrap 会写：

- `bootstrap.yml`: 本次使用的 spec 快照
- `bootstrap_env.json`: 实际应用的 env 映射（敏感 key 会被脱敏）
- `bootstrap_summary.json`: ok/failed_index/total_results
- `bootstrap_cmdXX_tryYY_*`: 每条命令每次 attempt 的 stdout/stderr/result

## 示例：uv + venv（推荐）

这个模式会在 `.aider_fsm/venv` 下创建隔离 venv、安装依赖，并把 venv 的 `bin/` prepend 到 `PATH`，
让后续 pipeline stages 无需硬编码解释器路径：

```yaml
version: 1
cmds:
  - uv venv .aider_fsm/venv
  - uv pip install -r requirements.txt
env:
  PATH: ".aider_fsm/venv/bin:$PATH"
```

## 相关实现文件（代码指向）

- `runner/bootstrap.py`: bootstrap.yml 解析、env 展开、执行与 artifacts 落盘
- `runner/env_local.py`: stage-only 执行路径中对 bootstrap 的调用与 env 应用
- `runner/pipeline_verify.py`: 执行 pipeline stages 时的安全策略/超时/证据记录（bootstrap 也复用同一策略）
