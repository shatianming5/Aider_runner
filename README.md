# Aider-FSM Runner

一个最小可用的“闭环”执行器：用 Aider Python API 在同一进程里驱动一个有限状态机（FSM），循环执行：

1) 读 `PLAN.md` + 读仓库状态  
2) 更新计划（只允许改 `PLAN.md`）  
3) 执行 `Next` 的唯一一步（不允许改 `PLAN.md`）  
4) 验收：先 `/test`，再本地 subprocess 再跑一次 `TEST_CMD`  
5) 通过则标记 Done；失败则修复或改计划  

## 安装

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

注：`aider-chat` 在较新的 Python 版本上可能没有发布兼容包；如果你本机默认 `python3` 过新，请优先用 `python3.12` 创建虚拟环境。

## 环境变量

- `OPENAI_API_KEY`：必需
- `OPENAI_API_BASE`：可选（OpenAI 兼容服务时使用，例如自建/第三方 endpoint）

## 运行

在目标仓库根目录运行（建议是 git 仓库，便于 guard 回滚）：

```bash
python fsm_runner.py --repo . --goal "你的目标" --test-cmd "pytest -q"
```

### 部署 + Benchmark（pipeline.yml）

如果你希望把“测试→部署→跑 benchmark→检查 metrics”也纳入验收闭环，在目标 repo 里放一个 `pipeline.yml`，然后：

```bash
python fsm_runner.py --repo . --pipeline pipeline.yml --ensure-kind
```

说明：

- `pipeline.yml` 是**人类提供的部署/评测契约**；runner 会在执行/计划更新阶段自动回滚对它的修改
- 产物默认落在 `.aider_fsm/artifacts/<run_id>/`（可用 `--artifacts-dir` 覆盖）
- 如果 `pipeline.yml` 里配置了 `auth` 且需要交互登录，用 `--unattended guided`（默认 strict 会拒绝可疑交互命令以避免 hang）

可选参数：

- `--seed path/to/file`：可重复，用于把入口文件加入 Aider 上下文
- `--model gpt-4o-mini`
- `--max-iters 200`
- `--max-fix 10`
- `--plan-path PLAN.md`
- `--pipeline pipeline.yml`：启用部署/benchmark 流水线验收
- `--artifacts-dir <path>`：产物目录（默认：pipeline.artifacts.out_dir 或 `.aider_fsm/artifacts`）
- `--ensure-tools`：macOS 上自动安装/校验 `colima/docker/kubectl/helm/kind`
- `--ensure-kind`：确保本地 kind 集群存在（通用）
- `--kind-name <name>`：kind 集群名（默认：pipeline.tooling.kind_cluster_name 或 `kind`）
- `--kind-config <path>`：kind 配置文件（可选）
- `--unattended strict|guided`：无人值守模式（strict 默认更安全；guided 允许 auth 交互）
- `--full-quickstart`：针对 AIOpsLab 的本地 kind Quick Start 做一次性 preflight（建集群/生成 config.yml/建 venv）
- `--preflight-only`：只跑一次验收命令后退出（不进入 Aider FSM loop）

## 让 Aider 自己跑（最大程度自动化）

用 Aider 的 `/run` 来执行“装环境 + AIOpsLab QuickStart preflight”，推荐用这个 runbook 包一层（会解析并返回真实退出码）：

```bash
python3 tools/aider_runbook.py
```

## 测试

```bash
python -m pytest -q
```
