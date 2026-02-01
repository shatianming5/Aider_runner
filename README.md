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

可选参数：

- `--seed path/to/file`：可重复，用于把入口文件加入 Aider 上下文
- `--model gpt-4o-mini`
- `--max-iters 200`
- `--max-fix 10`
- `--plan-path PLAN.md`
- `--ensure-tools`：macOS 上自动安装/校验 `colima/docker/kubectl/helm/kind`
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
