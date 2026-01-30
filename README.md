# AIOpsLab (Aider 自动部署版)

本仓库用于通过 **Aider** 自动化部署并运行 AIOpsLab 基准任务。下面是最小可跑通的流程。

## 环境要求

- Python 3.11+
- kubectl + Helm（用于部署集群工作负载）
- 可选：kind / Docker（本地集群）
- 已安装 aider（`pip install aider-chat` 或你自己的安装方式）

## 快速启动（Aider 自动部署）

> 下面步骤以当前仓库为工作目录。

### 1) 准备配置

```bash
cd /path/to/AIOpsLab
cp aiopslab/config.yml.example aiopslab/config.yml
```

根据你的集群修改：
- `k8s_host`
- `k8s_user`

如果你的模型服务 **不支持** `gpt-4-turbo-2024-04-09`，建议在 `config.yml` 中设置：
```yaml
qualitative_eval: false
```

### 2) 创建 Python 3.11 虚拟环境

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
```

### 3) 设置模型网关与 Key

```bash
export OPENAI_API_KEY=你的key
export OPENAI_BASE_URL=http://你的网关地址/v1
export OPENAI_API_BASE=http://你的网关地址/v1
```

### 4) 生成 Aider Prompt 并启动

```bash
cat > PROMPT_AIOpsLab.md <<'PROMPT'
AIDER_LOOP_CMD: OPENAI_BASE_URL=http://你的网关地址/v1 OPENAI_API_BASE=http://你的网关地址/v1 bash -lc "cd . && . .venv/bin/activate && python -m pip install -e . && python run_benchmark.py"
PROMPT

AIDER_WORKDIR=$(pwd) AIDER_YES_ALWAYS=1 aider --message-file PROMPT_AIOpsLab.md
```

运行完成后，会输出结果到 `benchmark_results.json`。

## 常见问题

### 1) `aiopslab-applications: is a directory`
Aider 误把子模块目录当成文件。请确保 `.aiderignore` 中包含：
```
aiopslab-applications/
```

### 2) `Package 'aiopslab' requires a different Python: 3.10.x`
说明你没有使用 Python 3.11 的虚拟环境。请确认：
```
. .venv/bin/activate
python -V   # 应该是 3.11.x
```

---

如果你需要“非 Aider”方式运行，可直接：
```bash
. .venv/bin/activate
python run_benchmark.py
```
