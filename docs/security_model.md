# 安全模型（Security model）

本文档：`docs/security_model.md`

runner 在执行 `pipeline.yml`、`.aider_fsm/bootstrap.yml`、`.aider_fsm/actions.yml` 中的命令时，会套用一个偏保守的安全策略，目标是：

- 降低明显破坏性/越权命令被执行的风险
- 在无人值守（unattended）模式下尽量避免交互式卡死
- 仍然保留足够的可用性（能跑真实仓库/真实 benchmark）

## Hard deny（永远禁止）

一些高风险破坏模式会被 **永久阻止**，例如：

- `rm -rf /`、删除系统关键路径
- fork bomb 等明显破坏性行为

## `security.mode`

- `safe`（默认）：在 hard deny 之外追加默认 denylist（例如 `sudo`、`docker system prune`、`mkfs`、`dd`、`reboot` 等）
- `system`：更宽松（但仍保留 hard deny）

此外你还可以配置：

- `security.denylist`：额外的正则模式黑名单（命中即阻止）
- `security.allowlist`：白名单（如果设置了，则命令必须至少匹配一条 allowlist 才允许执行）

## Unattended（无人值守）模式

- `--unattended strict`（默认）：
  - 设置 `CI=1`、`GIT_TERMINAL_PROMPT=0` 等，尽量强制非交互
  - 阻止“看起来需要交互”的命令（例如缺少非交互参数的 `docker login`）
- `--unattended guided`：
  - 允许 `pipeline.auth.interactive: true` 的命令以交互方式运行（适合需要人工输入 token/OTP 的场景）

注意：无论是哪种模式，runner 都会记录 artifacts 与退出码，保证可审计性。

如果你在没有 `pipeline.yml` 的情况下运行（例如让 OpenCode scaffold 合同），runner 仍会套用默认的 `safe` deny 模式。

## 相关实现文件（代码指向）

- `runner/security.py`: hard deny / denylist / allowlist 的核心策略
- `runner/pipeline_verify.py`: 具体执行命令时的策略应用、超时与 artifacts 记录
- `runner/bootstrap.py`: bootstrap 命令执行（同样受安全策略与 unattended 约束）
- `runner/opencode_tooling.py`: strict_opencode 场景下的写文件/执行工具调用限制
