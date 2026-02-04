from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: str | Path | None, *, override: bool = False) -> list[str]:
    """中文说明：
    - 含义：从 dotenv 文件读取 KEY=VALUE 并写入 `os.environ`。
    - 内容：忽略空行/注释；支持 `export KEY=VALUE`；可选择是否覆盖已存在的环境变量；返回写入的 key 列表（不返回 value 以避免泄漏）。
    - 可简略：可能（可替换为第三方库 `python-dotenv`；但当前实现更可控/可审计）。

    ---

    English (original intent):
    Load KEY=VALUE pairs into os.environ.
    """
    if path is None:
        return []

    raw = str(path).strip()
    if not raw:
        return []

    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()

    if not p.exists() or not p.is_file():
        return []

    written: list[str] = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("export "):
            s = s[len("export ") :].lstrip()
        if "=" not in s:
            continue

        key, value = s.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue

        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]

        if override or key not in os.environ:
            os.environ[key] = value
            written.append(key)

    return written
