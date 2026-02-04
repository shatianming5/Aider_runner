from __future__ import annotations

import os
from pathlib import Path

from runner.dotenv import load_dotenv


def test_load_dotenv_sets_missing_vars(tmp_path: Path, monkeypatch):
    """中文说明：
    - 含义：验证 `load_dotenv` 能写入“当前缺失”的环境变量，并忽略坏行/注释。
    - 内容：构造包含注释、export、空值、坏行的 .env 文件，删除现有 env 后加载并断言写入集合和值。
    - 可简略：否（覆盖解析容错与写入语义；属于 dotenv 行为的核心保障）。
    """
    p = tmp_path / ".env"
    p.write_text(
        "\n".join(
            [
                "# comment",
                "OPENAI_API_KEY=abc",
                'export FOO="bar baz"',
                "EMPTY=",
                "BADLINE",
                "",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("FOO", raising=False)
    monkeypatch.delenv("EMPTY", raising=False)

    written = load_dotenv(p)
    assert set(written) == {"OPENAI_API_KEY", "FOO", "EMPTY"}
    assert os.environ["OPENAI_API_KEY"] == "abc"
    assert os.environ["FOO"] == "bar baz"
    assert os.environ["EMPTY"] == ""


def test_load_dotenv_does_not_override_by_default(tmp_path: Path, monkeypatch):
    """中文说明：
    - 含义：验证 `override=False` 时不会覆盖已有环境变量。
    - 内容：预先 setenv FOO，再加载 .env 中的 FOO，断言 written 为空且 env 保持不变。
    - 可简略：是（典型开关语义测试）。
    """
    p = tmp_path / ".env"
    p.write_text("FOO=from_file\n", encoding="utf-8")
    monkeypatch.setenv("FOO", "preexisting")
    written = load_dotenv(p, override=False)
    assert written == []
    assert os.environ["FOO"] == "preexisting"


def test_load_dotenv_override_true(tmp_path: Path, monkeypatch):
    """中文说明：
    - 含义：验证 `override=True` 时允许覆盖已有环境变量。
    - 内容：预先 setenv FOO，再加载 .env 中的 FOO，断言 written 包含 FOO 且 env 被覆盖。
    - 可简略：是（典型开关语义测试）。
    """
    p = tmp_path / ".env"
    p.write_text("FOO=from_file\n", encoding="utf-8")
    monkeypatch.setenv("FOO", "preexisting")
    written = load_dotenv(p, override=True)
    assert written == ["FOO"]
    assert os.environ["FOO"] == "from_file"
