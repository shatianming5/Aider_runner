from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen


def _post_json(url: str, obj: dict) -> dict:
    raw = json.dumps(obj).encode("utf-8")
    req = Request(url, data=raw, method="POST", headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=5) as resp:
        out = resp.read()
    data = json.loads(out.decode("utf-8", errors="replace"))
    assert isinstance(data, dict)
    return data


def test_serve_openai_compat_echo_start_stop(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".aider_fsm").mkdir()

    runner_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(runner_root) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    runtime_env = repo / ".aider_fsm" / "runtime_env.json"
    pid_file = repo / ".aider_fsm" / "model_server.pid"
    log_file = repo / ".aider_fsm" / "server.log"

    start = subprocess.run(
        [
            sys.executable,
            "-m",
            "runner.ml.serve_openai_compat",
            "start",
            "--backend",
            "echo",
            "--model-dir",
            str(repo),
            "--host",
            "127.0.0.1",
            "--port",
            "0",
            "--runtime-env-out",
            str(runtime_env),
            "--pid-file",
            str(pid_file),
            "--log-file",
            str(log_file),
        ],
        cwd=str(repo),
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert start.returncode == 0
    assert runtime_env.exists()
    assert pid_file.exists()

    rt = json.loads(runtime_env.read_text(encoding="utf-8"))
    assert isinstance(rt, dict)
    base_url = rt.get("service", {}).get("base_url")
    assert isinstance(base_url, str) and base_url.startswith("http://127.0.0.1:")

    # Basic OpenAI-style call.
    res = _post_json(
        base_url + "/v1/chat/completions",
        {"model": "echo", "messages": [{"role": "user", "content": "hi"}], "temperature": 0},
    )
    assert res.get("object") == "chat.completion"
    choices = res.get("choices")
    assert isinstance(choices, list) and choices
    msg = choices[0].get("message", {})
    assert isinstance(msg, dict)
    assert str(msg.get("content") or "").startswith("echo:")

    stop = subprocess.run(
        [
            sys.executable,
            "-m",
            "runner.ml.serve_openai_compat",
            "stop",
            "--pid-file",
            str(pid_file),
        ],
        cwd=str(repo),
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert stop.returncode == 0

    # Best-effort: ensure the process is gone.
    time.sleep(0.2)
    assert not pid_file.exists()
