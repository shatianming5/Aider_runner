from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Protocol
from urllib.error import URLError
from urllib.request import Request, urlopen


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())


def _read_json_object(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    return data


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class ChatBackend(Protocol):
    model_id: str

    def chat(self, messages: list[dict[str, Any]], *, max_new_tokens: int, temperature: float, top_p: float) -> str: ...


@dataclass
class EchoBackend:
    """A tiny dependency-free backend for smoke/debug/testing."""

    model_id: str = "echo"

    def chat(self, messages: list[dict[str, Any]], *, max_new_tokens: int, temperature: float, top_p: float) -> str:
        last = ""
        for m in messages or []:
            if isinstance(m, dict) and m.get("role") in ("user", "system", "assistant"):
                c = m.get("content")
                if isinstance(c, str):
                    last = c
        s = f"echo: {last}"
        # Keep output bounded even if caller passes huge input.
        return s[: int(max(1, max_new_tokens)) * 4]


def _lazy_import_hf():
    try:
        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Missing ML dependencies. Install with:\n"
            "  pip install -r requirements-ml.txt\n"
            "(or install torch/transformers manually for your platform)"
        ) from e
    return torch, AutoModelForCausalLM, AutoTokenizer


def _fallback_prompt(messages: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for m in messages or []:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role") or "user")
        c = m.get("content")
        content = c if isinstance(c, str) else json.dumps(c, ensure_ascii=False)
        lines.append(f"{role}: {content}")
    lines.append("assistant:")
    return "\n".join(lines) + "\n"


@dataclass
class HfBackend:
    model_id: str
    model_dir: Path
    device: str

    def __post_init__(self) -> None:
        torch, AutoModelForCausalLM, AutoTokenizer = _lazy_import_hf()
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir), use_fast=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            str(self.model_dir),
            torch_dtype=(torch.float16 if self.device in ("cuda", "mps") else torch.float32),
            low_cpu_mem_usage=True,
        )
        self._model.to(self.device)
        self._model.eval()

        if getattr(self._tokenizer, "pad_token", None) is None and getattr(self._tokenizer, "eos_token", None) is not None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

    @staticmethod
    def _pick_device(device: str) -> str:
        torch, _AutoModelForCausalLM, _AutoTokenizer = _lazy_import_hf()
        d = str(device or "auto").strip().lower() or "auto"
        if d == "auto":
            if bool(getattr(torch.cuda, "is_available", lambda: False)()):  # pragma: no cover
                return "cuda"
            mps = getattr(torch.backends, "mps", None)
            if mps is not None and bool(getattr(mps, "is_available", lambda: False)()):  # pragma: no cover
                return "mps"
            return "cpu"
        if d in ("cpu", "cuda", "mps"):
            return d
        raise ValueError(f"invalid --device: {device} (expected auto/cpu/cuda/mps)")

    def chat(self, messages: list[dict[str, Any]], *, max_new_tokens: int, temperature: float, top_p: float) -> str:
        torch = self._torch
        tokenizer = self._tokenizer
        model = self._model

        prompt = ""
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)  # type: ignore[attr-defined]
            except Exception:
                prompt = _fallback_prompt(messages)
        else:
            prompt = _fallback_prompt(messages)

        encoded = tokenizer(prompt, return_tensors="pt")
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        in_len = int(encoded["input_ids"].shape[-1])
        do_sample = float(temperature or 0.0) > 0.0

        with torch.no_grad():
            out = model.generate(
                **encoded,
                max_new_tokens=int(max(1, max_new_tokens)),
                do_sample=bool(do_sample),
                temperature=float(temperature or 0.0) if do_sample else None,
                top_p=float(top_p or 1.0) if do_sample else None,
                pad_token_id=getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", None),
            )
        tokens = out[0][in_len:]
        return tokenizer.decode(tokens, skip_special_tokens=True)


class _ServerState(Protocol):
    backend: ChatBackend
    model_id: str


class _Handler(BaseHTTPRequestHandler):
    server: _ServerState  # type: ignore[assignment]

    def _json(self, status: int, obj: dict[str, Any]) -> None:
        raw = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _read_body(self) -> tuple[dict[str, Any] | None, str | None]:
        try:
            n = int(self.headers.get("Content-Length") or "0")
        except Exception:
            n = 0
        if n <= 0:
            return None, "missing_body"
        raw = self.rfile.read(n)
        try:
            data = json.loads(raw.decode("utf-8", errors="replace"))
        except Exception as e:
            return None, f"invalid_json: {e}"
        if not isinstance(data, dict):
            return None, "body_not_object"
        return data, None

    def log_message(self, fmt: str, *args: Any) -> None:  # pragma: no cover
        # Avoid noisy default logging; stdout/stderr may be redirected to artifacts.
        return

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            return self._json(200, {"ok": True, "ts": _now_iso()})
        if self.path == "/v1/models":
            return self._json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": str(self.server.model_id),
                            "object": "model",
                            "created": int(time.time()),
                            "owned_by": "local",
                        }
                    ],
                },
            )
        return self._json(404, {"error": {"message": "not_found", "type": "not_found"}})

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/v1/chat/completions":
            body, err = self._read_body()
            if err:
                return self._json(400, {"error": {"message": err, "type": "invalid_request"}})
            messages = body.get("messages") if body else None
            if not isinstance(messages, list):
                return self._json(400, {"error": {"message": "messages must be a list", "type": "invalid_request"}})

            max_tokens = body.get("max_tokens") if body else None
            temperature = body.get("temperature") if body else None
            top_p = body.get("top_p") if body else None
            try:
                max_new_tokens = int(max_tokens) if max_tokens is not None else int(os.environ.get("AIDER_FSM_MAX_NEW_TOKENS") or 256)
            except Exception:
                max_new_tokens = 256
            try:
                temp = float(temperature) if temperature is not None else float(os.environ.get("AIDER_FSM_TEMPERATURE") or 0.0)
            except Exception:
                temp = 0.0
            try:
                tp = float(top_p) if top_p is not None else float(os.environ.get("AIDER_FSM_TOP_P") or 1.0)
            except Exception:
                tp = 1.0

            try:
                text = self.server.backend.chat(messages, max_new_tokens=max_new_tokens, temperature=temp, top_p=tp)
            except Exception as e:
                return self._json(500, {"error": {"message": f"inference_error: {e}", "type": "server_error"}})

            now = int(time.time())
            resp = {
                "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                "object": "chat.completion",
                "created": now,
                "model": str(self.server.model_id),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": str(text)},
                        "finish_reason": "stop",
                    }
                ],
            }
            return self._json(200, resp)

        if self.path == "/v1/completions":
            body, err = self._read_body()
            if err:
                return self._json(400, {"error": {"message": err, "type": "invalid_request"}})
            prompt = body.get("prompt") if body else None
            if not isinstance(prompt, str):
                return self._json(400, {"error": {"message": "prompt must be a string", "type": "invalid_request"}})
            messages = [{"role": "user", "content": prompt}]
            try:
                text = self.server.backend.chat(messages, max_new_tokens=256, temperature=0.0, top_p=1.0)
            except Exception as e:
                return self._json(500, {"error": {"message": f"inference_error: {e}", "type": "server_error"}})
            now = int(time.time())
            resp = {
                "id": f"cmpl-{uuid.uuid4().hex[:12]}",
                "object": "text_completion",
                "created": now,
                "model": str(self.server.model_id),
                "choices": [
                    {
                        "index": 0,
                        "text": str(text),
                        "finish_reason": "stop",
                    }
                ],
            }
            return self._json(200, resp)

        return self._json(404, {"error": {"message": "not_found", "type": "not_found"}})


def _http_get_json(url: str, *, timeout_seconds: float) -> tuple[dict[str, Any] | None, str | None]:
    try:
        req = Request(url, method="GET", headers={"Accept": "application/json"})
        with urlopen(req, timeout=float(timeout_seconds)) as resp:
            raw = resp.read()
        data = json.loads(raw.decode("utf-8", errors="replace"))
        if not isinstance(data, dict):
            return None, "not_object"
        return data, None
    except URLError as e:
        return None, str(e)
    except Exception as e:
        return None, str(e)


def _wait_for_ready(runtime_env_path: Path, *, timeout_seconds: float) -> tuple[dict[str, Any] | None, str | None]:
    started = time.monotonic()
    last_err = ""
    while True:
        if time.monotonic() - started > float(timeout_seconds):
            return None, f"timeout_waiting_for_ready: {last_err}"
        if runtime_env_path.exists():
            obj = _read_json_object(runtime_env_path)
            if obj and isinstance(obj.get("service"), dict) and obj["service"].get("health_url"):
                health_url = str(obj["service"]["health_url"])
                data, err = _http_get_json(health_url, timeout_seconds=1.0)
                if data and bool(data.get("ok")):
                    return obj, None
                last_err = f"health_not_ready: {err or ''}"
        time.sleep(0.1)


def _run_server(
    *,
    backend: ChatBackend,
    host: str,
    port: int,
    runtime_env_out: Path,
    model_dir: Path,
) -> int:
    srv = ThreadingHTTPServer((host, int(port)), _Handler)
    srv.backend = backend  # type: ignore[attr-defined]
    srv.model_id = str(getattr(backend, "model_id", "model"))  # type: ignore[attr-defined]

    actual_host, actual_port = srv.server_address[0], int(srv.server_address[1])
    run_id = str(os.environ.get("AIDER_FSM_RUN_ID") or "").strip()
    runtime_env = {
        "ts": _now_iso(),
        "run_id": run_id,
        "service": {
            "base_url": f"http://{actual_host}:{actual_port}",
            "health_url": f"http://{actual_host}:{actual_port}/health",
        },
        "inference": {
            "type": "openai_compat",
            "model_dir": str(model_dir.resolve()),
            "model": str(getattr(backend, "model_id", "")),
        },
        "paths": {"rollout_path": ".aider_fsm/rollout.json", "metrics_path": ".aider_fsm/metrics.json"},
        "pid": int(os.getpid()),
    }
    _write_json(runtime_env_out, runtime_env)

    # Graceful shutdown on SIGTERM/SIGINT.
    def _shutdown(_signum: int, _frame: Any) -> None:  # pragma: no cover
        try:
            srv.shutdown()
        except Exception:
            pass

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    try:
        srv.serve_forever(poll_interval=0.25)
    finally:
        try:
            srv.server_close()
        except Exception:
            pass
    return 0


def _cmd_start(args: argparse.Namespace) -> int:
    model_dir = Path(str(args.model_dir)).expanduser().resolve()
    runtime_env_out = Path(str(args.runtime_env_out)).expanduser()
    pid_file = Path(str(args.pid_file)).expanduser()
    log_file = str(args.log_file or "").strip()
    backend = str(args.backend or "hf").strip().lower() or "hf"

    pid_file.parent.mkdir(parents=True, exist_ok=True)
    runtime_env_out.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "runner.ml.serve_openai_compat",
        "run",
        "--backend",
        backend,
        "--model-dir",
        str(model_dir),
        "--host",
        str(args.host),
        "--port",
        str(int(args.port)),
        "--runtime-env-out",
        str(runtime_env_out),
    ]

    stdout = None
    stderr = None
    log_fp = None
    if log_file:
        p = Path(log_file).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        log_fp = p.open("ab")
        stdout = log_fp
        stderr = log_fp
    else:
        stdout = subprocess.DEVNULL
        stderr = subprocess.DEVNULL

    proc = subprocess.Popen(cmd, stdout=stdout, stderr=stderr, cwd=os.getcwd())
    pid_file.write_text(str(int(proc.pid)) + "\n", encoding="utf-8")

    try:
        _obj, err = _wait_for_ready(runtime_env_out, timeout_seconds=float(args.startup_timeout))
        if err:
            try:
                proc.terminate()
            except Exception:
                pass
            return 2
        return 0
    finally:
        if log_fp is not None:
            try:
                log_fp.close()
            except Exception:
                pass


def _cmd_stop(args: argparse.Namespace) -> int:
    pid_file = Path(str(args.pid_file)).expanduser()
    if not pid_file.exists():
        return 0
    try:
        pid = int(pid_file.read_text(encoding="utf-8", errors="replace").strip() or "0")
    except Exception:
        pid = 0
    if pid <= 0:
        try:
            pid_file.unlink()
        except Exception:
            pass
        return 0

    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        try:
            pid_file.unlink()
        except Exception:
            pass
        return 0
    except Exception:
        return 1

    started = time.monotonic()
    while True:
        if time.monotonic() - started > float(args.timeout_seconds):
            break
        try:
            os.kill(pid, 0)
            time.sleep(0.1)
        except ProcessLookupError:
            break
        except Exception:
            break

    try:
        pid_file.unlink()
    except Exception:
        pass
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    model_dir = Path(str(args.model_dir)).expanduser().resolve()
    backend = str(args.backend or "hf").strip().lower() or "hf"
    model_id = str(args.model_id or model_dir.name).strip() or model_dir.name

    if backend == "echo":
        b: ChatBackend = EchoBackend(model_id=model_id)
    elif backend == "hf":
        device = HfBackend._pick_device(str(args.device or "auto"))
        b = HfBackend(model_id=model_id, model_dir=model_dir, device=device)
    else:
        raise SystemExit(f"invalid --backend: {backend} (expected hf|echo)")

    return _run_server(
        backend=b,
        host=str(args.host),
        port=int(args.port),
        runtime_env_out=Path(str(args.runtime_env_out)).expanduser(),
        model_dir=model_dir,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Minimal OpenAI-compatible chat completion server for local HF model dirs.")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("start", help="start server in background and wait until ready")
    sp.add_argument("--backend", default="hf", choices=("hf", "echo"))
    sp.add_argument("--model-dir", required=True)
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--port", type=int, default=0, help="0 means auto-pick a free port")
    sp.add_argument("--runtime-env-out", required=True, help="path to write runtime_env.json")
    sp.add_argument("--pid-file", required=True)
    sp.add_argument("--log-file", default="")
    sp.add_argument("--startup-timeout", type=float, default=20.0)
    sp.set_defaults(func=_cmd_start)

    sp = sub.add_parser("stop", help="stop server by pid file")
    sp.add_argument("--pid-file", required=True)
    sp.add_argument("--timeout-seconds", type=float, default=10.0)
    sp.set_defaults(func=_cmd_stop)

    sp = sub.add_parser("run", help="run server in foreground (used by start)")
    sp.add_argument("--backend", default="hf", choices=("hf", "echo"))
    sp.add_argument("--model-dir", required=True)
    sp.add_argument("--model-id", default="")
    sp.add_argument("--device", default="auto", help="auto|cpu|cuda|mps (hf backend only)")
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--port", type=int, default=0)
    sp.add_argument("--runtime-env-out", required=True)
    sp.set_defaults(func=_cmd_run)

    args = p.parse_args(list(argv) if argv is not None else None)
    fn = getattr(args, "func", None)
    if fn is None:
        raise SystemExit(2)
    return int(fn(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

