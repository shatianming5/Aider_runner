from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .train_lora import train_lora


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())


def _read_lines(path: Path) -> list[str]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    out: list[str] = []
    for line in raw.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        out.append(s)
    return out


@dataclass(frozen=True)
class BenchmarkRun:
    repo: str
    rc: int
    metrics: dict[str, Any] | None
    artifacts_dir: str
    stdout_tail: str
    stderr_tail: str


def _tail(s: str, n: int) -> str:
    t = str(s or "")
    return t if len(t) <= n else t[-n:]


def _run_opencode_run(
    *,
    repo: str,
    trained_model_dir: Path,
    opencode_model: str,
    artifacts_dir: Path,
    unattended: str,
    require_metrics: bool,
    repair_iters: int,
    env_overrides: dict[str, str],
) -> BenchmarkRun:
    args: list[str] = [
        sys.executable,
        "-m",
        "runner.opencode_run",
        "--repo",
        str(repo),
        "--trained-model-dir",
        str(trained_model_dir.resolve()),
        "--artifacts-dir",
        str(artifacts_dir.resolve()),
        "--unattended",
        str(unattended or "strict"),
        "--repair-iters",
        str(int(repair_iters or 0)),
        "--env-file",
        "",
    ]
    if str(opencode_model or "").strip():
        args.extend(["--model", str(opencode_model).strip()])
    if require_metrics:
        args.append("--require-metrics")
    else:
        args.append("--no-require-metrics")
    for k, v in (env_overrides or {}).items():
        args.extend(["--env", f"{str(k)}={str(v)}"])

    res = subprocess.run(args, check=False, capture_output=True, text=True)
    metrics: dict[str, Any] | None = None
    if res.stdout:
        try:
            parsed = json.loads(res.stdout.strip().splitlines()[-1])
            if isinstance(parsed, dict):
                metrics = parsed
        except Exception:
            metrics = None

    return BenchmarkRun(
        repo=str(repo),
        rc=int(res.returncode),
        metrics=metrics,
        artifacts_dir=str(artifacts_dir),
        stdout_tail=_tail(res.stdout or "", 2000),
        stderr_tail=_tail(res.stderr or "", 2000),
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Train a 0.5B-ish model and run benchmarks after each segment.")
    p.add_argument("--base-model", required=True, help="HF model id or local dir (first segment base)")
    p.add_argument("--out-root", required=True, help="output root directory for checkpoints and summary")
    p.add_argument("--benchmarks-file", required=True, help="text file: one repo/url per line")
    p.add_argument("--segments", type=int, default=1, help="how many train+eval segments to run (default: 1)")
    p.add_argument("--steps-per-segment", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", help="auto|cpu|cuda|mps")
    p.add_argument("--target-modules", default="", help="comma-separated; empty=auto infer")
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.05)

    p.add_argument("--opencode-model", default="", help="OpenCode model for scaffolding (provider/model)")
    p.add_argument("--unattended", choices=("strict", "guided"), default="strict")
    p.add_argument("--require-metrics", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--repair-iters", type=int, default=3, help="pass through to opencode_run (default: 3)")
    p.add_argument("--smoke-limit", type=int, default=20, help="pass AIDER_EVAL_LIMIT for smoke runs")
    p.add_argument("--full-after-last", action="store_true", help="also run a full pass after the final segment")
    args = p.parse_args(list(argv) if argv is not None else None)

    out_root = Path(str(args.out_root)).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    repos = _read_lines(Path(str(args.benchmarks_file)).expanduser())
    if not repos:
        raise SystemExit("benchmarks-file is empty")

    targets = [s.strip() for s in str(args.target_modules or "").split(",") if s.strip()]

    summary: dict[str, Any] = {
        "ts": _now_iso(),
        "base_model": str(args.base_model),
        "segments": int(args.segments),
        "steps_per_segment": int(args.steps_per_segment),
        "benchmarks": list(repos),
        "runs": [],
    }

    base = str(args.base_model)
    for seg_idx in range(int(args.segments)):
        ckpt_dir = out_root / f"ckpt_{seg_idx:03d}"
        train_res = train_lora(
            base_model=base,
            out_dir=ckpt_dir,
            steps=int(args.steps_per_segment),
            seq_len=int(args.seq_len),
            batch_size=int(args.batch_size),
            grad_accum=int(args.grad_accum),
            lr=float(args.lr),
            seed=int(args.seed),
            device=str(args.device),
            target_modules=targets or None,
            lora_r=int(args.lora_r),
            lora_alpha=int(args.lora_alpha),
            lora_dropout=float(args.lora_dropout),
        )
        base = str(train_res.model_dir)

        seg_entry: dict[str, Any] = {
            "segment": seg_idx,
            "model_dir": str(train_res.model_dir),
            "train": {
                "ok": bool(train_res.ok),
                "steps": int(train_res.steps),
                "wall_time_s": float(train_res.wall_time_s),
                "last_loss": train_res.last_loss,
            },
            "smoke": [],
            "full": [],
        }

        smoke_env = {"AIDER_EVAL_MODE": "smoke"}
        if int(args.smoke_limit or 0) > 0:
            smoke_env["AIDER_EVAL_LIMIT"] = str(int(args.smoke_limit))
        for repo in repos:
            run_dir = out_root / f"seg_{seg_idx:03d}" / "smoke" / Path(repo).name.replace("/", "_")
            run_dir.mkdir(parents=True, exist_ok=True)
            r = _run_opencode_run(
                repo=repo,
                trained_model_dir=Path(train_res.model_dir),
                opencode_model=str(args.opencode_model),
                artifacts_dir=run_dir,
                unattended=str(args.unattended),
                require_metrics=bool(args.require_metrics),
                repair_iters=int(args.repair_iters),
                env_overrides=smoke_env,
            )
            seg_entry["smoke"].append(r.__dict__)

        if bool(args.full_after_last) and seg_idx == int(args.segments) - 1:
            full_env = {"AIDER_EVAL_MODE": "full"}
            for repo in repos:
                run_dir = out_root / f"seg_{seg_idx:03d}" / "full" / Path(repo).name.replace("/", "_")
                run_dir.mkdir(parents=True, exist_ok=True)
                r = _run_opencode_run(
                    repo=repo,
                    trained_model_dir=Path(train_res.model_dir),
                    opencode_model=str(args.opencode_model),
                    artifacts_dir=run_dir,
                    unattended=str(args.unattended),
                    require_metrics=bool(args.require_metrics),
                    repair_iters=int(args.repair_iters),
                    env_overrides=full_env,
                )
                seg_entry["full"].append(r.__dict__)

        summary["runs"].append(seg_entry)
        (out_root / "train_and_benchmark_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
