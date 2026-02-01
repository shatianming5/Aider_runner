from __future__ import annotations

import argparse
import json
import platform
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


REQUIRED_TOOLS: tuple[tuple[str, str], ...] = (
    ("colima", "colima"),
    ("docker", "docker"),
    ("kubectl", "kubectl"),
    ("helm", "helm"),
    ("kind", "kind"),
)


def is_macos() -> bool:
    return sys.platform == "darwin"


def run(cmd: str, *, check: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        shell=True,
        text=True,
        check=check,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def which(cmd: str) -> str | None:
    return shutil.which(cmd)


def has_brew() -> bool:
    return which("brew") is not None


def missing_tools() -> list[tuple[str, str]]:
    return [(brew_formula, cmd) for brew_formula, cmd in REQUIRED_TOOLS if which(cmd) is None]


def brew_install(formula: str, *, dry_run: bool) -> None:
    if dry_run:
        return
    run(f"brew install {shlex.quote(formula)}", check=True)


def ensure_tools(*, dry_run: bool) -> None:
    if not is_macos():
        raise RuntimeError("bootstrap_macos.py only supports macOS.")
    if not has_brew():
        raise RuntimeError("Homebrew is required but `brew` was not found on PATH.")

    for t in missing_tools():
        brew_install(t[0], dry_run=dry_run)


def ensure_colima_started(*, cpu: int, memory_gb: int, disk_gb: int, dry_run: bool) -> None:
    if dry_run:
        return

    # Start is idempotent.
    run(
        " ".join(
            [
                "colima",
                "start",
                "--cpu",
                str(cpu),
                "--memory",
                str(memory_gb),
                "--disk",
                str(disk_gb),
            ]
        ),
        check=True,
    )

    # Ensure docker CLI uses the colima context when present.
    run("docker context use colima", check=False)


def ensure_docker_credential_helper(*, dry_run: bool) -> None:
    if dry_run:
        return

    cfg_path = Path.home() / ".docker" / "config.json"
    if not cfg_path.exists():
        return

    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return

    # Common when Docker Desktop was previously installed.
    if cfg.get("credsStore") == "desktop" and which("docker-credential-desktop") is None:
        if which("docker-credential-osxkeychain") is not None:
            cfg["credsStore"] = "osxkeychain"
            cfg_path.parent.mkdir(parents=True, exist_ok=True)
            cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def verify_runtime(*, dry_run: bool) -> None:
    if dry_run:
        return

    run("docker version", check=True)
    run("docker ps", check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Install & verify k8s toolchain on macOS (Homebrew + Colima)")
    parser.add_argument("--check-only", action="store_true", help="only report status; do not install/start")
    parser.add_argument("--cpu", type=int, default=4, help="colima cpu (default: 4)")
    parser.add_argument("--memory-gb", type=int, default=8, help="colima memory GB (default: 8)")
    parser.add_argument("--disk-gb", type=int, default=60, help="colima disk GB (default: 60)")
    args = parser.parse_args()

    if not is_macos():
        print("ERROR: this installer only supports macOS.", file=sys.stderr)
        return 2

    if not has_brew():
        print("ERROR: Homebrew not found. Install from https://brew.sh/ and re-run.", file=sys.stderr)
        return 2

    dry_run = bool(args.check_only)

    before = missing_tools()
    if dry_run:
        if before:
            print("MISSING:")
            for brew_formula, cmd in before:
                print(f"- {cmd} (brew: {brew_formula})")
            return 1
        print("OK: all required tools are present.")
        return 0

    ensure_tools(dry_run=False)
    ensure_colima_started(cpu=args.cpu, memory_gb=args.memory_gb, disk_gb=args.disk_gb, dry_run=False)
    ensure_docker_credential_helper(dry_run=False)
    verify_runtime(dry_run=False)

    after = missing_tools()
    if after:
        print("ERROR: tools still missing after install:")
        for brew_formula, cmd in after:
            print(f"- {cmd} (brew: {brew_formula})")
        return 1

    print("OK: toolchain installed and docker runtime is working.")
    print(f"platform: {platform.platform()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
