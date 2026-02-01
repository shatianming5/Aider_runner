from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


SENTINEL = "__AIDER_FSM_RC:"
AIOPSLAB_GIT_URL = "https://github.com/microsoft/AIOpsLab.git"
DEFAULT_DOCKERIO_MIRROR = "https://docker.m.daocloud.io"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def find_aider_exe(root: Path) -> Path:
    which = shutil.which("aider")
    if which:
        return Path(which)

    candidates = [
        root / ".venv" / "bin" / "aider",
        root / ".venv312" / "bin" / "aider",
        root / ".venv313" / "bin" / "aider",
        root / ".venv314" / "bin" / "aider",
    ]
    candidates.extend(sorted(root.glob(".venv*/bin/aider")))
    for c in candidates:
        if c.exists():
            return c

    raise RuntimeError(
        "aider executable not found. Create a venv and install deps: `python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt`."
    )


def find_venv_python(aider_exe: Path) -> Path:
    py = aider_exe.parent / "python"
    if not py.exists():
        raise RuntimeError(f"Cannot find venv python next to aider: {aider_exe}")
    return py


def parse_last_rc(text: str) -> int | None:
    rc: int | None = None
    for line in text.splitlines():
        s = line.strip()
        if not s.startswith(SENTINEL):
            continue
        tail = s[len(SENTINEL) :].strip()
        try:
            rc = int(tail)
        except Exception:
            continue
    return rc


def _parse_dotenv(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip("\"'").strip()
        if k:
            out[k] = v
    return out


def resolve_model(*, cli_model: str | None, env_file: Path | None) -> str:
    if cli_model:
        m = cli_model.strip()
        if m:
            return m

    env_model = (os.environ.get("AIDER_MODEL") or "").strip()
    if env_model:
        return env_model

    if env_file and env_file.exists():
        env = _parse_dotenv(env_file)
        m = (env.get("AIDER_MODEL") or "").strip()
        if m:
            return m
        chat_model = (env.get("CHAT_MODEL") or "").strip()
        if chat_model:
            if "/" in chat_model:
                return chat_model
            # If OPENAI-compatible endpoint is configured, default to the openai provider prefix.
            if env.get("OPENAI_API_BASE") or env.get("AIDER_OPENAI_API_BASE") or env.get("OPENAI_API_KEY"):
                return f"openai/{chat_model}"
            return chat_model

    raise RuntimeError(
        "Unable to resolve aider model. Provide --model, set AIDER_MODEL, or set CHAT_MODEL/AIDER_MODEL in the env file."
    )


def resolve_mirror(*, cli_mirror: str | None) -> str | None:
    if cli_mirror is not None:
        v = cli_mirror.strip()
        if v.lower() in ("0", "false", "none", "off"):
            return None
        return v or DEFAULT_DOCKERIO_MIRROR

    v = (os.environ.get("KIND_DOCKERIO_MIRROR") or os.environ.get("AIDER_FSM_DOCKERIO_MIRROR") or "").strip()
    if v.lower() in ("0", "false", "none", "off"):
        return None
    return v or DEFAULT_DOCKERIO_MIRROR


def run_aider(*, aider_exe: Path, env_file: Path | None, model: str, message: str, cwd: Path) -> str:
    cmd: list[str] = [
        str(aider_exe),
        "--no-git",
        "--yes-always",
        "--no-check-model-accepts-settings",
        "--no-show-model-warnings",
        "--no-show-release-notes",
        "--model",
        model,
        "--message",
        message,
        "--exit",
    ]
    if env_file and env_file.exists():
        cmd.extend(["--env-file", str(env_file)])

    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=str(cwd))
    assert p.stdout is not None
    buf: list[str] = []
    for line in p.stdout:
        sys.stdout.write(line)
        buf.append(line)
    p.wait()
    return "".join(buf)


def run_step(*, aider_exe: Path, env_file: Path | None, model: str, shell_cmd: str, cwd: Path) -> int:
    # Aider does not propagate /run exit codes. We print a sentinel with `$?` and parse it.
    message = f"/run {shell_cmd}; echo {SENTINEL}$?"
    out = run_aider(aider_exe=aider_exe, env_file=env_file, model=model, message=message, cwd=cwd)
    rc = parse_last_rc(out)
    return 99 if rc is None else rc


def create_workspace(*, parent: Path | None) -> tuple[Path, Path]:
    if parent is None:
        root = Path(tempfile.mkdtemp(prefix="verify_aiopslab_")).resolve()
    else:
        parent.mkdir(parents=True, exist_ok=True)
        root = Path(tempfile.mkdtemp(prefix="verify_aiopslab_", dir=str(parent))).resolve()
    return root, (root / "AIOpsLab")


def _ensure_safe_delete_root(path: Path, *, root: Path) -> None:
    path = path.resolve()
    if path in (Path("/"), Path.home(), root):
        raise RuntimeError(f"Refusing to operate on unsafe path: {path}")


def _reset_cmd(level: str, *, aiopslab_repo: Path) -> str:
    cmds: list[str] = []

    # Always attempt to tear down kind context/cluster first.
    cmds.append("kind delete cluster || true")
    cmds.append("kubectl config delete-context kind-kind || true")

    if level in ("medium", "hard"):
        cmds.append("colima stop || true")
        cmds.append("colima delete -f || true")
        cmds.append("docker system prune -af || true")
        cmds.append("docker volume prune -f || true")

    if level == "hard":
        cmds.append("brew uninstall --ignore-dependencies kind helm kubectl docker colima || true")
        cmds.append("brew cleanup || true")

    # Clean local artifacts in the target repo if it exists.
    cmds.append(f"rm -rf {aiopslab_repo}/.venv_aiopslab {aiopslab_repo}/.aider_fsm {aiopslab_repo}/.aider.tags.cache.v4 || true")
    cmds.append(f"rm -f {aiopslab_repo}/aiopslab/config.yml || true")

    return " && ".join(cmds)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the full environment+QuickStart preflight via aider /run")
    parser.add_argument(
        "--aiopslab-repo",
        default="",
        help="path to AIOpsLab repo (optional; if empty, a temp workspace is created)",
    )
    parser.add_argument(
        "--fresh-clone",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="delete and re-clone AIOpsLab into the workspace before running (default: true)",
    )
    parser.add_argument(
        "--workspace-parent",
        default="",
        help="parent directory to create the fresh workspace under (default: system temp dir)",
    )
    parser.add_argument(
        "--keep-workspace",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="keep the fresh workspace after a successful run (default: false)",
    )
    parser.add_argument(
        "--clone-url",
        default=AIOPSLAB_GIT_URL,
        help=f"AIOpsLab git url (default: {AIOPSLAB_GIT_URL})",
    )
    parser.add_argument(
        "--reset",
        choices=("none", "soft", "medium", "hard"),
        default="hard",
        help="cleanup level before running (default: hard)",
    )
    parser.add_argument(
        "--model",
        default="",
        help="aider model name (default: resolved from env/AIDER_MODEL/CHAT_MODEL)",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="env file to pass to aider (default: .env under this repo)",
    )
    parser.add_argument(
        "--mirror",
        default="",
        help="docker.io mirror for kind/containerd when Docker Hub is blocked (default: env or https://docker.m.daocloud.io; use 'off' to disable)",
    )
    parser.add_argument("--skip-tests", action="store_true", help="skip `python -m pytest -q`")
    parser.add_argument("--skip-bootstrap", action="store_true", help="skip tools/bootstrap_macos.py")
    args = parser.parse_args()

    root = repo_root()
    aider_exe = find_aider_exe(root)
    venv_python = find_venv_python(aider_exe)
    env_file = (root / args.env_file).resolve() if args.env_file else None
    model = resolve_model(cli_model=args.model, env_file=env_file)
    mirror = resolve_mirror(cli_mirror=args.mirror if args.mirror else None)

    workspace_root: Path | None = None
    aiopslab_repo: Path
    if args.aiopslab_repo.strip():
        aiopslab_repo = Path(args.aiopslab_repo).expanduser().resolve()
        _ensure_safe_delete_root(aiopslab_repo, root=root)
    else:
        parent = Path(args.workspace_parent).expanduser().resolve() if args.workspace_parent.strip() else None
        workspace_root, aiopslab_repo = create_workspace(parent=parent)
        _ensure_safe_delete_root(workspace_root, root=root)

    if args.reset != "none":
        rc = run_step(
            aider_exe=aider_exe,
            env_file=env_file,
            model=model,
            shell_cmd=_reset_cmd(args.reset, aiopslab_repo=aiopslab_repo),
            cwd=root,
        )
        if rc != 0:
            return rc

        # Verify the reset produced a "missing tools" state when requested.
        if args.reset == "hard":
            rc = run_step(
                aider_exe=aider_exe,
                env_file=env_file,
                model=model,
                shell_cmd=f"cd {root} && {venv_python} tools/bootstrap_macos.py --check-only",
                cwd=root,
            )
            # check-only returns 1 when tools are missing (expected after hard reset)
            if rc not in (0, 1):
                return rc
            if rc == 0:
                print("ERROR: hard reset expected missing toolchain, but check-only reports OK.", file=sys.stderr)
                return 4

    if args.fresh_clone:
        clone_parent = aiopslab_repo.parent
        rc = run_step(
            aider_exe=aider_exe,
            env_file=env_file,
            model=model,
            shell_cmd=(
                f"rm -rf {aiopslab_repo} && "
                f"mkdir -p {clone_parent} && "
                f"git clone --recurse-submodules {args.clone_url} {aiopslab_repo}"
            ),
            cwd=root,
        )
        if rc != 0:
            return rc

    if not args.skip_tests:
        rc = run_step(
            aider_exe=aider_exe,
            env_file=env_file,
            model=model,
            shell_cmd=f"cd {root} && {venv_python} -m pytest -q",
            cwd=root,
        )
        if rc != 0:
            return rc

    if not args.skip_bootstrap:
        rc = run_step(
            aider_exe=aider_exe,
            env_file=env_file,
            model=model,
            shell_cmd=f"cd {root} && {venv_python} tools/bootstrap_macos.py",
            cwd=root,
        )
        if rc != 0:
            return rc

    mirror_export = ""
    if mirror:
        mirror_export = f"export KIND_DOCKERIO_MIRROR={mirror} && "
    rc = run_step(
        aider_exe=aider_exe,
        env_file=env_file,
        model=model,
        shell_cmd=(
            f"{mirror_export}"
            f"cd {root} && "
            f"{venv_python} fsm_runner.py --repo {aiopslab_repo} --ensure-tools --full-quickstart --preflight-only"
        ),
        cwd=root,
    )
    if rc != 0:
        if workspace_root:
            print(f"Workspace kept for inspection: {workspace_root}", file=sys.stderr)
        return rc

    if workspace_root and not args.keep_workspace:
        _ensure_safe_delete_root(workspace_root, root=root)
        run_step(
            aider_exe=aider_exe,
            env_file=env_file,
            model=model,
            shell_cmd=f"rm -rf {workspace_root}",
            cwd=root,
        )
    elif workspace_root:
        print(f"Workspace kept: {workspace_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
