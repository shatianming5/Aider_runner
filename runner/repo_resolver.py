from __future__ import annotations

import os
import re
import subprocess
import shutil
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


_OWNER_REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_GITHUB_ARCHIVE_HOSTS = {"github.com", "www.github.com"}


def is_probably_repo_url(repo: str) -> bool:
    s = str(repo or "").strip()
    if not s:
        return False
    if s.startswith(("http://", "https://", "ssh://", "git@")):
        return True
    if s.endswith(".git") and ("/" in s or ":" in s):
        return True
    if _OWNER_REPO_RE.match(s):
        return True
    return False


def normalize_repo_url(repo: str) -> str:
    """Normalize shorthand forms into a git-cloneable URL."""
    s = str(repo or "").strip()
    if _OWNER_REPO_RE.match(s):
        # Default to GitHub for shorthand `owner/repo`.
        return f"https://github.com/{s}.git"
    return s


def _repo_slug(repo_url: str) -> str:
    s = repo_url.strip().rstrip("/")
    if s.startswith("git@") and ":" in s:
        s = s.split(":", 1)[1]
    if "://" in s:
        s = s.split("://", 1)[1]
    s = s.rstrip(".git")
    parts = [p for p in re.split(r"[/:]", s) if p]
    name = parts[-1] if parts else "repo"
    owner = parts[-2] if len(parts) >= 2 else "remote"
    slug = f"{owner}_{name}"
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", slug)
    return slug[:80]


def _parse_github_owner_repo(url: str) -> tuple[str, str] | None:
    s = str(url or "").strip()
    if not s:
        return None

    # https://github.com/<owner>/<repo>(.git)?
    m = re.match(r"^https?://([^/]+)/([^/]+)/([^/]+?)(?:\.git)?/?$", s)
    if m:
        host, owner, repo = m.group(1), m.group(2), m.group(3)
        if host.lower() in _GITHUB_ARCHIVE_HOSTS:
            return owner, repo

    # git@github.com:<owner>/<repo>(.git)?
    m = re.match(r"^git@([^:]+):([^/]+)/([^/]+?)(?:\.git)?$", s)
    if m:
        host, owner, repo = m.group(1), m.group(2), m.group(3)
        if host.lower() in _GITHUB_ARCHIVE_HOSTS:
            return owner, repo

    # ssh://git@github.com/<owner>/<repo>(.git)?
    m = re.match(r"^ssh://git@([^/]+)/([^/]+)/([^/]+?)(?:\.git)?/?$", s)
    if m:
        host, owner, repo = m.group(1), m.group(2), m.group(3)
        if host.lower() in _GITHUB_ARCHIVE_HOSTS:
            return owner, repo

    return None


def _download_file(url: str, *, out_path: Path, timeout_seconds: int = 60) -> tuple[bool, str]:
    try:
        req = Request(
            url,
            headers={
                "User-Agent": "opencode-fsm/1.0",
                "Accept": "application/octet-stream",
            },
            method="GET",
        )
        with urlopen(req, timeout=timeout_seconds) as resp:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("wb") as f:
                while True:
                    chunk = resp.read(1024 * 256)
                    if not chunk:
                        break
                    f.write(chunk)
        return True, ""
    except HTTPError as e:
        return False, f"HTTPError {getattr(e, 'code', '')}: {str(e)}"
    except URLError as e:
        return False, f"URLError: {str(e)}"
    except OSError as e:
        return False, f"OSError: {str(e)}"


def _extract_github_zip(zip_path: Path, *, extract_dir: Path, repo_name: str) -> Path:
    extract_dir.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
    except zipfile.BadZipFile as e:
        raise RuntimeError(f"invalid_zip: {e}") from e

    dirs = [p for p in extract_dir.iterdir() if p.is_dir()]
    if len(dirs) == 1:
        return dirs[0]

    prefix = f"{repo_name}-"
    candidates = [d for d in dirs if d.name.startswith(prefix)]
    if len(candidates) == 1:
        return candidates[0]

    raise RuntimeError(f"unexpected_zip_layout: dirs={[d.name for d in dirs]}")


def _archive_clone_github(
    *,
    owner: str,
    repo: str,
    dest: Path,
    env: dict[str, str],
    timeout_seconds: int = 60,
) -> tuple[bool, str]:
    """Best-effort fallback clone via GitHub archive zip.

    Returns (ok, detail). On success, the repo is extracted into dest.
    """
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    dest.parent.mkdir(parents=True, exist_ok=True)

    errors: list[str] = []
    zip_path = dest.parent / f"{dest.name}.zip"
    extract_dir = dest.parent / f"{dest.name}_extract"
    shutil.rmtree(extract_dir, ignore_errors=True)
    try:
        for branch in ("main", "master"):
            url = f"https://github.com/{owner}/{repo}/archive/refs/heads/{branch}.zip"
            ok, err = _download_file(url, out_path=zip_path, timeout_seconds=timeout_seconds)
            if not ok:
                errors.append(f"{url}: {err}")
                continue

            try:
                root = _extract_github_zip(zip_path, extract_dir=extract_dir, repo_name=repo)
            except Exception as e:
                errors.append(f"{url}: extract_failed: {e}")
                continue

            try:
                shutil.move(str(root), str(dest))
            except Exception as e:
                errors.append(f"{url}: move_failed: {e}")
                continue

            # Initialize git for revert guards (best effort).
            subprocess.run(["git", "-C", str(dest), "init"], env=env, check=False, capture_output=True, text=True)
            subprocess.run(["git", "-C", str(dest), "config", "user.name", "opencode-fsm"], env=env, check=False)
            subprocess.run(["git", "-C", str(dest), "config", "user.email", "opencode-fsm@example.com"], env=env, check=False)
            subprocess.run(["git", "-C", str(dest), "add", "-A"], env=env, check=False, capture_output=True, text=True)
            subprocess.run(
                ["git", "-C", str(dest), "commit", "-m", "init", "--no-gpg-sign"],
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )
            return True, f"archive_branch={branch}"
    finally:
        try:
            zip_path.unlink()
        except Exception:
            pass
        shutil.rmtree(extract_dir, ignore_errors=True)

    return False, "; ".join(errors[-5:])  # tail


@dataclass(frozen=True)
class PreparedRepo:
    repo: Path
    cloned_from: str | None = None


def prepare_repo(repo_arg: str, *, clones_dir: Path | None = None) -> PreparedRepo:
    """Return a usable local repo path.

    - If repo_arg is an existing local path -> use it.
    - If repo_arg looks like a git URL -> clone to clones_dir (or /tmp).
    """
    raw = str(repo_arg or "").strip()
    if not raw:
        raise ValueError("--repo is required")

    p = Path(raw).expanduser()
    if p.exists():
        return PreparedRepo(repo=p.resolve(), cloned_from=None)

    if not is_probably_repo_url(raw):
        raise FileNotFoundError(f"repo path not found: {raw}")

    url = normalize_repo_url(raw)
    base = clones_dir or (Path(tempfile.gettempdir()) / "aider_fsm_targets")
    base = base.expanduser().resolve()
    base.mkdir(parents=True, exist_ok=True)

    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    dest = base / f"{_repo_slug(url)}_{ts}"
    env = dict(os.environ)
    env.setdefault("GIT_TERMINAL_PROMPT", "0")

    # Depth=1 keeps it fast; users can re-run with a local clone if needed.
    cmd = ["git", "clone", "--depth", "1", url, str(dest)]
    proc = subprocess.run(cmd, text=True, capture_output=True, env=env)
    if proc.returncode != 0:
        # Clean up partial clones to avoid confusing fallbacks.
        shutil.rmtree(dest, ignore_errors=True)

        owner_repo = _parse_github_owner_repo(url)
        if owner_repo:
            owner, name = owner_repo
            ok, detail = _archive_clone_github(owner=owner, repo=name, dest=dest, env=env)
            if ok:
                return PreparedRepo(repo=dest.resolve(), cloned_from=url)

            raise RuntimeError(
                "git clone failed; GitHub archive fallback also failed\n"
                f"git_cmd: {' '.join(cmd)}\n"
                f"git_rc: {proc.returncode}\n"
                f"git_stdout: {proc.stdout[-2000:]}\n"
                f"git_stderr: {proc.stderr[-2000:]}\n"
                f"archive_detail: {detail}\n"
            )

        raise RuntimeError(
            "git clone failed\n"
            f"cmd: {' '.join(cmd)}\n"
            f"rc: {proc.returncode}\n"
            f"stdout: {proc.stdout[-2000:]}\n"
            f"stderr: {proc.stderr[-2000:]}\n"
        )

    # Make sure local commits (if any) won't fail due to missing identity.
    subprocess.run(["git", "-C", str(dest), "config", "user.name", "opencode-fsm"], env=env, check=False)
    subprocess.run(["git", "-C", str(dest), "config", "user.email", "opencode-fsm@example.com"], env=env, check=False)

    return PreparedRepo(repo=dest.resolve(), cloned_from=url)
