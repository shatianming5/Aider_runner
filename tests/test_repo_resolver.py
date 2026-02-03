from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pytest

from runner.repo_resolver import is_probably_repo_url, prepare_repo


@pytest.mark.parametrize(
    ("s", "ok"),
    [
        ("https://github.com/evalplus/evalplus", True),
        ("https://github.com/evalplus/evalplus.git", True),
        ("git@github.com:evalplus/evalplus.git", True),
        ("ssh://git@github.com/evalplus/evalplus.git", True),
        ("evalplus/evalplus", True),
        (".", False),
        ("", False),
        (" /tmp/repo ", False),
    ],
)
def test_is_probably_repo_url(s: str, ok: bool):
    assert is_probably_repo_url(s) is ok


@dataclass(frozen=True)
class _FakeCompletedProcess:
    returncode: int
    stdout: str = ""
    stderr: str = ""


class _FakeHTTPResponse:
    def __init__(self, data: bytes):
        self._bio = io.BytesIO(data)

    def read(self, n: int = -1) -> bytes:
        return self._bio.read(n)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _make_zip_bytes(root_dir: str, files: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for rel, content in files.items():
            zf.writestr(f"{root_dir}/{rel}", content)
    return buf.getvalue()


def test_prepare_repo_github_archive_fallback_on_git_clone_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    url = "https://github.com/foo/bar.git"
    zip_bytes = _make_zip_bytes("bar-main", {"README.md": "hello\n"})

    def fake_run(cmd, *args, **kwargs):
        # Fail `git clone`, succeed everything else.
        if isinstance(cmd, list) and len(cmd) >= 2 and cmd[0] == "git" and cmd[1] == "clone":
            return _FakeCompletedProcess(returncode=1, stdout="", stderr="blocked")
        return _FakeCompletedProcess(returncode=0, stdout="", stderr="")

    def fake_urlopen(req, *args, **kwargs):
        target = str(getattr(req, "full_url", req))
        assert target.endswith("/archive/refs/heads/main.zip")
        return _FakeHTTPResponse(zip_bytes)

    monkeypatch.setattr("runner.repo_resolver.subprocess.run", fake_run)
    monkeypatch.setattr("runner.repo_resolver.urlopen", fake_urlopen)

    prepared = prepare_repo(url, clones_dir=tmp_path)
    assert prepared.cloned_from == url
    assert (prepared.repo / "README.md").read_text(encoding="utf-8") == "hello\n"
