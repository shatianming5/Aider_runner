import tools.bootstrap_macos as bm


def test_missing_tools_all_missing(monkeypatch):
    monkeypatch.setattr(bm, "which", lambda _cmd: None)
    missing = bm.missing_tools()
    assert [t[1] for t in missing] == [t[1] for t in bm.REQUIRED_TOOLS]


def test_missing_tools_none_missing(monkeypatch):
    monkeypatch.setattr(bm, "which", lambda _cmd: "/usr/bin/fake")
    assert bm.missing_tools() == []


def test_brew_install_dry_run_noop(monkeypatch):
    def _boom(*_a, **_k):
        raise AssertionError("run() should not be called in dry-run mode")

    monkeypatch.setattr(bm, "run", _boom)
    bm.brew_install("colima", dry_run=True)


def test_ensure_colima_started_dry_run_noop(monkeypatch):
    def _boom(*_a, **_k):
        raise AssertionError("run() should not be called in dry-run mode")

    monkeypatch.setattr(bm, "run", _boom)
    bm.ensure_colima_started(cpu=4, memory_gb=8, disk_gb=60, dry_run=True)


def test_ensure_docker_credential_helper_dry_run_noop():
    bm.ensure_docker_credential_helper(dry_run=True)
