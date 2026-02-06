from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace


def test_train_and_benchmark_uses_env_api_in_order(monkeypatch, tmp_path: Path):
    from runner.ml import train_and_benchmark as mod

    targets_file = tmp_path / "targets.txt"
    targets = [
        "https://example.com/target-alpha",
        "https://example.com/target-beta",
    ]
    targets_file.write_text("\n".join(targets) + "\n", encoding="utf-8")

    out_root = tmp_path / "out"
    clones_dir = tmp_path / "clones"
    clones_dir.mkdir(parents=True, exist_ok=True)
    events: list[tuple[str, str, str]] = []
    train_calls: list[str] = []

    def fake_train_lora(**kwargs):
        seg = len(train_calls)
        model_dir = tmp_path / f"model_seg_{seg:03d}"
        model_dir.mkdir(parents=True, exist_ok=True)
        train_calls.append(str(kwargs.get("base_model") or ""))
        return SimpleNamespace(ok=True, steps=1, wall_time_s=0.1, last_loss=0.01, model_dir=str(model_dir))

    class FakeSession:
        def __init__(self, repo: str):
            self.repo = str(repo)

        def rollout(self, _llm, **kwargs):
            mode = str(kwargs.get("mode") or "")
            events.append(("rollout", self.repo, mode))
            return SimpleNamespace(ok=True, artifacts_dir=tmp_path, rollout_path=tmp_path / "rollout.json", verify=None)

        def evaluate(self, **kwargs):
            mode = str(kwargs.get("mode") or "")
            events.append(("evaluate", self.repo, mode))
            return SimpleNamespace(
                ok=True,
                artifacts_dir=tmp_path,
                metrics_path=tmp_path / "metrics.json",
                metrics={"ok": True, "score": 0.5},
                verify=None,
            )

    def fake_setup(cfg, **_kwargs):
        repo = str((cfg or {}).get("repo") or "")
        assert (cfg or {}).get("clones_dir") == clones_dir.resolve()
        events.append(("setup", repo, ""))
        return FakeSession(repo)

    def fake_teardown(*, session=None, **_kwargs):
        repo = str(getattr(session, "repo", "") or "")
        events.append(("teardown", repo, ""))
        return True

    monkeypatch.setattr(mod, "train_lora", fake_train_lora)
    monkeypatch.setattr(mod.env, "setup", fake_setup)
    monkeypatch.setattr(mod.env, "teardown", fake_teardown)

    rc = mod.main(
        [
            "--base-model",
            "base-model",
            "--out-root",
            str(out_root),
            "--targets-file",
            str(targets_file),
            "--clones-dir",
            str(clones_dir),
            "--segments",
            "2",
            "--steps-per-segment",
            "1",
            "--smoke-limit",
            "7",
            "--full-after-last",
            "--env-file",
            "",
        ]
    )
    assert rc == 0

    summary_path = out_root / "train_and_benchmark_summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary.get("targets") == targets
    runs = summary.get("runs")
    assert isinstance(runs, list) and len(runs) == 2

    seg0 = runs[0]
    seg1 = runs[1]
    assert len(seg0.get("smoke") or []) == 2
    assert len(seg0.get("full") or []) == 0
    assert len(seg1.get("smoke") or []) == 2
    assert len(seg1.get("full") or []) == 2

    for group in (seg0.get("smoke") or []) + (seg1.get("smoke") or []) + (seg1.get("full") or []):
        assert group.get("rc") == 0
        assert group.get("rollout_ok") is True
        assert group.get("evaluation_ok") is True
        metrics = group.get("metrics") or {}
        assert metrics.get("ok") is True
        assert group.get("mode") in ("smoke", "full")

    setup_sequence = [target for ev, target, _mode in events if ev == "setup"]
    expected_setup_sequence = [
        targets[0],
        targets[1],
        targets[0],
        targets[1],
        targets[0],
        targets[1],
    ]
    assert setup_sequence == expected_setup_sequence

    rollout_modes = [(target, mode) for ev, target, mode in events if ev == "rollout"]
    evaluate_modes = [(target, mode) for ev, target, mode in events if ev == "evaluate"]
    expected_modes = [
        (targets[0], "smoke"),
        (targets[1], "smoke"),
        (targets[0], "smoke"),
        (targets[1], "smoke"),
        (targets[0], "full"),
        (targets[1], "full"),
    ]
    assert rollout_modes == expected_modes
    assert evaluate_modes == expected_modes
