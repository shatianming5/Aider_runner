from __future__ import annotations

from pathlib import Path

from runner.contract_hints import suggest_contract_hints


def test_suggest_contract_hints_from_readme_code_fence(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "README.md").write_text(
        "\n".join(
            [
                "# Demo",
                "",
                "Run evaluation:",
                "```bash",
                "python -m evalplus.evaluate --dataset humanevalplus --model openai",
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )

    hints = suggest_contract_hints(repo)
    assert "python -m evalplus.evaluate --dataset humanevalplus --model openai" in hints.commands
    # Anchors should contain the module and its top-level package to make audits robust.
    assert "evalplus.evaluate" in hints.anchors
    assert "evalplus" in hints.anchors

