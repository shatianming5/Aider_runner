from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ContractHints:
    """Best-effort command hints extracted from the repo itself (no hardcoding)."""

    commands: list[str]
    anchors: list[str]


_FENCE_RE = re.compile(r"```[a-zA-Z0-9_-]*\n(?P<body>.*?)\n```", re.DOTALL)


def _iter_md_paths(repo: Path, *, max_files: int) -> list[Path]:
    repo = Path(repo).resolve()
    md_paths: list[Path] = []
    for pat in ("README*.md", "docs/**/*.md"):
        md_paths.extend(sorted(repo.glob(pat)))
    md_paths = [p for p in md_paths if p.is_file()]
    return md_paths[: int(max(1, max_files))]


def _tokenize_hint(cmd: str) -> list[str]:
    try:
        return shlex.split(cmd)
    except Exception:
        # Fallback tokenization; good enough for anchors.
        return [t for t in re.split(r"\s+", str(cmd or "").strip()) if t]


def _extract_anchors(hints: list[str]) -> list[str]:
    """Extract high-signal tokens we can use to audit "did you use doc hints?".

    This is intentionally heuristic and benchmark-agnostic.
    """
    skip_first = {
        "bash",
        "sh",
        "zsh",
        "python",
        "python3",
        "pip",
        "pip3",
        "uv",
        "conda",
        "poetry",
        "make",
        "npm",
        "node",
        "docker",
        "sudo",
    }
    skip_tokens = {
        "install",
        "uninstall",
        "run",
        "start",
        "stop",
        "setup",
        "build",
        "download",
        "clone",
        "create",
        "remove",
        "update",
        "upgrade",
        "requirements.txt",
        "requirements-ml.txt",
    }
    out: list[str] = []
    seen: set[str] = set()
    for raw in hints or []:
        tokens = _tokenize_hint(raw)
        if not tokens:
            continue
        # Prefer module invocations: `python -m pkg.module ...`
        if "-m" in tokens:
            try:
                mod = tokens[tokens.index("-m") + 1]
            except Exception:
                mod = ""
            mod = str(mod or "").strip()
            if mod and len(mod) >= 6:
                if mod not in seen:
                    seen.add(mod)
                    out.append(mod)
                pkg = mod.split(".", 1)[0].strip()
                if pkg and pkg not in seen and len(pkg) >= 5:
                    seen.add(pkg)
                    out.append(pkg)
                continue

        first = str(tokens[0] or "").strip()
        if not first:
            continue
        if first in skip_first:
            # If the first token is too generic, try to grab a later high-signal token.
            for t in tokens[1:6]:
                tt = str(t or "").strip()
                if not tt or tt.startswith("-"):
                    continue
                if tt.lower() in skip_tokens:
                    continue
                # Prefer dotted modules/binaries or long-ish names (avoid `install`, `run` etc).
                if "." in tt and len(tt) >= 6 and tt not in seen:
                    seen.add(tt)
                    out.append(tt)
                    break
                if re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{5,}", tt) and tt not in seen:
                    seen.add(tt)
                    out.append(tt)
                    break
            continue

        # Use the binary name when it is not generic.
        if re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{4,}", first) and first not in seen:
            seen.add(first)
            out.append(first)
            if "." in first:
                pkg = first.split(".", 1)[0].strip()
                if pkg and pkg not in seen and len(pkg) >= 5:
                    seen.add(pkg)
                    out.append(pkg)
    return out[:12]


def suggest_contract_hints(repo: Path, *, max_files: int = 8, max_candidates: int = 20) -> ContractHints:
    """Extract candidate evaluation/benchmark commands from repo docs.

    - Generic: only uses repo content; no benchmark-specific logic.
    - Best-effort: returns an empty list if nothing is found.
    """
    md_paths = _iter_md_paths(repo, max_files=max_files)

    interest_re = re.compile(
        r"(?i)("
        r"\\beval\\b|"
        r"evaluate|evaluation|"
        r"benchmark|leaderboard|quick\\s+start|"
        r"evalplus|lm_eval|miniwob|"
        r"pytest|inspect"
        r")"
    )
    seen: set[str] = set()
    candidates: list[str] = []

    for p in md_paths:
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        text = text[:200_000]
        for m in _FENCE_RE.finditer(text):
            body = (m.group("body") or "").strip()
            if not body:
                continue

            # Reconstruct multi-line shell commands that use `\` line continuations.
            block_cmds: list[str] = []
            cur = ""
            for raw in body.splitlines():
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                low = line.lower()
                if "| bash" in low or "| sh" in low:
                    continue

                if cur:
                    if cur.rstrip().endswith("\\"):
                        cur = cur.rstrip()[:-1].rstrip() + " " + line.lstrip()
                        continue
                    block_cmds.append(cur)
                    cur = ""

                # Skip orphan option lines.
                if line.startswith("-"):
                    continue
                cur = line

            if cur:
                block_cmds.append(cur)

            for cmd in block_cmds:
                if not interest_re.search(cmd):
                    continue
                if cmd in seen:
                    continue
                seen.add(cmd)
                candidates.append(cmd)
                if len(candidates) >= int(max(1, max_candidates)):
                    anchors = _extract_anchors(candidates)
                    return ContractHints(commands=candidates, anchors=anchors)

    anchors = _extract_anchors(candidates)
    return ContractHints(commands=candidates, anchors=anchors)
