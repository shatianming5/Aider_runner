from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


STATE_VERSION = 1
STDIO_TAIL_CHARS = 8000
PLAN_TAIL_CHARS = 20000
DEFAULT_DOCKERIO_MIRROR = "https://docker.m.daocloud.io"

_STEP_RE = re.compile(
    r"^\s*-\s*\[\s*([xX ])\s*\]\s*\(STEP_ID=([0-9]+)\)\s*(.*?)\s*$"
)

_KIND_IMAGE_RE = re.compile(r"^\s*image:\s*([^\s#]+)\s*$")


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())


def _tail(s: str, n: int) -> str:
    if len(s) <= n:
        return s
    return s[-n:]


def run_cmd(cmd: str, cwd: Path) -> tuple[int, str, str]:
    p = subprocess.run(
        cmd,
        cwd=str(cwd),
        shell=True,
        text=True,
        capture_output=True,
    )
    return p.returncode, _tail(p.stdout, STDIO_TAIL_CHARS), _tail(p.stderr, STDIO_TAIL_CHARS)


def _ensure_dirs(repo: Path) -> tuple[Path, Path, Path]:
    state_dir = repo / ".aider_fsm"
    logs_dir = state_dir / "logs"
    state_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)
    return state_dir, logs_dir, state_dir / "state.json"


def default_state(*, repo: Path, plan_rel: str, model: str, test_cmd: str) -> dict[str, Any]:
    return {
        "version": STATE_VERSION,
        "repo": str(repo),
        "plan_path": plan_rel,
        "model": model,
        "test_cmd": test_cmd,
        "iter_idx": 0,
        "fsm_state": "S0_BOOTSTRAP",
        "current_step_id": None,
        "current_step_text": None,
        "fix_attempts": 0,
        "last_test_rc": None,
        "last_exit_reason": None,
        "updated_at": _now_iso(),
    }


def load_state(path: Path, defaults: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return dict(defaults)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return dict(defaults)
    if not isinstance(data, dict):
        return dict(defaults)
    merged = dict(defaults)
    merged.update(data)
    merged["updated_at"] = _now_iso()
    return merged


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def plan_template(goal: str, test_cmd: str) -> str:
    goal = goal.strip() or "<fill goal>"
    return (
        "# PLAN\n"
        "\n"
        "## Goal\n"
        f"- {goal}\n"
        "\n"
        "## Acceptance\n"
        f"- [ ] TEST_CMD passes: `{test_cmd}`\n"
        "\n"
        "## Next (exactly ONE item)\n"
        "- [ ] (STEP_ID=001) 生成 Backlog：把目标拆成最小不可分的若干步（每步一次编辑+一次验收可闭合）\n"
        "\n"
        "## Backlog\n"
        "\n"
        "## Done\n"
        "- [x] (STEP_ID=000) 初始化计划文件\n"
        "\n"
        "## Notes\n"
        "- \n"
    )


def ensure_plan_file(plan_abs: Path, goal: str, test_cmd: str) -> None:
    if plan_abs.exists():
        return
    plan_abs.parent.mkdir(parents=True, exist_ok=True)
    plan_abs.write_text(plan_template(goal, test_cmd), encoding="utf-8")


def _extract_section_lines(lines: list[str], heading_prefix: str) -> list[str] | None:
    start = None
    for i, line in enumerate(lines):
        if line.strip().startswith(heading_prefix):
            start = i + 1
            break
    if start is None:
        return None
    end = len(lines)
    for j in range(start, len(lines)):
        if lines[j].strip().startswith("## "):
            end = j
            break
    return lines[start:end]


def _parse_step_lines(section_lines: list[str]) -> tuple[list[dict[str, Any]], list[str]]:
    steps: list[dict[str, Any]] = []
    bad: list[str] = []
    for line in section_lines:
        if not re.match(r"^\s*-\s*\[", line):
            continue
        m = _STEP_RE.match(line)
        if not m:
            bad.append(line.strip())
            continue
        checked = m.group(1).lower() == "x"
        steps.append({"id": m.group(2), "text": m.group(3).strip(), "checked": checked})
    return steps, bad


def find_duplicate_step_ids(plan_text: str) -> list[str]:
    lines = plan_text.splitlines()
    ids: list[str] = []
    for heading in ("## Next", "## Backlog", "## Done"):
        section = _extract_section_lines(lines, heading)
        if section is None:
            continue
        steps, _bad = _parse_step_lines(section)
        ids.extend([s["id"] for s in steps])
    counts: dict[str, int] = {}
    for sid in ids:
        counts[sid] = counts.get(sid, 0) + 1
    return [sid for sid, c in counts.items() if c > 1]


def parse_next_step(plan_text: str) -> tuple[dict[str, str] | None, str | None]:
    lines = plan_text.splitlines()
    section = _extract_section_lines(lines, "## Next")
    if section is None:
        return None, "missing_next_section"
    steps, bad = _parse_step_lines(section)
    if bad:
        return None, "bad_next_line"
    if len(steps) == 0:
        return None, None
    if len(steps) != 1:
        return None, "next_count_not_one"
    if steps[0]["checked"]:
        return None, "next_is_checked"
    dups = find_duplicate_step_ids(plan_text)
    if dups:
        return None, "duplicate_step_id"
    return {"id": steps[0]["id"], "text": steps[0]["text"]}, None


def parse_backlog_open_count(plan_text: str) -> tuple[int, str | None]:
    lines = plan_text.splitlines()
    section = _extract_section_lines(lines, "## Backlog")
    if section is None:
        return 0, "missing_backlog_section"
    steps, bad = _parse_step_lines(section)
    if bad:
        return 0, "bad_backlog_line"
    return sum(1 for s in steps if not s["checked"]), None


def parse_plan(plan_text: str) -> dict[str, Any]:
    next_step, next_err = parse_next_step(plan_text)
    backlog_open, backlog_err = parse_backlog_open_count(plan_text)
    errors: list[str] = []
    if next_err:
        errors.append(next_err)
    if backlog_err:
        errors.append(backlog_err)
    return {
        "next_step": next_step,
        "backlog_open_count": backlog_open,
        "errors": errors,
    }


def build_snapshot(repo: Path, plan_abs: Path) -> tuple[dict[str, Any], str]:
    plan_text = ""
    if plan_abs.exists():
        plan_text = plan_abs.read_text(encoding="utf-8", errors="replace")
    plan_text = _tail(plan_text, PLAN_TAIL_CHARS)

    state_text = ""
    state_path = repo / ".aider_fsm" / "state.json"
    if state_path.exists():
        state_text = state_path.read_text(encoding="utf-8", errors="replace")
    state_text = _tail(state_text, PLAN_TAIL_CHARS)

    rc1, gs, ge = run_cmd("git status --porcelain", repo)
    rc2, diff, de = run_cmd("git diff --stat", repo)
    rc3, names, ne = run_cmd("git diff --name-only", repo)

    env_probe = {
        "pyproject.toml": (repo / "pyproject.toml").exists(),
        "package.json": (repo / "package.json").exists(),
        "pytest.ini": (repo / "pytest.ini").exists(),
        "requirements.txt": (repo / "requirements.txt").exists(),
        "setup.cfg": (repo / "setup.cfg").exists(),
    }

    snapshot = {
        "repo": str(repo),
        "plan_path": str(plan_abs),
        "env_probe": env_probe,
        "git": {
            "status_rc": rc1,
            "status": gs,
            "status_err": ge,
            "diff_stat_rc": rc2,
            "diff_stat": diff,
            "diff_stat_err": de,
            "diff_names_rc": rc3,
            "diff_names": names,
            "diff_names_err": ne,
        },
        "plan_md": plan_text,
        "state_json": state_text,
    }

    env_lines = "\n".join([f"- {k}: {'yes' if v else 'no'}" for k, v in env_probe.items()])
    snapshot_text = (
        "[SNAPSHOT]\n"
        f"repo: {repo}\n"
        f"plan_path: {plan_abs}\n"
        "env_probe:\n"
        f"{env_lines}\n"
        "git_status_porcelain:\n"
        f"{gs}\n"
        "git_diff_stat:\n"
        f"{diff}\n"
        "changed_files:\n"
        f"{names}\n"
        "state_json:\n"
        f"{state_text}\n"
        "plan_md:\n"
        f"{plan_text}\n"
    )
    return snapshot, snapshot_text


def get_git_changed_files(repo: Path) -> list[str] | None:
    rc, out, _err = run_cmd("git diff --name-only", repo)
    if rc != 0:
        return None
    return [line.strip() for line in out.splitlines() if line.strip()]


def git_checkout(repo: Path, paths: list[str]) -> tuple[int, str, str]:
    if not paths:
        return 0, "", ""
    cmd = "git checkout -- " + " ".join(shlex.quote(p) for p in paths)
    return run_cmd(cmd, repo)


def non_plan_changes(changed_files: list[str], plan_rel: str) -> list[str]:
    return [p for p in changed_files if p != plan_rel]


def create_coder(*, model_name: str, fnames: list[str]):
    try:
        from aider.coders import Coder
        from aider.io import InputOutput
        from aider.models import Model
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Failed to import aider. Install deps with `pip install -r requirements.txt`."
        ) from e

    io = InputOutput(yes=True)
    model = Model(model_name)
    return Coder.create(main_model=model, fnames=fnames, io=io)


def _run_py_script(script: Path, args: list[str], *, cwd: Path) -> tuple[int, str, str]:
    cmd = [sys.executable, str(script), *args]
    p = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True)
    return p.returncode, _tail(p.stdout, STDIO_TAIL_CHARS), _tail(p.stderr, STDIO_TAIL_CHARS)


def _find_python312() -> str | None:
    for candidate in ("python3.12", "/opt/homebrew/bin/python3.12"):
        if "/" in candidate:
            if Path(candidate).exists():
                return candidate
            continue
        path = shutil.which(candidate)
        if path:
            return path
    return None


def _dockerhub_registry_reachable(repo: Path) -> bool:
    rc, _out, _err = run_cmd("curl -I -sS --max-time 5 https://registry-1.docker.io/v2/ >/dev/null", repo)
    return rc == 0


def _dockerio_mirror_url() -> str | None:
    v = (os.environ.get("KIND_DOCKERIO_MIRROR") or os.environ.get("AIDER_FSM_DOCKERIO_MIRROR") or "").strip()
    if v.lower() in ("0", "false", "none", "off"):
        return None
    return v or DEFAULT_DOCKERIO_MIRROR


def _strip_url_scheme(url: str) -> str:
    return re.sub(r"^https?://", "", url.strip())


def _dockerhub_mirror_ref(image: str, mirror_host: str) -> str | None:
    image = image.strip()
    if not image:
        return None

    # If the image already specifies a non-docker.io registry, don't rewrite.
    first = image.split("/", 1)[0]
    if "." in first or ":" in first:
        if first == "docker.io" and "/" in image:
            image = image.split("/", 1)[1]
        else:
            return None

    if "/" not in image:
        image = f"library/{image}"

    mirror_host = mirror_host.strip().rstrip("/")
    return f"{mirror_host}/{image}"


def _docker_image_exists(repo: Path, image: str) -> bool:
    rc, _out, _err = run_cmd(f"docker image inspect {shlex.quote(image)} >/dev/null 2>&1", repo)
    return rc == 0


def _docker_pull(repo: Path, image: str) -> bool:
    rc, _out, _err = run_cmd(f"docker pull {shlex.quote(image)}", repo)
    return rc == 0


def _dockerfile_base_image(dockerfile: Path) -> str | None:
    for raw in dockerfile.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.upper().startswith("FROM "):
            return line.split(None, 1)[1].strip()
    return None


def _is_aiopslab_kind_node_image(image: str) -> bool:
    return bool(re.search(r"/aiopslab-kind-(arm|x86):latest$", image))


def _ensure_aiopslab_kind_node_image(repo: Path, image: str, *, mirror_url: str | None) -> None:
    dockerfile = repo / "kind" / "Dockerfile"
    if not dockerfile.exists():
        raise RuntimeError("AIOpsLab kind image build requires kind/Dockerfile but it was not found.")

    kind_context = repo / "kind"

    base = _dockerfile_base_image(dockerfile)
    if not base:
        raise RuntimeError("Failed to determine base image from kind/Dockerfile.")

    if not _docker_image_exists(repo, base):
        if not _docker_pull(repo, base):
            if not mirror_url:
                raise RuntimeError("Docker Hub is unreachable and KIND_DOCKERIO_MIRROR is disabled.")
            mirror_host = _strip_url_scheme(mirror_url)
            mirror_ref = _dockerhub_mirror_ref(base, mirror_host)
            if not mirror_ref:
                raise RuntimeError(f"Cannot mirror-pull non-dockerhub base image: {base}")
            if not _docker_pull(repo, mirror_ref):
                raise RuntimeError(f"Failed to pull base image from mirror: {mirror_ref}")
            run_cmd(f"docker tag {shlex.quote(mirror_ref)} {shlex.quote(base)}", repo)

    rc, out, err = run_cmd(
        " ".join(
            [
                "docker",
                "build",
                "-t",
                shlex.quote(image),
                "-f",
                shlex.quote(str(dockerfile)),
                shlex.quote(str(kind_context)),
            ]
        ),
        repo,
    )
    if rc != 0:
        raise RuntimeError(f"Failed to build kind node image {image}:\n{out}\n{err}")


def _read_kind_node_images(cfg: Path) -> list[str]:
    images: list[str] = []
    for line in cfg.read_text(encoding="utf-8", errors="replace").splitlines():
        m = _KIND_IMAGE_RE.match(line)
        if not m:
            continue
        img = m.group(1).strip()
        if img and img not in images:
            images.append(img)
    return images


def _write_kind_config_with_dockerio_mirror(cfg: Path, mirror_url: str, out_dir: Path) -> Path:
    mirror_url = mirror_url.strip()
    if not mirror_url:
        return cfg

    text = cfg.read_text(encoding="utf-8", errors="replace")
    if "containerdConfigPatches" in text:
        return cfg

    lines = text.splitlines(True)
    insert_at = None
    for i, line in enumerate(lines):
        if line.strip().startswith("apiVersion:"):
            insert_at = i + 1
            break
    if insert_at is None:
        insert_at = 0

    block = (
        "containerdConfigPatches:\n"
        "  - |-\n"
        "    [plugins.\"io.containerd.grpc.v1.cri\".registry.mirrors.\"docker.io\"]\n"
        f"      endpoint = [\"{mirror_url}\"]\n"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{cfg.stem}-mirror{cfg.suffix}"
    out_path.write_text("".join(lines[:insert_at]) + block + "".join(lines[insert_at:]), encoding="utf-8")
    return out_path


def _ensure_kind_can_pull_image(repo: Path, image: str) -> None:
    name = "aider-fsm-pulltest"
    run_cmd(f"kubectl delete pod {shlex.quote(name)} --ignore-not-found", repo)
    run_cmd(
        " ".join(
            [
                "kubectl",
                "run",
                shlex.quote(name),
                f"--image={shlex.quote(image)}",
                "--restart=Never",
                "--command",
                "--",
                "sleep",
                "3600",
            ]
        ),
        repo,
    )
    rc, out, err = run_cmd(f"kubectl wait --for=condition=Ready pod/{shlex.quote(name)} --timeout=120s", repo)
    run_cmd(f"kubectl delete pod {shlex.quote(name)} --ignore-not-found --wait=false", repo)
    if rc != 0:
        raise RuntimeError(f"Kind cannot pull images from docker.io (or mirror):\n{out}\n{err}")


def _ensure_aiopslab_config(repo: Path) -> None:
    example = repo / "aiopslab" / "config.yml.example"
    cfg = repo / "aiopslab" / "config.yml"
    if cfg.exists():
        return
    if not example.exists():
        raise RuntimeError("Full QuickStart requires aiopslab/config.yml.example but it was not found.")

    text = example.read_text(encoding="utf-8", errors="replace").splitlines(True)
    user = os.environ.get("USER") or os.environ.get("LOGNAME") or "user"
    out: list[str] = []
    for line in text:
        if line.lstrip().startswith("k8s_host:"):
            out.append("k8s_host: kind\n")
        elif line.lstrip().startswith("k8s_user:"):
            out.append(f"k8s_user: {user}\n")
        else:
            out.append(line)
    cfg.write_text("".join(out), encoding="utf-8")


def _ensure_aiopslab_venv(repo: Path) -> None:
    venv_py = repo / ".venv_aiopslab" / "bin" / "python"
    if venv_py.exists():
        return

    py312 = _find_python312()
    if not py312:
        raise RuntimeError("Full QuickStart requires python3.12 but it was not found on PATH.")

    run_cmd(f"{shlex.quote(py312)} -m venv .venv_aiopslab", repo)
    run_cmd(".venv_aiopslab/bin/python -m pip install -e .", repo)


def _ensure_kind_cluster(repo: Path) -> None:
    kind_dir = repo / "kind"
    arm_cfg = kind_dir / "kind-config-arm.yaml"
    x86_cfg = kind_dir / "kind-config-x86.yaml"

    machine = platform.machine().lower()
    if machine in ("arm64", "aarch64") and arm_cfg.exists():
        cfg = arm_cfg
    elif x86_cfg.exists():
        cfg = x86_cfg
    elif arm_cfg.exists():
        cfg = arm_cfg
    else:
        raise RuntimeError("Full QuickStart requires kind config under kind/ but none was found.")

    mirror_url = _dockerio_mirror_url()
    if mirror_url and not _dockerhub_registry_reachable(repo):
        cfg = _write_kind_config_with_dockerio_mirror(cfg, mirror_url, repo / ".aider_fsm")

    for image in _read_kind_node_images(cfg):
        if _docker_image_exists(repo, image):
            continue
        if _is_aiopslab_kind_node_image(image):
            _ensure_aiopslab_kind_node_image(repo, image, mirror_url=mirror_url)
            continue
        if _docker_pull(repo, image):
            continue
        if mirror_url:
            mirror_ref = _dockerhub_mirror_ref(image, _strip_url_scheme(mirror_url))
            if mirror_ref and _docker_pull(repo, mirror_ref):
                run_cmd(f"docker tag {shlex.quote(mirror_ref)} {shlex.quote(image)}", repo)
                continue
        raise RuntimeError(f"Failed to pull required kind node image: {image}")

    rc, out, _err = run_cmd("kind get clusters", repo)
    clusters = {line.strip() for line in out.splitlines() if line.strip()} if rc == 0 else set()
    if "kind" not in clusters:
        run_cmd(f"kind create cluster --config {shlex.quote(str(cfg))}", repo)

    run_cmd("kubectl wait --for=condition=Ready nodes --all --timeout=180s", repo)
    _ensure_kind_can_pull_image(repo, "alpine:3.19")


def _default_full_quickstart_test_cmd() -> str:
    # Non-interactive checks that imply QuickStart prerequisites are working.
    return (
        "kubectl wait --for=condition=Ready nodes --all --timeout=180s"
        " && test -f aiopslab/config.yml"
        " && test -x .venv_aiopslab/bin/python"
        " && .venv_aiopslab/bin/python -c 'from aiopslab.orchestrator import Orchestrator; "
        "o=Orchestrator(); "
        "print(\"PROBLEMS\", len(o.probs.get_problem_ids())); "
        "print(\"NAMESPACES\", len(o.kubectl.list_namespaces().items))'"
    )


def make_plan_update_prompt(snapshot_text: str, test_cmd: str, *, extra: str = "") -> str:
    extra = extra.strip()
    extra_block = f"\n[EXTRA]\n{extra}\n" if extra else ""
    return (
        "你是一个严格的计划编辑器。你的任务：只修改 PLAN.md，把计划变成机器可解析且可执行。\n"
        "\n"
        "硬约束：\n"
        "1) 只能修改 PLAN.md，禁止修改任何其他文件。\n"
        "2) `## Next (exactly ONE item)` 下面必须恰好 1 条未完成任务：`- [ ] (STEP_ID=NNN) ...`\n"
        "3) 所有 step 必须是最小原子：一次编辑 + 一次验收即可闭合。\n"
        "4) `## Acceptance` 里必须包含：`- [ ] TEST_CMD passes: `{TEST_CMD}``\n"
        "5) 不确定点写入 `## Notes`；需要人类信息则标记 Blocked（写清楚需要什么）。\n"
        "\n"
        f"TEST_CMD: {test_cmd}\n"
        "\n"
        f"{snapshot_text}"
        f"{extra_block}"
        "\n"
        "现在只改 PLAN.md。\n"
    )


def make_execute_prompt(snapshot_text: str, step: dict[str, str]) -> str:
    return (
        "你是一个严格的执行器。你的任务：只实现 `Next` 这一个 step。\n"
        "\n"
        "硬约束：\n"
        "1) 只能做这一件事，不要顺手重构，不要做额外功能。\n"
        "2) 改动越小越好。\n"
        "3) 禁止修改 PLAN.md（如果发现需要改计划，请停止改代码）。\n"
        "\n"
        f"NEXT_STEP: (STEP_ID={step['id']}) {step['text']}\n"
        "\n"
        f"{snapshot_text}\n"
    )


def make_fix_or_replan_prompt(step: dict[str, str], test_stdout: str, test_stderr: str) -> str:
    return (
        "测试失败了。你必须二选一：\n"
        "A) 修复代码直到 TEST_CMD 通过（优先）。\n"
        "B) 如果确实缺信息/不可闭合：修改 PLAN.md，把该 step 拆小或标记 Blocked，并写清楚需要的人类输入。\n"
        "\n"
        f"FAILED_STEP: (STEP_ID={step['id']}) {step['text']}\n"
        "\n"
        "[TEST_STDOUT]\n"
        f"{test_stdout}\n"
        "\n"
        "[TEST_STDERR]\n"
        f"{test_stderr}\n"
    )


def make_mark_done_prompt(step: dict[str, str]) -> str:
    return (
        "当前 step 已通过验收。只修改 PLAN.md：\n"
        f"1) 把 `- [ ] (STEP_ID={step['id']}) ...` 从 Next 移到 Done，并改成 `- [x]`。\n"
        "2) 从 Backlog 里挑选一条最小原子任务放入 Next（保证 Next 仍然恰好 1 条）。\n"
        "3) 如果 Backlog 为空，则让 Next 为空（Next 区块保留标题但无条目）。\n"
    )


def make_block_step_prompt(step: dict[str, str], last_failure: str) -> str:
    return (
        "单步修复次数已超限。只修改 PLAN.md：\n"
        f"1) 把该 step 从 Next 移除，并在 Notes 写明 Blocked 原因与需要的人类信息。\n"
        "2) 从 Backlog 里挑选一条放入 Next（若无则 Next 置空）。\n"
        "\n"
        f"BLOCKED_STEP: (STEP_ID={step['id']}) {step['text']}\n"
        "\n"
        "[LAST_FAILURE]\n"
        f"{_tail(last_failure, STDIO_TAIL_CHARS)}\n"
    )


def _coder_run(coder, text: str, *, log_path: Path, iter_idx: int, fsm_state: str, event: str) -> None:
    append_jsonl(
        log_path,
        {
            "ts": _now_iso(),
            "iter_idx": iter_idx,
            "fsm_state": fsm_state,
            "event": event,
            "prompt_preview": _tail(text, 1200),
        },
    )
    coder.run(text)


def main() -> int:
    parser = argparse.ArgumentParser(description="Aider-FSM runner (minimal, single-process loop)")
    parser.add_argument("--repo", default=".", help="repo root path (default: .)")
    parser.add_argument("--goal", default="", help="goal for PLAN.md (used only when PLAN.md is missing)")
    parser.add_argument("--model", default="gpt-4o-mini", help="model name (default: gpt-4o-mini)")
    parser.add_argument("--test-cmd", default="pytest -q", help='acceptance command (default: "pytest -q")')
    parser.add_argument("--seed", action="append", default=[], help="seed file path (repeatable)")
    parser.add_argument("--max-iters", type=int, default=200, help="max iterations (default: 200)")
    parser.add_argument("--max-fix", type=int, default=10, help="max fix attempts per step (default: 10)")
    parser.add_argument("--plan-path", default="PLAN.md", help="plan file path relative to repo (default: PLAN.md)")
    parser.add_argument("--ensure-tools", action="store_true", help="auto-install/verify docker+k8s tools on macOS")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="run setup/checks then exit; does not invoke aider FSM loop",
    )
    parser.add_argument(
        "--full-quickstart",
        action="store_true",
        help="run AIOpsLab local kind Quick Start preflight (creates cluster/config/venv)",
    )
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    plan_rel = args.plan_path
    plan_abs = (repo / plan_rel).resolve()

    state_dir, logs_dir, state_path = _ensure_dirs(repo)
    run_id = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_path = logs_dir / f"run_{run_id}.jsonl"

    if args.ensure_tools:
        bootstrap = Path(__file__).resolve().parent / "tools" / "bootstrap_macos.py"
        if not bootstrap.exists():
            raise RuntimeError("Missing tools/bootstrap_macos.py")

        rc, out, err = _run_py_script(bootstrap, [], cwd=repo)
        append_jsonl(
            log_path,
            {
                "ts": _now_iso(),
                "iter_idx": 0,
                "fsm_state": "S0_BOOTSTRAP",
                "event": "ensure_tools",
                "rc": rc,
                "stdout_tail": out,
                "stderr_tail": err,
            },
        )
        if rc != 0:
            if out:
                print(out, file=sys.stderr)
            if err:
                print(err, file=sys.stderr)
            return 2

    if args.full_quickstart:
        _ensure_kind_cluster(repo)
        _ensure_aiopslab_config(repo)
        _ensure_aiopslab_venv(repo)
        args.test_cmd = _default_full_quickstart_test_cmd()

    if args.preflight_only:
        trc, tout, terr = run_cmd(args.test_cmd, repo)
        append_jsonl(
            log_path,
            {
                "ts": _now_iso(),
                "iter_idx": 0,
                "fsm_state": "PREFLIGHT",
                "event": "preflight_test",
                "test_rc": trc,
                "stdout_tail": tout,
                "stderr_tail": terr,
            },
        )
        if tout:
            print(tout)
        if terr:
            print(terr, file=sys.stderr)
        return 0 if trc == 0 else 3

    ensure_plan_file(plan_abs, args.goal, args.test_cmd)
    os.chdir(repo)

    defaults = default_state(repo=repo, plan_rel=plan_rel, model=args.model, test_cmd=args.test_cmd)
    state = load_state(state_path, defaults)
    save_state(state_path, state)

    fnames = [plan_rel] + list(args.seed)
    coder = create_coder(model_name=args.model, fnames=fnames)

    for iter_idx in range(1, args.max_iters + 1):
        state["iter_idx"] = iter_idx
        state["fsm_state"] = "S1_SNAPSHOT"
        save_state(state_path, state)

        snapshot, snapshot_text = build_snapshot(repo, plan_abs)
        append_jsonl(
            log_path,
            {
                "ts": _now_iso(),
                "iter_idx": iter_idx,
                "fsm_state": state["fsm_state"],
                "event": "snapshot",
                "snapshot": {k: snapshot[k] for k in ("repo", "plan_path", "env_probe", "git")},
            },
        )

        # S2: plan update (only PLAN.md)
        state["fsm_state"] = "S2_PLAN_UPDATE"
        save_state(state_path, state)
        plan_text: str | None = None
        parsed: dict[str, Any] | None = None
        next_step: dict[str, str] | None = None
        backlog_open = 0

        for attempt in range(1, 4):
            extra = ""
            if attempt > 1:
                extra = "上一次计划更新越界或格式错误：请严格只改 PLAN.md，并确保 Next 恰好 1 条。"

            _coder_run(
                coder,
                make_plan_update_prompt(snapshot_text, args.test_cmd, extra=extra),
                log_path=log_path,
                iter_idx=iter_idx,
                fsm_state=state["fsm_state"],
                event=f"plan_update_attempt_{attempt}",
            )

            changed = get_git_changed_files(repo)
            if changed is not None:
                illegal = non_plan_changes(changed, plan_rel)
                if illegal:
                    rc, out, err = git_checkout(repo, illegal)
                    append_jsonl(
                        log_path,
                        {
                            "ts": _now_iso(),
                            "iter_idx": iter_idx,
                            "fsm_state": state["fsm_state"],
                            "event": "plan_update_revert_non_plan_files",
                            "illegal_files": illegal,
                            "git_checkout_rc": rc,
                            "git_checkout_out": out,
                            "git_checkout_err": err,
                        },
                    )
                    if attempt == 3:
                        state["last_exit_reason"] = "PLAN_UPDATE_TOUCHED_CODE"
                        state["updated_at"] = _now_iso()
                        save_state(state_path, state)
                        return 2
                    continue

            plan_text = plan_abs.read_text(encoding="utf-8", errors="replace")
            parsed = parse_plan(plan_text)

            if parsed["errors"]:
                append_jsonl(
                    log_path,
                    {
                        "ts": _now_iso(),
                        "iter_idx": iter_idx,
                        "fsm_state": state["fsm_state"],
                        "event": "plan_parse_error",
                        "errors": parsed["errors"],
                    },
                )
                if attempt == 3:
                    state["last_exit_reason"] = "PLAN_PARSE_ERROR"
                    state["updated_at"] = _now_iso()
                    save_state(state_path, state)
                    return 2
                continue

            next_step = parsed["next_step"]
            backlog_open = int(parsed["backlog_open_count"])
            if next_step is None and backlog_open > 0:
                append_jsonl(
                    log_path,
                    {
                        "ts": _now_iso(),
                        "iter_idx": iter_idx,
                        "fsm_state": state["fsm_state"],
                        "event": "plan_inconsistent_missing_next",
                        "backlog_open_count": backlog_open,
                    },
                )
                if attempt == 3:
                    state["last_exit_reason"] = "MISSING_NEXT_STEP"
                    state["updated_at"] = _now_iso()
                    save_state(state_path, state)
                    return 2
                continue

            break

        if next_step is None:
            # no next, no backlog: verify once and decide done/needs plan
            state["fsm_state"] = "S4_VERIFY"
            save_state(state_path, state)
            _coder_run(
                coder,
                f"/test {args.test_cmd}",
                log_path=log_path,
                iter_idx=iter_idx,
                fsm_state=state["fsm_state"],
                event="aider_test",
            )
            trc, tout, terr = run_cmd(args.test_cmd, repo)
            state["last_test_rc"] = trc
            save_state(state_path, state)
            append_jsonl(
                log_path,
                {
                    "ts": _now_iso(),
                    "iter_idx": iter_idx,
                    "fsm_state": state["fsm_state"],
                    "event": "verify",
                    "test_rc": trc,
                    "stdout_tail": tout,
                    "stderr_tail": terr,
                },
            )
            if trc == 0:
                state["last_exit_reason"] = "DONE"
                state["updated_at"] = _now_iso()
                save_state(state_path, state)
                return 0

            # failing but no tasks -> ask plan to add next step for fixing tests
            snapshot2, snapshot_text2 = build_snapshot(repo, plan_abs)
            _coder_run(
                coder,
                make_plan_update_prompt(
                    snapshot_text2,
                    args.test_cmd,
                    extra="当前 TEST_CMD 失败但 Next/Backlog 为空：请添加一个最小 Next 来修复测试。",
                ),
                log_path=log_path,
                iter_idx=iter_idx,
                fsm_state="S2_PLAN_UPDATE",
                event="plan_update_due_to_failing_tests_no_steps",
            )
            continue

        # S3: execute
        state["fsm_state"] = "S3_EXECUTE_STEP"
        state["current_step_id"] = next_step["id"]
        state["current_step_text"] = next_step["text"]
        save_state(state_path, state)

        plan_before = plan_text
        _coder_run(
            coder,
            make_execute_prompt(snapshot_text, next_step),
            log_path=log_path,
            iter_idx=iter_idx,
            fsm_state=state["fsm_state"],
            event="execute_step",
        )

        plan_after = plan_abs.read_text(encoding="utf-8", errors="replace") if plan_abs.exists() else ""
        if plan_after != plan_before:
            # guard: execution must not change PLAN.md; revert using content (works even without git)
            plan_abs.write_text(plan_before, encoding="utf-8")
            append_jsonl(
                log_path,
                {
                    "ts": _now_iso(),
                    "iter_idx": iter_idx,
                    "fsm_state": state["fsm_state"],
                    "event": "execute_touched_plan_reverted",
                },
            )
            continue

        # S4: verify
        state["fsm_state"] = "S4_VERIFY"
        save_state(state_path, state)

        _coder_run(
            coder,
            f"/test {args.test_cmd}",
            log_path=log_path,
            iter_idx=iter_idx,
            fsm_state=state["fsm_state"],
            event="aider_test",
        )
        trc, tout, terr = run_cmd(args.test_cmd, repo)
        state["last_test_rc"] = trc
        save_state(state_path, state)
        append_jsonl(
            log_path,
            {
                "ts": _now_iso(),
                "iter_idx": iter_idx,
                "fsm_state": state["fsm_state"],
                "event": "verify",
                "test_rc": trc,
                "stdout_tail": tout,
                "stderr_tail": terr,
            },
        )

        # S5: decide
        state["fsm_state"] = "S5_DECIDE"
        save_state(state_path, state)

        if trc == 0:
            state["fix_attempts"] = 0
            save_state(state_path, state)
            _coder_run(
                coder,
                make_mark_done_prompt(next_step),
                log_path=log_path,
                iter_idx=iter_idx,
                fsm_state=state["fsm_state"],
                event="mark_done",
            )
            continue

        state["fix_attempts"] = int(state.get("fix_attempts") or 0) + 1
        save_state(state_path, state)
        if int(state["fix_attempts"]) <= int(args.max_fix):
            _coder_run(
                coder,
                make_fix_or_replan_prompt(next_step, tout, terr),
                log_path=log_path,
                iter_idx=iter_idx,
                fsm_state=state["fsm_state"],
                event=f"fix_or_replan_attempt_{state['fix_attempts']}",
            )
        else:
            state["fix_attempts"] = 0
            save_state(state_path, state)
            _coder_run(
                coder,
                make_block_step_prompt(next_step, tout + "\n" + terr),
                log_path=log_path,
                iter_idx=iter_idx,
                fsm_state=state["fsm_state"],
                event="block_step",
            )

    state["last_exit_reason"] = "MAX_ITERS"
    state["updated_at"] = _now_iso()
    save_state(state_path, state)
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
