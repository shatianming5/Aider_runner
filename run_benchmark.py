import os
import sys
import types

# Force a single model for both the primary agent and the judge/evaluator stack.
# This avoids mismatches where the judge uses a different default model and fails.
os.environ.setdefault("OPENAI_MODEL", "gpt-5.2")

# Make the benchmark runnable in a clean environment where wandb may not be installed.
# `clients/gpt.py` imports wandb at import time in some versions; provide a tiny stub if absent.
try:
    import wandb  # noqa: F401
except Exception:
    wandb_stub = types.ModuleType("wandb")

    def _noop(*args, **kwargs):
        return None

    wandb_stub.init = _noop
    wandb_stub.login = _noop
    wandb_stub.finish = _noop
    wandb_stub.log = _noop
    wandb_stub.config = {}
    sys.modules["wandb"] = wandb_stub

import asyncio
from clients.gpt import GPTAgent
from aiopslab.orchestrator import Orchestrator
from aiopslab.orchestrator.problems.registry import ProblemRegistry


def _normalize_openai_base_url(base_url: str) -> str:
    """
    Normalize OPENAI_BASE_URL so it can be safely used with or without an embedded /v1.
    Examples:
      - https://api.openai.com      -> https://api.openai.com/v1
      - https://api.openai.com/v1   -> https://api.openai.com/v1
      - https://host/prefix/v1/     -> https://host/prefix/v1
    """
    base_url = (base_url or "https://api.openai.com").strip().rstrip("/")
    if base_url.endswith("/v1"):
        return base_url
    return f"{base_url}/v1"


async def run_benchmark():
    """Run the benchmark using a fixed model."""
    model = "gpt-5.2"

    # Normalize base URL and keep both env var spellings consistent.
    normalized_base = _normalize_openai_base_url(
        os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE") or "https://api.openai.com"
    )
    os.environ["OPENAI_BASE_URL"] = normalized_base
    os.environ["OPENAI_API_BASE"] = normalized_base

    # Ensure the judge/evaluator stack uses the same model too.
    os.environ["OPENAI_MODEL"] = model

    print(f"Using model: {model}")
    print(f"Using OPENAI_BASE_URL: {os.environ['OPENAI_BASE_URL']}")

    problems = ProblemRegistry().PROBLEM_REGISTRY
    for pid in problems:
        agent = GPTAgent()
        agent.llm.model = model  # Fixed model for the agent

        orchestrator = Orchestrator()
        orchestrator.register_agent(agent, name="benchmark-agent")

        problem_desc, instructs, apis = orchestrator.init_problem(pid)
        agent.init_context(problem_desc, instructs, apis)
        await orchestrator.start_problem(max_steps=30)


if __name__ == "__main__":
    asyncio.run(run_benchmark())
