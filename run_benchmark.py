import os

# Force a single model for both the primary agent and the judge/evaluator stack.
# This avoids mismatches where the judge uses a different default model and fails.
os.environ.setdefault("OPENAI_MODEL", "gpt-5.2")

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
    base_url = (base_url or "https://api.openai.com").rstrip("/")
    if base_url.endswith("/v1"):
        return base_url
    return f"{base_url}/v1"


async def run_benchmark():
    """Run the benchmark using a fixed model."""
    model = "gpt-5.2"

    # Ensure base URL is normalized for any downstream OpenAI client usage.
    os.environ["OPENAI_BASE_URL"] = _normalize_openai_base_url(os.getenv("OPENAI_BASE_URL", "https://api.openai.com"))

    print(f"Using model: {model}")

    problems = ProblemRegistry().PROBLEM_REGISTRY
    for pid in problems:
        agent = GPTAgent()
        agent.llm.model = model  # Set the fixed model for the agent

        orchestrator = Orchestrator()
        orchestrator.register_agent(agent, name="benchmark-agent")

        problem_desc, instructs, apis = orchestrator.init_problem(pid)
        agent.init_context(problem_desc, instructs, apis)
        await orchestrator.start_problem(max_steps=30)


if __name__ == "__main__":
    asyncio.run(run_benchmark())
