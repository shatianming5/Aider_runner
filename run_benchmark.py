import os
import asyncio
from clients.gpt import GPTAgent
from aiopslab.orchestrator import Orchestrator
from aiopslab.orchestrator.problems.registry import ProblemRegistry

def select_model():
    """Select the best available model from the OpenAI API."""
    import requests

    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com").rstrip("/")
    try:
        url = f"{base_url}/models" if base_url.endswith("/v1") else f"{base_url}/v1/models"
        response = requests.get(url, headers={
            "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"
        })
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Failed to fetch models: {e}")

    models = response.json().get("data", [])
    if not models:
        raise RuntimeError("No models found in the response.")

    models = response.json().get("data", [])
    priority = ["gpt-4o-msra", "gpt-4o", "gpt-4", "gpt-35-turbo"]
    for model in priority:
        if any(model in m["id"] for m in models):
            return model

    raise RuntimeError("No suitable model found.")

async def run_benchmark():
    """Run the benchmark using the selected model."""
    model = select_model()
    print(f"Using model: {model}")

    problems = ProblemRegistry().PROBLEM_REGISTRY
    for pid in problems:
        agent = GPTAgent()
        agent.llm.model = model  # Set the selected model for the agent

        orchestrator = Orchestrator()
        orchestrator.register_agent(agent, name="benchmark-agent")

        problem_desc, instructs, apis = orchestrator.init_problem(pid)
        agent.init_context(problem_desc, instructs, apis)
        await orchestrator.start_problem(max_steps=30)

if __name__ == "__main__":
    asyncio.run(run_benchmark())
