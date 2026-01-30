# Runbook

## Prerequisites

- Python (recommended 3.11)
- Poetry
- (Optional) kubectl + a reachable cluster for Kubernetes-dependent tests/tasks

## Bootstrap

From repo root:

```bash
bash scripts/bootstrap.sh
```

This installs Poetry dependencies and creates `aiopslab/config.yml` from `aiopslab/config.yml.example` if needed.

## Run the FastAPI service locally

```bash
bash scripts/run_local.sh
```

Defaults: `HOST=0.0.0.0`, `PORT=8000`.

Health check:

```bash
curl -sS http://127.0.0.1:8000/health
```

## Run the CLI

```bash
poetry run python cli.py
```

## Tests

Default (minimal, no kubectl):

```bash
bash scripts/smoke_test.sh
```

Full test suite (may require kubectl/cluster):

```bash
RUN_UNIT_TESTS=1 bash scripts/ci.sh
```

## Benchmark

Example (with a custom OpenAI-compatible endpoint):

```bash
OPENAI_BASE_URL=http://127.0.0.1:38889/v1 \
OPENAI_API_KEY=your_key \
poetry run python run_benchmark.py
```

`run_benchmark.py` normalizes `OPENAI_BASE_URL` to avoid `/v1/v1/...` issues and forces both the primary model and judge to `gpt-5.2`.
