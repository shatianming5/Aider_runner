# Repo Overview

This repository contains the AIOpsLab framework, including:

- A FastAPI service entrypoint: `service.py` (includes `/health`)
- A CLI entrypoint: `cli.py`
- The orchestrator/evaluator stack under `aiopslab/`
- Model clients under `clients/`
- Tests under `tests/` (some tests require `kubectl`/a live Kubernetes cluster)

Dependency management uses Poetry (`pyproject.toml`).

## Configuration

A config example is provided at:

- `aiopslab/config.yml.example`

Before running, copy it to:

- `aiopslab/config.yml`

The bootstrap script does this automatically if `config.yml` is missing.
