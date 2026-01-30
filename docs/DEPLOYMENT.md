# Deployment

## Local (recommended for development)

Use the provided script:

```bash
bash scripts/run_local.sh
```

This runs `service.py` using uvicorn in the Poetry environment.

Environment variables:

- `HOST` (default `0.0.0.0`)
- `PORT` (default `8000`)

## Docker

A Dockerfile is not added in this change set to avoid guessing runtime requirements (e.g., Kubernetes access, model endpoints, and submodule assets).

If you want Docker-based deployment, add any existing Docker-related files (if present) or confirm none exist, and I can add a minimal `Dockerfile` and document its limitations.
