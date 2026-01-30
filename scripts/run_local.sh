#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

bash scripts/bootstrap.sh

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

poetry run python -m uvicorn service:app --host "$HOST" --port "$PORT"
