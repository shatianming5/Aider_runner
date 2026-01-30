#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[smoke] Python version:"
poetry run python -V

echo "[smoke] Import core modules:"
poetry run python -c "import aiopslab; import clients; print('imports-ok')"

echo "[smoke] Import FastAPI service entrypoint:"
poetry run python -c "import service; print('service-import-ok')"

echo "[smoke] CLI entrypoint import:"
poetry run python -c "import cli; print('cli-import-ok')"

echo "[smoke] Done."
