#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

bash scripts/bootstrap.sh

echo "[ci] Lint (optional)"
if poetry run python -c "import ruff" >/dev/null 2>&1; then
  poetry run ruff check .
elif poetry run python -c "import flake8" >/dev/null 2>&1; then
  poetry run flake8
else
  echo "[ci] No linter installed (ruff/flake8 not found); skipping lint."
fi

echo "[ci] Test"
if [[ "${RUN_UNIT_TESTS:-0}" == "1" ]]; then
  poetry run pytest -q
else
  bash scripts/smoke_test.sh
fi

echo "[ci] Build"
echo "[ci] No build step configured; skipping."
