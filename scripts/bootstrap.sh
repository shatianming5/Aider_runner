#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if ! command -v poetry >/dev/null 2>&1; then
  echo "Poetry is required but was not found. Install it first: https://python-poetry.org/docs/#installation"
  exit 1
fi

# Install dependencies (non-interactive)
poetry install --no-interaction

# Ensure runtime config exists
if [[ -f "aiopslab/config.yml.example" && ! -f "aiopslab/config.yml" ]]; then
  cp "aiopslab/config.yml.example" "aiopslab/config.yml"
  echo "Created aiopslab/config.yml from aiopslab/config.yml.example"
fi
