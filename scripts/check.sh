#!/usr/bin/env bash
# Development quality checks
set -e

cd "$(dirname "$0")/.."

echo "==> Formatting check (black)"
uv run black --check backend/

echo "==> All checks passed"
