#!/bin/bash
# Run linting on the codebase
set -e

echo "Running Ruff formatter..."
uv run ruff format .

echo "Running Ruff linter..."
uv run ruff check . --fix

# echo "Running mypy type checker..."
# uv run mypy telco_support_agent