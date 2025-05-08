#!/bin/bash
# Run linting on the codebase
set -e

echo "Running Ruff formatter..."
poetry run ruff format .

echo "Running Ruff linter..."
poetry run ruff check . --fix

# echo "Running mypy type checker..."
# poetry run mypy telco_support_agent