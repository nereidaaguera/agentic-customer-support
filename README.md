# Telco Support Agent

Production-grade AI-powered customer support Agent System built on Databricks.

## Overview

This project implements a multi-agent system that processes telecom customer support queries in real-time, retrieves relevant information, and provides human agents with contextually appropriate responses.

## Features

- Multi-agent supervisor orchestrating specialized sub-agents
- Vector Search for unstructured data retrieval
- Tool-calling agents for structured data retrieval
- Agent lifecycle tracked using MLflow and Unity Catalog
- Production monitoring using Agent Evaluation and MLflow Tracing
- Lakehouse App for front-end user interface

## Repo Structure

```
telco_support_agent/
├── README.md
├── pyproject.toml           # Poetry for dependency management 
├── .pre-commit-config.yaml  # Pre-commit hooks configuration
├── databricks.yml           # DAB config
│
├── telco_support_agent/    
│   ├── __init__.py
│   ├── config.py            # Config management
│   │
│   ├── agents/              # Agent implementations
│   │   ├── __init__.py
│   │   ├── supervisor.py    # Supervisor agent
│   │   ├── account.py       # Account management agent
│   │   ├── billing.py       # Billing agent
│   │   ├── tech_support.py  # Tech support agent
│   │   └── product.py       # Product info agent
│   │
│   ├── tools/               # Agent tools
│   │   ├── __init__.py
│   │   └── ...
│   │
│   ├── data/                # Data generation and management
│   │   ├── __init__.py
│   │   ├── config.py        # Data generation configuration
│   │   │
│   │   ├── generators/      # Data generators
│   │   ├── schemas/         # Pydantic models for data
│   │   └── loaders/         # Data loaders
│   │
│   ├── evaluation/          # Evaluation framework
│   │   ├── __init__.py
│   │   └── ...
│   │
│   └── ui/                  # Lakehouse app UI
│       ├── __init__.py
│       └── app.py
│
├── notebooks/               # Databricks notebooks
│
├── scripts/                 # Utility scripts
│   └── lint.sh              # Linting script
│
└── tests/                   # Unit / integration tests
    ├── __init__.py
    └── ...
```

## Development Environment

This project is designed to run on [Databricks Runtime 16.4 ML LTS](https://docs.databricks.com/aws/en/release-notes/runtime/16.4lts-ml), which uses **Python 3.12.3**. For consistency between development and production environments, we recommend using the same Python version locally.

### Environment Setup

#### Prerequisites
- Python 3.12.3 (matches Databricks Runtime 16.4 ML LTS)
- Poetry for dependency management

#### Local Development

For local development, create a virtual environment using Poetry:

```bash
# Install Python 3.12.3
pyenv install 3.12.3
pyenv local 3.12.3

# Check Python version
python --version  # Should output Python 3.12.3

# Install poetry
curl -sSL https://install.python-poetry.org | python3 -

# Configure poetry to use Python 3.12
poetry env use 3.12.3

# Install dependencies
poetry install
```

#### Development Installation

For development, install the package in editable mode:

```bash
# install in development mode
pip install -e .
```

### Generating requirements.txt

We use a `requirements.txt` file to pip install dependencies in notebooks. To generate the `requirements.txt` from `pyproject.toml`:

```bash
# generate requirements.txt from pyproject.toml
poetry export -f requirements.txt --output requirements.txt --without-hashes
```

The generated `requirements.txt` will include all dependencies specified in `pyproject.toml`. When running on Databricks Runtime 16.4 ML LTS, many of these dependencies are already pre-installed, but this approach ensures compatibility with both local development and Databricks environments.

### Databricks Development

When running on Databricks:

1. Use Databricks Runtime 16.4 ML LTS
2. Install project-specific dependencies at the start of your notebook:

```python
%pip install -r /path/to/project/requirements.txt
```

## Code Quality & Linting

This project uses Ruff for linting and formatting, mypy for type checking, and pre-commit hooks for automating quality checks.

### Setting Up

```bash
# install development dependencies
poetry install

# set up pre-commit hooks
# NOTE: may run into issues with this if also running Databricks pre-commit git hooks
poetry run pre-commit install
```

### Running Linting Manually

```bash
# use the provided script
chmod +x scripts/lint.sh
./scripts/lint.sh

# or run commands individually
poetry run ruff format .                  # Format code
poetry run ruff check . --fix             # Run linter with auto-fixes
poetry run mypy telco_support_agent       # Type checking
```

### Code Style

The project follows these standards:
- Google-style docstrings
- 88-character line length (Black-compatible)
- Double quotes for strings
- Type annotations for all functions
- Pre-commit hooks enforce these standards automatically

## Data Generation

The project includes a synthetic data generation module that creates realistic telecom customer data:

- Customer profiles and subscriptions
- Plans, devices, and promotions
- Billing and usage records
- Knowledge base articles and support tickets

## Testing

TODO

## Deployment

The project uses Databricks Asset Bundles (DAB) for deployment:

```bash
# deploy to dev env
databricks bundle deploy -t dev

# deploy to prod env
databricks bundle deploy -t prod
```