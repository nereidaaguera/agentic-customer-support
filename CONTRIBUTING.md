# Contributing to Telco Support Agent

## Development Workflow


1. **Create a new branch** for your feature or bugfix, prepending with `<your-name>/` (e.g., `niall/add-customer-data-generation`)
2. **Install dependencies** using Poetry: `poetry install` (requires `poetry>=2.1.2`)
3. **Make changes**, following coding standards 
4. **Run tests and linting** to ensure quality: `./scripts/lint.sh` and `pytest`
5. **Commit changes** with clear, descriptive commit messages
6. **Push branch** and submit a pull request. Tag at least one reviewer

## Code Style

We use Ruff for linting and formatting, with configurations in `pyproject.toml`. Our code style includes:

- Google-style docstrings
- 88-character line length
- Double quotes for strings
- Type annotations for all functions

pre-commit hooks will automatically enforce these standards when you commit.

## Setting Up Pre-commit Hooks

```bash
# install pre-commit hooks
# may hit issues with this if Databricks pre-commit git hooks installed. Can skip
poetry run pre-commit install
```

## Running Tests

TODO

## Adding New Dependencies

1. Add the dependency using Poetry: `poetry add package-name`
2. Regenerate `requirements.txt`: `poetry export -f requirements.txt --output requirements.txt --without-dev`

## Pull Request Process

1. Update the README.md or documentation with details of changes if appropriate
2. The PR should work on the main build and pass all tests
3. PRs require approval by at least one maintainer

Thanks for the contributions!
