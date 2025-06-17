# Contributing to EarthSense AI

Thank you for your interest in contributing to EarthSense AI! We appreciate your time and effort in making this project better.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Environment Setup](#development-environment-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Reporting Issues](#reporting-issues)
- [Code Style Guide](#code-style-guide)
- [Documentation](#documentation)
- [License](#license)

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Getting Started

1. **Fork** the repository on GitHub
2. **Clone** your fork locally
   ```bash
   git clone https://github.com/yourusername/earthsense-ai.git
   cd earthsense-ai
   ```
3. Set up the development environment (see below)

## Development Environment Setup

1. Create a Python virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install development dependencies:
   ```bash
   pip install -r requirements-dev.txt
   pip install -e .
   ```

3. Set up pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Making Changes

1. Create a new branch for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b bugfix/issue-number-description
   ```

2. Make your changes following the code style guide

3. Run tests to ensure nothing is broken:
   ```bash
   pytest
   ```

4. Commit your changes with a descriptive message:
   ```bash
   git commit -m "Add feature: brief description of changes"
   ```

## Testing

We use `pytest` for testing. To run the test suite:

```bash
pytest
```

To run tests with coverage report:

```bash
pytest --cov=src tests/
```

## Pull Request Process

1. Ensure your fork is up to date with the main repository
2. Create a pull request with a clear title and description
3. Reference any related issues in your PR description
4. Ensure all tests pass and code coverage remains high
5. Request review from at least one maintainer

## Reporting Issues

When reporting issues, please include:

- A clear, descriptive title
- Steps to reproduce the issue
- Expected vs. actual behavior
- Environment details (OS, Python version, etc.)
- Any relevant logs or error messages

## Code Style Guide

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with the following additions:

- Maximum line length: 88 characters (Black default)
- Use absolute imports
- Type hints are required for all function/method signatures
- Docstrings follow Google style

### Pre-commit Hooks

We use pre-commit hooks to maintain code quality. The following hooks are configured:

- `black` - Code formatting
- `isort` - Import sorting
- `flake8` - Linting
- `mypy` - Static type checking

## Documentation

- Update documentation when adding new features or changing behavior
- Follow Google-style docstrings for all public functions and classes
- Keep README.md up to date
- Document any environment variables in .env.example

## License

By contributing to EarthSense AI, you agree that your contributions will be licensed under the [MIT License](LICENSE).
