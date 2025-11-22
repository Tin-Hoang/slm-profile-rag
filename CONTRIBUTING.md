# Contributing to SLM Profile RAG

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Development Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Tin-Hoang/slm-profile-rag.git
   cd slm-profile-rag
   ```

2. **Create a virtual environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install in development mode**:
   ```bash
   pip install -e ".[dev]"
   ```

4. **Copy environment template**:
   ```bash
   cp .env.example .env
   ```

## Code Quality

### Formatting and Linting

This project uses [Ruff](https://github.com/astral-sh/ruff) for linting and formatting.

Format code:
```bash
ruff format .
```

Check linting:
```bash
ruff check .
```

Auto-fix linting issues:
```bash
ruff check --fix .
```

### Running Tests

Run all tests:
```bash
pytest
```

Run with coverage:
```bash
pytest --cov=slm_profile_rag tests/
```

Run specific test:
```bash
pytest tests/test_config.py::test_config_loads_yaml -v
```

## Project Structure

```
slm-profile-rag/
├── src/slm_profile_rag/      # Main package
│   ├── __init__.py
│   ├── app.py                # Gradio UI
│   ├── config.py             # Configuration loader
│   ├── document_processor.py # Document loading
│   └── rag_pipeline.py       # RAG implementation
├── tests/                    # Test files
├── profile_docs/             # Profile documents
├── config.yaml               # Configuration
├── pyproject.toml            # Project metadata
└── README.md
```

## Making Changes

1. **Create a branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the coding standards

3. **Add tests** for new functionality

4. **Run tests and linting**:
   ```bash
   ruff format .
   ruff check .
   pytest
   ```

5. **Commit your changes**:
   ```bash
   git add .
   git commit -m "Description of changes"
   ```

6. **Push and create a pull request**:
   ```bash
   git push origin feature/your-feature-name
   ```

## Coding Standards

- Follow PEP 8 style guide
- Use type hints where appropriate
- Write docstrings for all public functions/classes
- Keep functions small and focused
- Add comments for complex logic
- Write tests for new features

## Pull Request Process

1. Update documentation if needed
2. Add tests for new features
3. Ensure all tests pass
4. Update README.md if needed
5. Request review from maintainers

## Bug Reports

When filing a bug report, include:
- Python version
- Operating system
- Steps to reproduce
- Expected behavior
- Actual behavior
- Error messages/logs

## Feature Requests

For feature requests, describe:
- The problem you're trying to solve
- Your proposed solution
- Alternative solutions considered
- Any additional context

## Questions?

Feel free to open an issue for questions or clarifications.
