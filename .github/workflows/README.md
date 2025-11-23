# GitHub Actions Workflows

This directory contains automated CI/CD workflows for the SLM Profile RAG project.

![Tests](https://github.com/Tin-Hoang/slm-profile-rag/actions/workflows/tests.yml/badge.svg)
![Lint](https://github.com/Tin-Hoang/slm-profile-rag/actions/workflows/lint.yml/badge.svg)
![Code Quality](https://github.com/Tin-Hoang/slm-profile-rag/actions/workflows/code-quality.yml/badge.svg)
![Security Scan](https://github.com/Tin-Hoang/slm-profile-rag/actions/workflows/security-scan.yml/badge.svg)
[![codecov](https://codecov.io/gh/Tin-Hoang/slm-profile-rag/branch/main/graph/badge.svg)](https://codecov.io/gh/Tin-Hoang/slm-profile-rag)

## Workflows Overview

### 🧪 tests.yml - Unit Tests
**Triggers:** Push/PR to main, master, develop branches; Manual dispatch

**Purpose:** Runs comprehensive unit tests across multiple Python versions

**Jobs:**
- **test**: Runs pytest with coverage on Python 3.10, 3.11, and 3.12
  - Installs dependencies
  - Runs ruff linting and formatting checks
  - Executes pytest with coverage reporting
  - Uploads coverage to Codecov (Python 3.11 only)

- **test-docker**: Validates Docker image build
  - Builds Docker image to ensure Dockerfile is valid
  - Uses caching for faster builds

### 🔍 lint.yml - Linting (Existing)
**Triggers:** Push/PR to main, develop branches

**Purpose:** Fast linting checks using ruff

**Jobs:**
- Runs ruff check on all Python files
- Runs ruff format check
- Uses UV for fast dependency installation

### 📊 code-quality.yml - Code Quality Analysis
**Triggers:** Push/PR to main, master, develop branches

**Purpose:** Advanced code quality checks

**Jobs:**
- Python syntax validation
- Code complexity analysis using radon
- Maintainability index calculation

### ✅ pre-commit.yml - Pre-commit Hooks
**Triggers:** Push/PR to main, master, develop branches

**Purpose:** Validates pre-commit hooks configuration

**Jobs:**
- Runs all pre-commit hooks defined in `.pre-commit-config.yaml`
- Ensures code formatting and quality standards

### 📦 dependency-review.yml - Dependency Security
**Triggers:** Pull requests only

**Purpose:** Reviews dependency changes for security issues

**Jobs:**
- Scans for vulnerable dependencies
- Checks for problematic licenses (GPL-3.0, AGPL-3.0)
- Fails on moderate or higher severity vulnerabilities

### 🔒 security-scan.yml - Security Scanning
**Triggers:** Push/PR to main, master, develop branches; Weekly schedule (Mondays)

**Purpose:** Comprehensive security vulnerability scanning

**Jobs:**
- **Safety**: Scans Python dependencies for known vulnerabilities
- **Bandit**: Static security analysis of Python code
- **pip-audit**: Audits Python packages for known vulnerabilities
- Uploads security reports as artifacts

## Local Development

### Running Tests Locally

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run all tests
pytest tests/ -v

# Run tests with coverage
pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

# Run specific test file
pytest tests/test_main_document_loader.py -v
```

### Running Linting Locally

```bash
# Install ruff
pip install ruff

# Check code
ruff check src tests app.py

# Format code
ruff format src tests app.py
```

### Running Pre-commit Hooks Locally

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run on all files
pre-commit run --all-files
```

### Running Security Scans Locally

```bash
# Install security tools
pip install safety bandit pip-audit

# Run safety check
safety check

# Run bandit
bandit -r src/

# Run pip-audit
pip-audit
```

## Continuous Integration Matrix

| Workflow | Python Versions | OS | Caching | Coverage |
|----------|----------------|-----|---------|----------|
| tests.yml | 3.10, 3.11, 3.12 | Ubuntu | ✅ pip | ✅ Codecov |
| lint.yml | 3.11 | Ubuntu | ✅ UV | ❌ |
| code-quality.yml | 3.11 | Ubuntu | ✅ pip | ❌ |
| pre-commit.yml | 3.11 | Ubuntu | ❌ | ❌ |
| security-scan.yml | 3.11 | Ubuntu | ✅ pip | ❌ |

## Troubleshooting

### Tests Failing Locally but Passing in CI (or vice versa)

1. Ensure you're using the correct Python version
2. Clear pytest cache: `pytest --cache-clear`
3. Reinstall dependencies: `pip install -e ".[dev]" --force-reinstall`

### Codecov Upload Issues

1. Ensure `CODECOV_TOKEN` is set in repository secrets
2. Check coverage.xml is generated: `pytest tests/ --cov=src --cov-report=xml`

### Docker Build Failures

1. Test locally: `docker build -t slm-profile-rag:test .`
2. Check Dockerfile syntax
3. Ensure all required files are present

## Contributing

When contributing to this project:

1. Ensure all workflows pass before submitting a PR
2. Add tests for new features
3. Run pre-commit hooks locally
4. Update this README if adding new workflows

## Maintenance

### Updating Dependencies

- GitHub Actions are set to use latest stable versions (v4, v5)
- Review and update action versions quarterly
- Monitor Dependabot alerts for security updates

### Adding New Workflows

1. Create a new `.yml` file in this directory
2. Follow the naming convention: `<purpose>.yml`
3. Add appropriate triggers and jobs
4. Update this README with workflow documentation
5. Add status badge to main README.md
