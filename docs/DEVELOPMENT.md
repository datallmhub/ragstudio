# Development Guide

## Setting Up Your Environment

1. **Clone the repository**
   ```bash
   git clone https://github.com/datallmhub/ragstudio.git
   cd ragstudio
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   make install-dev
   ```

4. **Install pre-commit hooks**
   ```bash
   make pre-commit-install
   ```

## Project Structure

- `src/ragctl`: Main package source
- `tests/`: Pytest test suite
- `docs/`: Documentation
- `scripts/`: Helper scripts
- `Makefile`: Common tasks

## Common Tasks

### Running Tests
```bash
make test          # Run all tests
make test-cli      # Run only CLI tests
make coverage      # Generate coverage report
```

### Code Quality
```bash
make format        # Format code with ruff
make lint          # Check code quality
```

### Building Documentation
```bash
make docs
```

## Adding a New Feature

1. **Create a branch**: `feature/my-feature`
2. **Write tests**: TDD is encouraged
3. **Implement feature**: Keep functions small and focused
4. **Verify**: Run `make ci-all`
5. **Submit PR**: Reference related issues

## Debugging

You can use the `RAGCTL_DEBUG=1` environment variable to enable verbose logging:

```bash
RAGCTL_DEBUG=1 ragctl chunk mydoc.pdf
```

## Release Process

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create a git tag: `git tag v0.1.4`
4. Push tag: `git push origin v0.1.4`
5. CI will automatically build and publish to PyPI
