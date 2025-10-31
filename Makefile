# Atlas-RAG CLI - Development Makefile

.PHONY: help test lint format install-dev test-cli test-cli-e2e test-cli-quick pre-commit pre-commit-install ci-lint ci-test ci-security ci-all backup-data restore-data

# Default target
help:
	@echo "Atlas-RAG CLI - Available Commands:"
	@echo ""
	@echo "Development:"
	@echo "  install-dev     Install development dependencies"
	@echo "  test            Run all tests with coverage"
	@echo "  test-cli        Run CLI pytest tests"
	@echo "  test-cli-e2e    Run comprehensive CLI E2E tests"
	@echo "  test-cli-quick  Run quick CLI validation"
	@echo "  lint            Run linting (ruff, mypy)"
	@echo "  format          Format code (black, isort)"
	@echo ""
	@echo "Quality Assurance:"
	@echo "  pre-commit-install  Install pre-commit hooks"
	@echo "  pre-commit          Run pre-commit checks"
	@echo ""
	@echo "CI/CD:"
	@echo "  ci-lint         Run linting checks"
	@echo "  ci-test         Run tests with coverage"
	@echo "  ci-security     Run security checks"
	@echo "  ci-all          Run all CI checks"
	@echo ""
	@echo "Data Management:"
	@echo "  backup-data     Create backup of data and logs"
	@echo "  restore-data    Restore from backup"

# Development
install-dev:
	poetry install --with dev

test:
	pytest tests/ -v --cov=src --cov-report=html

test-cli:
	@echo "🧪 Running CLI pytest tests..."
	pytest tests/cli/ -v --no-cov
	@echo "✅ CLI tests passed!"

test-cli-e2e:
	@echo "🎯 Running comprehensive CLI E2E tests..."
	@bash tests/cli_e2e_test.sh
	@echo "✅ CLI E2E tests complete!"

test-cli-quick:
	@echo "⚡ Running quick CLI validation..."
	@echo "Testing core commands..."
	@atlas-rag --version
	@atlas-rag --help > /dev/null
	@atlas-rag chunk --help > /dev/null
	@atlas-rag batch --help > /dev/null
	@atlas-rag ingest --help > /dev/null
	@atlas-rag eval --help > /dev/null
	@echo "✅ Quick validation passed!"

lint:
	poetry run ruff check src/ tests/
	poetry run mypy src/

format:
	poetry run black src/ tests/
	poetry run isort src/ tests/

# Quality Assurance
pre-commit-install:
	@echo "📦 Installing pre-commit hooks..."
	pre-commit install
	@echo "✅ Pre-commit hooks installed!"

pre-commit:
	@echo "🔍 Running pre-commit checks..."
	pre-commit run --all-files
	@echo "✅ Pre-commit checks complete!"

# CI/CD targets
ci-lint:
	@echo "🔍 Running CI linting checks..."
	black --check src/ tests/
	isort --check-only src/ tests/
	flake8 src/ tests/
	@echo "✅ Linting checks passed!"

ci-test:
	@echo "🧪 Running CI tests..."
	pytest tests/ --cov=src --cov-report=xml --cov-report=html --cov-report=term-missing -v
	@echo "✅ Tests passed!"

ci-security:
	@echo "🔒 Running security checks..."
	safety check || true
	bandit -r src/ -f json -o bandit-report.json || true
	@echo "✅ Security checks complete!"

ci-all: ci-lint ci-test ci-security
	@echo "✅ All CI checks passed!"

# Backup and restore
backup-data:
	@echo "📦 Creating backup..."
	tar -czf atlas-rag-backup-$(shell date +%Y%m%d-%H%M%S).tar.gz data/ logs/
	@echo "✅ Backup created!"

restore-data:
	@echo "📥 Restoring from backup..."
	@read -p "Enter backup filename: " backup; \
	tar -xzf $$backup
	@echo "✅ Data restored!"
