# Contributing to RAG Studio

First off, thank you for considering contributing to RAG Studio! 🎉

It's people like you that make RAG Studio a great tool for the community. We welcome contributions from everyone, whether you're fixing a typo, adding a feature, or improving documentation.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
  - [Reporting Bugs](#reporting-bugs)
  - [Suggesting Enhancements](#suggesting-enhancements)
  - [Your First Code Contribution](#your-first-code-contribution)
  - [Pull Requests](#pull-requests)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Commit Messages](#commit-messages)
- [Documentation](#documentation)

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior by opening an issue or contacting the maintainers.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the [issue tracker](https://github.com/datallmhub/ragstudio/issues) to see if the problem has already been reported. When you create a bug report, include as many details as possible using our bug report template.

**Good bug reports include:**
- A clear and descriptive title
- Steps to reproduce the problem
- Expected vs. actual behavior
- RAG Studio version (`ragctl --version`)
- Python version and operating system
- Sample files or error logs if applicable

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion:
- Use a clear and descriptive title
- Provide a detailed description of the proposed functionality
- Explain why this enhancement would be useful
- List any alternative solutions you've considered

### Your First Code Contribution

Unsure where to begin? Look for issues labeled:
- `good first issue` - Simple issues perfect for newcomers
- `help wanted` - Issues where we'd especially appreciate contributions
- `documentation` - Improvements to docs are always welcome!

### Pull Requests

1. **Fork the repository** and create your branch from `main`
2. **Make your changes** following our coding standards
3. **Add tests** if you've added functionality
4. **Update documentation** if you've changed APIs or behavior
5. **Ensure tests pass** by running `make test`
6. **Run linters** with `make lint`
7. **Submit your pull request**

## Development Setup

### Prerequisites

- Python 3.10 or higher
- pip and virtualenv
- Git

### Setup Instructions

```bash
# Clone your fork
git 

cd ragstudio

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install development dependencies
make install-dev

# Install pre-commit hooks
make pre-commit-install

# Verify installation
ragctl --version
make test
```

### Project Structure

```
ragstudio/
├── src/                  # Source code
│   ├── ragctl/          # Main package
│   │   ├── cli/         # CLI commands
│   │   ├── core/        # Core functionality
│   │   ├── ocr/         # OCR engines
│   │   ├── chunking/    # Text chunking
│   │   └── storage/     # Storage backends
├── tests/               # Test files
├── docs/                # Documentation
├── scripts/             # Utility scripts
└── pyproject.toml       # Project configuration
```

## Coding Standards

We use automated tools to maintain code quality:

### Code Formatting

- **Ruff**: For linting and formatting
- **Pre-commit hooks**: Automatically run on commits

```bash
# Format code
make format

# Run linters
make lint

# Fix auto-fixable issues
ruff check --fix .
```

### Python Style Guide

- Follow [PEP 8](https://peps.python.org/pep-0008/)
- Use type hints for function signatures
- Maximum line length: 100 characters
- Use descriptive variable names
- Add docstrings for public functions and classes

### Example

```python
from typing import Optional

def process_document(
    file_path: str,
    max_tokens: int = 400,
    strategy: str = "semantic"
) -> list[dict]:
    """Process a document and return chunks.
    
    Args:
        file_path: Path to the document file
        max_tokens: Maximum tokens per chunk
        strategy: Chunking strategy to use
        
    Returns:
        List of chunk dictionaries with metadata
        
    Raises:
        FileNotFoundError: If file_path doesn't exist
        ValueError: If strategy is invalid
    """
    # Implementation
    pass
```

## Testing

We maintain high test coverage to ensure reliability.

### Running Tests

```bash
# Run all tests
make test

# Run specific test file
pytest tests/test_chunking.py

# Run with coverage report
make coverage

# Run tests in watch mode
pytest-watch
```

### Writing Tests

- Place tests in `tests/` directory
- Name test files `test_*.py`
- Use descriptive test names: `test_should_chunk_document_into_semantic_parts`
- Use pytest fixtures for common setup
- Aim for >80% code coverage for new features

### Example Test

```python
import pytest
from ragctl.chunking import SemanticChunker

def test_semantic_chunker_respects_max_tokens():
    """Semantic chunker should not exceed max_tokens."""
    chunker = SemanticChunker(max_tokens=100)
    text = "word " * 500  # 500 words
    
    chunks = chunker.chunk(text)
    
    for chunk in chunks:
        assert chunk.token_count <= 100
```

## Commit Messages

We follow [Conventional Commits](https://www.conventionalcommits.org/) for clear history:

### Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

### Examples

```
feat(ocr): add support for PaddleOCR fallback

Implement automatic fallback to PaddleOCR when EasyOCR fails.
This improves reliability for complex document layouts.

Closes #123
```

```
fix(cli): handle missing config file gracefully

Previously crashed with FileNotFoundError. Now uses default
config and shows helpful warning message.

Fixes #456
```

## Documentation

Good documentation is crucial for adoption:

### When to Update Docs

- Adding new features → Update relevant docs + add usage examples
- Changing APIs → Update API reference and migration guides
- Fixing bugs → Add troubleshooting entry if user-facing
- Configuration changes → Update config examples

### Documentation Types

- **README.md**: Quick start and overview
- **docs/**: Detailed guides and references
- **Docstrings**: In-code API documentation
- **CHANGELOG.md**: Keep updated with notable changes

### Writing Style

- Be clear and concise
- Use examples liberally
- Assume readers are smart but unfamiliar with the project
- Use active voice: "Run the command" not "The command should be run"

## Development Workflow

### Typical Flow

1. **Check existing issues** or create a new one to discuss your idea
2. **Fork and clone** the repository
3. **Create a branch**: `git checkout -b feature/your-feature-name`
4. **Make changes** with good commit messages
5. **Add tests** for your changes
6. **Run tests and linters**: `make ci-all`
7. **Push to your fork**: `git push origin feature/your-feature-name`
8. **Create a Pull Request** with a clear description

### Getting Help

- 💬 **Discussions**: Ask questions in [GitHub Discussions](https://github.com/datallmhub/ragstudio/discussions)
- 🐛 **Issues**: Report bugs or request features
- 📚 **Documentation**: Check [docs/](docs/) for guides

## Recognition

Contributors are recognized in our README and releases. We appreciate all contributions, from code to documentation to bug reports!

## Questions?

Don't hesitate to ask! We're here to help:
- Open an issue with the `question` label
- Start a discussion in GitHub Discussions
- Comment on relevant issues or pull requests

Thank you for contributing to RAG Studio! 🚀
