# Changelog

All notable changes to RAG Studio will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.3] - 2025-10-30

### ✅ Testing & Quality Improvements

**Test Coverage Expansion** - 37% → 41%:
- Added 60 comprehensive tests for security utilities
- security.py coverage: 0% → 89%
- Total unit tests: 436 → 496 (+60 tests)
- All 496 tests passing (100% success rate)

**New Test Modules**:
- test_security.py (60 tests) - Comprehensive security validation tests
  - SecurityConfig and environment loading
  - Path security (path traversal, symlinks, pattern validation)
  - File size validation (individual and batch)
  - Disk space validation
  - MIME type validation
  - Metadata sanitization

**Database Migration**:
- Completed PostgreSQL → SQLite migration
- Removed all PostgreSQL-specific code (JSONB → JSON)
- metadata_store.py: 17% → 89% coverage

### 🧹 Code Quality

**Git History Cleanup**:
- Removed .archive directory from git tracking
- Clean commit history for production release

## [0.1.2] - 2025-10-29

### 🎨 Rebranding

**RAG Studio** - Professional rebrand with improved naming:
- **Project name**: Atlas-RAG → **RAG Studio**
- **Package name**: atlas-rag → **ragctl** (follows kubectl/systemctl convention)
- **CLI command**: `atlas-rag` → `ragctl`
- **Repository**: horiz-data/atlas-rag → horiz-data/ragstudio

### ⚡ Performance Improvements

**CLI Startup Optimization** - 100x faster:
- Implemented lazy imports for heavy modules (torch, transformers, langchain)
- Startup time: 5-10s → 0.08s
- `ragctl --version` now instant
- Commands load dependencies only when needed

### ✨ New Features

**User Experience**:
- Suppressed transformers FutureWarning for cleaner output
- Updated all help text and branding to RAG Studio
- Improved version display format
