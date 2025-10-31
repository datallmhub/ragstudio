# Security Documentation

Atlas-RAG v0.1.0 includes comprehensive security features to protect against common vulnerabilities.

## Overview

All CLI commands include built-in security validations that run automatically to protect your system and data.

## Security Features

### 1. Path Traversal Protection

Prevents malicious path patterns from accessing files outside allowed directories.

**Protected Against:**
- `../../../etc/passwd`
- Absolute paths: `/etc/*`
- Home directory expansion: `~/.ssh/*`

**Example:**
```bash
$ atlas-rag batch /data --pattern "../*.txt"
✗ Dangerous pattern blocked: ../*.txt
```

### 2. Symlink Protection

Blocks processing of symbolic links by default to prevent symlink attacks.

**Example:**
```bash
$ atlas-rag chunk /path/to/symlink.txt
✗ Symlink blocked: /path/to/symlink.txt
Set ATLAS_ALLOW_SYMLINKS=true to enable.
```

### 3. File Size Limits

Prevents processing of oversized files that could cause memory exhaustion.

**Default:** Max 100MB per file

**Configuration:**
```bash
export ATLAS_MAX_FILE_SIZE_MB=500
```

### 4. Batch Size Limits

Protects against processing an excessive number of files.

**Default:** Max 10,000 files per batch

### 5. Metadata Sanitization

Automatically escapes HTML/XML characters in metadata.

### 6. Disk Space Validation

Checks available disk space before processing.

---

## Configuration

```bash
# Security settings
export ATLAS_MAX_FILE_SIZE_MB=100
export ATLAS_MAX_BATCH_FILES=10000
export ATLAS_ALLOW_SYMLINKS=false
export ATLAS_SANITIZE_METADATA=true
```

For complete documentation, see the main [Security Guide](README.md).
