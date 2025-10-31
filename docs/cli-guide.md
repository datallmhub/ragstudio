# CLI Command Reference

Complete reference for all `ragctl` CLI commands.

## Installation

```bash
pip install ragctl
```

## Command Overview

```bash
ragctl --version          # Show version
ragctl --help             # Show help
ragctl <command> --help   # Command-specific help
```

## Available Commands

| Command | Description |
|---------|-------------|
| `chunk` | Chunk a single document |
| `batch` | Process multiple files in batch mode |
| `ingest` | Ingest chunks into Qdrant vector store |
| `eval` | Evaluate chunking quality |
| `info` | Display system information |
| `retry` | Retry failed files from previous run |

---

## `chunk` - Chunk a Single Document

Process a single document and split it into chunks.

### Basic Usage

```bash
# Basic chunking
ragctl chunk document.txt

# With output file
ragctl chunk document.txt -o chunks.json

# Display chunks in terminal
ragctl chunk document.txt --show

# Limit displayed chunks
ragctl chunk document.txt --show --limit 5
```

### Chunking Strategies

```bash
# Semantic chunking (default)
ragctl chunk doc.txt --strategy semantic

# Sentence-based chunking
ragctl chunk doc.txt --strategy sentence

# Token-based chunking
ragctl chunk doc.txt --strategy token
```

### Parameters

```bash
# Custom chunk size
ragctl chunk doc.txt --max-tokens 512

# Custom overlap
ragctl chunk doc.txt --overlap 100

# Combine parameters
ragctl chunk doc.txt --strategy token --max-tokens 256 --overlap 50
```

### Full Options

```
Options:
  --strategy, -s       Chunking strategy (semantic|sentence|token)
  --max-tokens, -m     Maximum tokens per chunk (50-2000)
  --overlap, -ol       Token overlap between chunks (0-500)
  --output, -o         Output JSON file path
  --show               Display chunks in terminal
  --limit, -l          Max chunks to display (1-100)
  --advanced-ocr       Use intelligent OCR routing (for PDFs)
  --help               Show help message
```

---

## `batch` - Batch Processing

Process multiple files at once with automatic retry and error handling.

### Basic Usage

```bash
# Process all files in directory
ragctl batch ./documents

# Process with pattern
ragctl batch ./documents --pattern "*.txt"

# Recursive processing
ragctl batch ./documents --pattern "*.pdf" --recursive
```

### Output Options

```bash
# One file per document
ragctl batch ./docs -o ./output

# Single combined file
ragctl batch ./docs -o all_chunks.json --single-file
```

### Processing Modes

```bash
# Interactive mode (default) - ask on errors
ragctl batch ./docs

# Auto-continue on errors
ragctl batch ./docs --auto-continue

# Stop on first error
ragctl batch ./docs --auto-stop

# Skip failed files automatically
ragctl batch ./docs --auto-skip
```

### Full Options

```
Arguments:
  directory            Directory containing files to process

Options:
  --pattern, -p        File pattern (e.g., '*.txt', '*.pdf')
  --strategy, -s       Chunking strategy
  --max-tokens, -m     Maximum tokens per chunk
  --overlap, -ol       Token overlap
  --output, -o         Output directory or file
  --single-file        Combine all chunks in one file
  --recursive, -r      Process subdirectories
  --auto-continue      Continue on errors
  --auto-stop          Stop on first error
  --auto-skip          Skip failed files
  --help               Show help message
```

---

## `ingest` - Vector Store Ingestion

Ingest chunks into Qdrant vector store with automatic embedding generation.

### Basic Usage

```bash
# Basic ingestion
ragctl ingest chunks.json

# Custom collection
ragctl ingest chunks.json --collection my_docs

# Custom Qdrant URL
ragctl ingest chunks.json --qdrant-url http://192.168.1.100:6333
```

### Collection Management

```bash
# Recreate collection (⚠️ deletes existing data)
ragctl ingest chunks.json --collection my_docs --recreate

# Custom embedding dimension
ragctl ingest chunks.json --embedding-dim 768

# Custom batch size
ragctl ingest chunks.json --batch-size 64
```

### Pipeline Example

```bash
# Complete pipeline: chunk → ingest
ragctl chunk doc.txt -o /tmp/chunks.json && \
ragctl ingest /tmp/chunks.json --collection my_collection
```

### Full Options

```
Arguments:
  chunks_file          JSON file containing chunks

Options:
  --collection, -c     Qdrant collection name
  --qdrant-url         Qdrant server URL
  --recreate           Recreate collection (⚠️ deletes data)
  --embedding-dim      Embedding vector dimension (128-4096)
  --batch-size         Chunks per batch (1-1000)
  --help               Show help message
```

---

## `eval` - Quality Evaluation

Evaluate chunking quality and compare strategies.

### Basic Usage

```bash
# Evaluate chunks
ragctl eval chunks.json

# Compare strategies
ragctl eval doc.txt --compare

# Detailed output
ragctl eval chunks.json --detailed
```

---

## `retry` - Retry Failed Files

Retry files that failed in previous batch runs.

### Basic Usage

```bash
# Show last failed run
ragctl retry --show

# Retry last failed run
ragctl retry

# Retry specific run
ragctl retry run_20251028_123456
```

---

## `info` - System Information

Display system status and capabilities.

```bash
ragctl info
```

Output:
- API status (if available)
- Vector store status
- Local capabilities
- System version

---

## Security Features

All commands include built-in security validations:

- ✓ **Path traversal protection** - Blocks `../` patterns
- ✓ **Symlink protection** - Rejects symbolic links
- ✓ **File size limits** - Max 100MB per file (configurable)
- ✓ **Batch size limits** - Max 10,000 files per batch
- ✓ **Metadata sanitization** - Escapes HTML/XML

See [Security Documentation](security/) for details.

---

## Environment Variables

Configure security and behavior:

```bash
# Security
export ATLAS_MAX_FILE_SIZE_MB=100
export ATLAS_MAX_BATCH_FILES=10000
export ATLAS_ALLOW_SYMLINKS=false
export ATLAS_SANITIZE_METADATA=true

# Processing
export ATLAS_DEFAULT_STRATEGY=semantic
export ATLAS_DEFAULT_MAX_TOKENS=400
```

---

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Error |
| 124 | Timeout |

---

## Examples

### Basic Workflow

```bash
# 1. Chunk a document
ragctl chunk document.pdf -o chunks.json --show

# 2. Ingest to vector store
ragctl ingest chunks.json --collection docs

# 3. Check system info
ragctl info
```

### Batch Processing

```bash
# Process all PDFs recursively
ragctl batch ./documents \
  --pattern "*.pdf" \
  --recursive \
  --auto-continue \
  -o ./output

# Retry failed files
ragctl retry
```

### Advanced Chunking

```bash
# Semantic chunking with custom parameters
ragctl chunk large_doc.txt \
  --strategy semantic \
  --max-tokens 512 \
  --overlap 100 \
  -o chunks.json
```

---

For more examples, see [Examples Documentation](examples.md).
