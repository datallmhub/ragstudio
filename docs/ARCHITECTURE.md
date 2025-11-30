# RAG Studio Architecture

## Overview

RAG Studio (`ragctl`) is designed as a modular, pipeline-based application for processing documents. It follows a clean architecture pattern with distinct layers for CLI interaction, core logic, and infrastructure adapters.

```mermaid
graph TD
    CLI[CLI Layer (Typer)] --> Core[Core Logic]
    Core --> Ingestion[Ingestion Engine]
    Core --> Processing[Processing Pipeline]
    Core --> Storage[Storage Adapters]
    
    Ingestion --> Loaders[Document Loaders]
    Ingestion --> OCR[OCR Engine]
    
    Processing --> Chunking[Chunking Strategies]
    Processing --> Metadata[Metadata Enrichment]
    
    Storage --> FileSystem[File System]
    Storage --> VectorDB[Vector DB (Qdrant)]
```

## Core Components

### 1. CLI Layer (`src/ragctl/cli`)
Built with **Typer**, this layer handles user interaction, argument parsing, and output formatting using **Rich**. It delegates actual work to the Core layer.

### 2. Ingestion Engine (`src/ragctl/ingestion`)
Responsible for loading files and extracting text.
- **Loaders**: Wrappers around `unstructured`, `pypdf`, etc.
- **OCR**: A cascade system that tries multiple OCR engines:
  1. **EasyOCR**: High quality, GPU-accelerated
  2. **PaddleOCR**: Good for tables/layout
  3. **Tesseract**: Reliable fallback

### 3. Processing Pipeline (`src/ragctl/processing`)
Transforms raw text into semantic chunks.
- **Chunking**: Uses LangChain's `RecursiveCharacterTextSplitter`
- **Enrichment**: Adds metadata (page numbers, source, timestamps)

### 4. Storage Adapters (`src/ragctl/storage`)
Handles outputting processed data.
- **FileAdapter**: JSON, JSONL, CSV
- **VectorAdapter**: Qdrant, Chroma (planned)

## Data Flow

1. **Input**: User provides file/directory path
2. **Validation**: File type checked, permissions verified
3. **Loading**: Content extracted (with OCR if needed)
4. **Chunking**: Text split according to strategy
5. **Output**: Chunks saved to disk or vector DB

## Design Principles

- **Fail-safe**: Processing one bad file shouldn't stop the batch
- **Traceable**: Every chunk can be traced back to source file/page
- **Configurable**: All parameters tunable via config/CLI
- **Modular**: Easy to add new loaders, OCR engines, or outputs
