"""
Atlas-RAG Core Modules

Core functionality that can be used independently without Prefect workflows.
Includes API, CLI, chunking, vector database, caching, and configuration.
"""

__version__ = "0.1.0"

# Core modules for basic RAG functionality
__all__ = [
    "api",
    "cache",
    "chunk",
    "cli",
    "config",
    "vector",
]