"""
Vector database integration module.

Provides connectors for popular vector databases to enable RAG applications.
"""

from src.core.vector.base import VectorStore, VectorStoreConfig
from src.core.vector.embeddings import embed_chunks, ChunkEmbedding
from src.core.vector.qdrant_store import QdrantStore

# Alias for backwards compatibility
QdrantVectorStore = QdrantStore

__all__ = [
    "VectorStore",
    "VectorStoreConfig",
    "embed_chunks",
    "ChunkEmbedding",
    "QdrantStore",
    "QdrantVectorStore",
]
