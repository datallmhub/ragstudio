"""
Chunk embedding generation for vector databases.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

from src.workflows.io.schema import Chunk
from src.workflows.ml.embeddings import compute_batch_embeddings

LOGGER = logging.getLogger(__name__)


@dataclass
class ChunkEmbedding:
    """Container for a chunk with its embedding."""

    chunk: Chunk
    embedding: List[float]

    @property
    def id(self) -> str:
        """Get chunk ID."""
        return self.chunk.id

    @property
    def text(self) -> str:
        """Get chunk text."""
        return self.chunk.text

    @property
    def metadata(self) -> dict:
        """Get chunk metadata."""
        return self.chunk.metadata


def embed_chunks(
    chunks: List[Chunk],
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 32,
    normalize: bool = True,
) -> List[ChunkEmbedding]:
    """
    Generate embeddings for a list of chunks.

    Args:
        chunks: List of chunks to embed
        embedding_model: Name of the sentence-transformers model
        batch_size: Batch size for embedding generation
        normalize: Whether to normalize embeddings to unit length

    Returns:
        List of ChunkEmbedding objects with chunks and their embeddings
    """
    if not chunks:
        LOGGER.warning("No chunks provided for embedding")
        return []

    LOGGER.info(
        "Generating embeddings for %d chunks using %s",
        len(chunks),
        embedding_model
    )

    # Extract text from chunks
    texts = [chunk.text for chunk in chunks]

    # Generate embeddings
    embeddings = compute_batch_embeddings(
        texts,
        model_name=embedding_model,
        batch_size=batch_size,
        normalize=normalize,
    )

    # Combine chunks with embeddings
    chunk_embeddings = [
        ChunkEmbedding(chunk=chunk, embedding=embedding.tolist())
        for chunk, embedding in zip(chunks, embeddings)
    ]

    LOGGER.info("Generated %d embeddings", len(chunk_embeddings))

    return chunk_embeddings


def embed_single_chunk(
    chunk: Chunk,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    normalize: bool = True,
) -> ChunkEmbedding:
    """
    Generate embedding for a single chunk.

    Args:
        chunk: Chunk to embed
        embedding_model: Name of the sentence-transformers model
        normalize: Whether to normalize embedding to unit length

    Returns:
        ChunkEmbedding object with chunk and its embedding
    """
    chunk_embeddings = embed_chunks(
        [chunk],
        embedding_model=embedding_model,
        batch_size=1,
        normalize=normalize,
    )

    return chunk_embeddings[0]
