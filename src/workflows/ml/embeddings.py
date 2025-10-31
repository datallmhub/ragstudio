"""
Embedding generation module using Hugging Face Sentence Transformers.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Union

import numpy as np
from sentence_transformers import SentenceTransformer

LOGGER = logging.getLogger(__name__)

# Global model instance for efficiency
_EMBEDDING_MODEL: Optional[SentenceTransformer] = None


def get_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> SentenceTransformer:
    """
    Get or create the embedding model instance.
    
    Args:
        model_name: Name of the sentence transformer model
        
    Returns:
        SentenceTransformer model instance
    """
    global _EMBEDDING_MODEL
    
    if _EMBEDDING_MODEL is None:
        LOGGER.info("Loading embedding model: %s", model_name)
        _EMBEDDING_MODEL = SentenceTransformer(model_name)
        LOGGER.info("Embedding model loaded successfully")
    
    return _EMBEDDING_MODEL


def compute_text_embedding(
    text: str,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    normalize: bool = True
) -> np.ndarray:
    """
    Compute embedding for a single text.
    
    Args:
        text: Input text to embed
        model_name: Name of the sentence transformer model
        normalize: Whether to normalize the embedding
        
    Returns:
        Embedding vector as numpy array
    """
    if not text.strip():
        LOGGER.warning("Empty text provided for embedding")
        model = get_embedding_model(model_name)
        # Return zero vector with correct dimension
        return np.zeros(model.get_sentence_embedding_dimension())
    
    model = get_embedding_model(model_name)
    
    try:
        embedding = model.encode(text, normalize_embeddings=normalize)
        LOGGER.debug("Computed embedding for text of length %d", len(text))
        return embedding
    except Exception as exc:
        LOGGER.error("Failed to compute embedding: %s", exc)
        # Return zero vector as fallback
        return np.zeros(model.get_sentence_embedding_dimension())


def compute_batch_embeddings(
    texts: List[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    normalize: bool = True,
    batch_size: int = 32
) -> List[np.ndarray]:
    """
    Compute embeddings for a batch of texts.
    
    Args:
        texts: List of input texts to embed
        model_name: Name of the sentence transformer model
        normalize: Whether to normalize the embeddings
        batch_size: Batch size for processing
        
    Returns:
        List of embedding vectors
    """
    if not texts:
        return []
    
    model = get_embedding_model(model_name)
    
    try:
        embeddings = model.encode(
            texts,
            normalize_embeddings=normalize,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 100
        )
        LOGGER.info("Computed embeddings for %d texts", len(texts))
        return [emb for emb in embeddings]
    except Exception as exc:
        LOGGER.error("Failed to compute batch embeddings: %s", exc)
        # Return zero vectors as fallback
        dim = model.get_sentence_embedding_dimension()
        return [np.zeros(dim) for _ in texts]


def get_embedding_dimension(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> int:
    """
    Get the dimension of embeddings from the model.
    
    Args:
        model_name: Name of the sentence transformer model
        
    Returns:
        Embedding dimension
    """
    model = get_embedding_model(model_name)
    return model.get_sentence_embedding_dimension()


def clear_embedding_model() -> None:
    """Clear the global embedding model to free memory."""
    global _EMBEDDING_MODEL
    _EMBEDDING_MODEL = None
    LOGGER.info("Embedding model cleared from memory")
