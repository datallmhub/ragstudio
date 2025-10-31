"""
Unit tests for ML embeddings module.
"""

import numpy as np
import pytest

from src.workflows.ml.embeddings import (
    get_embedding_model,
    compute_text_embedding,
    compute_batch_embeddings,
    get_embedding_dimension,
    clear_embedding_model,
)


@pytest.fixture(autouse=True)
def cleanup_model():
    """Clear the global embedding model before and after each test."""
    clear_embedding_model()
    yield
    clear_embedding_model()


def test_get_embedding_model():
    """Test getting the embedding model instance."""
    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    model = get_embedding_model(model_name)

    assert model is not None
    assert model.get_sentence_embedding_dimension() == 384


def test_get_embedding_model_singleton():
    """Test that the same model instance is reused."""
    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    model1 = get_embedding_model(model_name)
    model2 = get_embedding_model(model_name)

    # Should be the same instance
    assert model1 is model2


def test_compute_text_embedding_basic():
    """Test computing embedding for a basic text."""
    text = "This is a test document."

    embedding = compute_text_embedding(text)

    assert isinstance(embedding, np.ndarray)
    assert len(embedding) == 384  # MiniLM-L6-v2 dimension
    assert embedding.dtype in [np.float32, np.float64]


def test_compute_text_embedding_empty_text():
    """Test computing embedding for empty text."""
    text = ""

    embedding = compute_text_embedding(text)

    # Should return zero vector
    assert isinstance(embedding, np.ndarray)
    assert len(embedding) == 384
    assert np.allclose(embedding, 0.0)


def test_compute_text_embedding_whitespace_only():
    """Test computing embedding for whitespace-only text."""
    text = "   \n\t  "

    embedding = compute_text_embedding(text)

    # Should return zero vector
    assert isinstance(embedding, np.ndarray)
    assert len(embedding) == 384
    assert np.allclose(embedding, 0.0)


def test_compute_text_embedding_normalized():
    """Test that embeddings are normalized by default."""
    text = "This is a test document."

    embedding = compute_text_embedding(text, normalize=True)

    # Normalized embeddings should have L2 norm close to 1
    norm = np.linalg.norm(embedding)
    assert np.isclose(norm, 1.0, atol=1e-6)


def test_compute_text_embedding_not_normalized():
    """Test computing non-normalized embeddings."""
    text = "This is a test document."

    embedding = compute_text_embedding(text, normalize=False)

    # Non-normalized embeddings might have norm != 1
    assert isinstance(embedding, np.ndarray)
    assert len(embedding) == 384


def test_compute_text_embedding_different_texts():
    """Test that different texts produce different embeddings."""
    text1 = "This is about cars."
    text2 = "This is about programming."

    embedding1 = compute_text_embedding(text1)
    embedding2 = compute_text_embedding(text2)

    # Embeddings should be different
    assert not np.allclose(embedding1, embedding2)


def test_compute_batch_embeddings_basic():
    """Test computing embeddings for a batch of texts."""
    texts = [
        "First document about cars.",
        "Second document about programming.",
        "Third document about cooking.",
    ]

    embeddings = compute_batch_embeddings(texts)

    assert isinstance(embeddings, list)
    assert len(embeddings) == 3
    assert all(isinstance(emb, np.ndarray) for emb in embeddings)
    assert all(len(emb) == 384 for emb in embeddings)


def test_compute_batch_embeddings_empty_list():
    """Test computing embeddings for an empty list."""
    texts = []

    embeddings = compute_batch_embeddings(texts)

    assert embeddings == []


def test_compute_batch_embeddings_single_item():
    """Test computing embeddings for a single-item batch."""
    texts = ["Single document."]

    embeddings = compute_batch_embeddings(texts)

    assert len(embeddings) == 1
    assert isinstance(embeddings[0], np.ndarray)
    assert len(embeddings[0]) == 384


def test_compute_batch_embeddings_normalized():
    """Test that batch embeddings are normalized."""
    texts = [
        "First document.",
        "Second document.",
    ]

    embeddings = compute_batch_embeddings(texts, normalize=True)

    # All embeddings should have L2 norm close to 1
    for embedding in embeddings:
        norm = np.linalg.norm(embedding)
        assert np.isclose(norm, 1.0, atol=1e-6)


def test_compute_batch_embeddings_custom_batch_size():
    """Test computing embeddings with custom batch size."""
    texts = ["Document " + str(i) for i in range(10)]

    embeddings = compute_batch_embeddings(texts, batch_size=2)

    assert len(embeddings) == 10
    assert all(len(emb) == 384 for emb in embeddings)


def test_get_embedding_dimension():
    """Test getting the embedding dimension."""
    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    dimension = get_embedding_dimension(model_name)

    assert dimension == 384


def test_clear_embedding_model():
    """Test clearing the global embedding model."""
    # Load the model
    get_embedding_model()

    # Clear it
    clear_embedding_model()

    # After clearing, a new call should reload the model
    model = get_embedding_model()
    assert model is not None


def test_embeddings_consistency():
    """Test that the same text produces the same embedding."""
    text = "Test document for consistency."

    embedding1 = compute_text_embedding(text)
    embedding2 = compute_text_embedding(text)

    # Should produce identical embeddings
    assert np.allclose(embedding1, embedding2)


def test_batch_vs_single_embedding_consistency():
    """Test that batch and single embeddings are consistent."""
    text = "Test document for consistency."

    single_embedding = compute_text_embedding(text)
    batch_embeddings = compute_batch_embeddings([text])

    # Should produce the same embedding
    assert np.allclose(single_embedding, batch_embeddings[0])
