"""
Unit tests for ML feature engineering module.
"""

import numpy as np
import pytest
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.workflows.ml.feature_engineering import (
    combine_features,
    create_feature_vector,
    extract_structural_features,
    normalize_features,
    get_feature_dimension,
    encode_strategy_labels,
    decode_strategy_labels,
    batch_create_feature_vectors,
)


def test_extract_structural_features_basic():
    """Test extraction of basic structural features."""
    profile = {
        "length_tokens": 100,
        "length_chars": 500,
        "hierarchy_depth": 2,
        "structure_score": 0.7,
        "avg_sentence_length": 20.0,
        "has_headings": True,
        "has_tables": False,
        "has_lists": True,
        "lang": "fr",
        "type": "fiche_technique",
    }

    features = extract_structural_features(profile)

    # Check shape and type
    assert isinstance(features, np.ndarray)
    assert features.dtype == np.float32

    # Check that we have 17 features (5 numeric + 3 boolean + 4 language + 5 type)
    assert len(features) == 17

    # Check numeric features
    assert features[0] == pytest.approx(100)  # length_tokens
    assert features[1] == pytest.approx(500)  # length_chars
    assert features[2] == pytest.approx(2)    # hierarchy_depth
    assert features[3] == pytest.approx(0.7)  # structure_score
    assert features[4] == pytest.approx(20.0)  # avg_sentence_length

    # Check boolean features (converted to 0/1)
    assert features[5] == 1  # has_headings
    assert features[6] == 0  # has_tables
    assert features[7] == 1  # has_lists

    # Check language encoding (fr should be 1, others 0)
    assert features[8] == 1.0  # fr
    assert features[9] == 0.0  # en
    assert features[10] == 0.0  # es
    assert features[11] == 0.0  # other


def test_extract_structural_features_missing_values():
    """Test extraction with missing optional values."""
    profile = {
        "length_tokens": 50,
    }

    features = extract_structural_features(profile)

    # Should use defaults for missing values
    assert len(features) == 17
    assert features[0] == pytest.approx(50)
    assert features[1] == pytest.approx(0)  # default length_chars


def test_extract_structural_features_language_encoding():
    """Test language one-hot encoding."""
    languages = ["fr", "en", "es", "de"]
    expected_encodings = [
        [1.0, 0.0, 0.0, 0.0],  # fr
        [0.0, 1.0, 0.0, 0.0],  # en
        [0.0, 0.0, 1.0, 0.0],  # es
        [0.0, 0.0, 0.0, 1.0],  # de (other)
    ]

    for lang, expected in zip(languages, expected_encodings):
        profile = {"lang": lang}
        features = extract_structural_features(profile)
        # Language features are at indices 8-11
        assert list(features[8:12]) == expected


def test_normalize_features_single_vector():
    """Test normalization of a single feature vector."""
    features = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    # Without pre-fitted scaler
    normalized = normalize_features(features)

    assert isinstance(normalized, np.ndarray)
    assert normalized.shape == features.shape
    # StandardScaler should make mean ~0 and std ~1 for this single sample
    assert normalized.mean() == pytest.approx(0.0, abs=1e-10)


def test_normalize_features_with_scaler():
    """Test normalization with a pre-fitted scaler."""
    train_features = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    test_features = np.array([2.0, 3.0, 4.0])

    # Fit scaler on training data
    scaler = StandardScaler()
    scaler.fit(train_features)

    # Normalize test features with fitted scaler
    normalized = normalize_features(test_features, scaler)

    assert isinstance(normalized, np.ndarray)
    assert len(normalized) == 3


def test_encode_strategy_labels():
    """Test encoding of strategy labels."""
    strategies = ["semantic", "recursive", "late", "semantic", "recursive"]

    encoded, encoder = encode_strategy_labels(strategies)

    # Check types
    assert isinstance(encoded, np.ndarray)
    assert isinstance(encoder, LabelEncoder)

    # Check shape
    assert len(encoded) == len(strategies)

    # Check that same strategies have same encoding
    assert encoded[0] == encoded[3]  # both "semantic"
    assert encoded[1] == encoded[4]  # both "recursive"

    # Check classes
    assert set(encoder.classes_) == {"semantic", "recursive", "late"}


def test_decode_strategy_labels():
    """Test decoding of strategy labels."""
    strategies = ["semantic", "recursive", "late"]
    encoded, encoder = encode_strategy_labels(strategies)

    decoded = decode_strategy_labels(encoded, encoder)

    assert decoded == strategies


def test_get_feature_dimension():
    """Test getting total feature dimension."""
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"

    total_dim = get_feature_dimension(embedding_model)

    # Should be embedding_dim (384 for MiniLM-L6-v2) + 17 structural features
    assert total_dim == 384 + 17
    assert total_dim == 401


def test_combine_features():
    """Test combining embeddings with structural features."""
    text = "Test document with some content."
    profile = {
        "length_tokens": 10,
        "length_chars": len(text),
        "hierarchy_depth": 1,
        "structure_score": 0.5,
        "avg_sentence_length": 10.0,
        "has_headings": False,
        "has_tables": False,
        "has_lists": False,
        "lang": "en",
        "type": "article",
    }

    features = combine_features(text, profile)

    # Should be embeddings (384) + structural (17) = 401
    assert len(features) == 401
    assert isinstance(features, np.ndarray)


def test_create_feature_vector_no_normalization():
    """Test creating feature vector without normalization."""
    text = "Sample text for testing."
    profile = {
        "length_tokens": 5,
        "length_chars": len(text),
        "hierarchy_depth": 1,
        "structure_score": 0.3,
        "avg_sentence_length": 5.0,
        "lang": "en",
    }

    features = create_feature_vector(text, profile, normalize=False)

    assert isinstance(features, np.ndarray)
    assert len(features) == 401


def test_create_feature_vector_with_normalization():
    """Test creating feature vector with normalization."""
    text = "Sample text for testing."
    profile = {
        "length_tokens": 5,
        "length_chars": len(text),
        "hierarchy_depth": 1,
        "structure_score": 0.3,
        "avg_sentence_length": 5.0,
        "lang": "en",
    }

    features = create_feature_vector(text, profile, normalize=True)

    assert isinstance(features, np.ndarray)
    assert len(features) == 401


def test_batch_create_feature_vectors():
    """Test creating feature vectors for a batch."""
    texts = [
        "First document",
        "Second document",
        "Third document",
    ]
    profiles = [
        {"length_tokens": 2, "lang": "en"},
        {"length_tokens": 2, "lang": "en"},
        {"length_tokens": 2, "lang": "en"},
    ]

    feature_matrix = batch_create_feature_vectors(texts, profiles, normalize=False)

    # Check shape
    assert feature_matrix.shape == (3, 401)
    assert isinstance(feature_matrix, np.ndarray)


def test_batch_create_feature_vectors_with_normalization():
    """Test batch feature creation with normalization."""
    texts = [
        "First document",
        "Second document",
        "Third document",
    ]
    profiles = [
        {"length_tokens": 2, "lang": "en"},
        {"length_tokens": 2, "lang": "en"},
        {"length_tokens": 2, "lang": "en"},
    ]

    feature_matrix = batch_create_feature_vectors(texts, profiles, normalize=True)

    # Check shape
    assert feature_matrix.shape == (3, 401)

    # After normalization, mean should be close to 0 for each feature
    # Using reasonable tolerance for float32 precision
    feature_means = feature_matrix.mean(axis=0)
    assert np.allclose(feature_means, 0.0, atol=1e-6)


def test_batch_create_feature_vectors_mismatched_lengths():
    """Test that mismatched texts and profiles raises error."""
    texts = ["First document", "Second document"]
    profiles = [{"length_tokens": 2}]  # Only one profile

    with pytest.raises(ValueError, match="Number of texts must match number of profiles"):
        batch_create_feature_vectors(texts, profiles)
