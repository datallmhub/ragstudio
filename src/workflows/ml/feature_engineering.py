"""
Feature engineering module for combining embeddings with structural features.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Union

import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

from src.workflows.ml.embeddings import compute_text_embedding, get_embedding_dimension

LOGGER = logging.getLogger(__name__)


def combine_features(
    text: str,
    profile: Dict[str, any],
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> np.ndarray:
    """
    Combine text embeddings with structural features.
    
    Args:
        text: Input text
        profile: Document profile from analyze_document()
        embedding_model: Name of the embedding model
        
    Returns:
        Combined feature vector
    """
    # Compute text embedding
    embedding = compute_text_embedding(text, model_name=embedding_model)
    
    # Extract structural features
    structural_features = extract_structural_features(profile)
    
    # Combine features
    combined = np.concatenate([embedding, structural_features])
    
    LOGGER.debug(
        "Combined features: embedding_dim=%d, structural_dim=%d, total_dim=%d",
        len(embedding), len(structural_features), len(combined)
    )
    
    return combined


def extract_structural_features(profile: Dict[str, any]) -> np.ndarray:
    """
    Extract structural features from document profile.
    
    Args:
        profile: Document profile from analyze_document()
        
    Returns:
        Array of structural features
    """
    features = []
    
    # Numeric features
    features.extend([
        profile.get("length_tokens", 0),
        profile.get("length_chars", 0),
        profile.get("hierarchy_depth", 1),
        profile.get("structure_score", 0.0),
        profile.get("avg_sentence_length", 0.0),
    ])
    
    # Boolean features (converted to 0/1)
    features.extend([
        int(profile.get("has_headings", False)),
        int(profile.get("has_tables", False)),
        int(profile.get("has_lists", False)),
    ])
    
    # Language encoding (simple one-hot for common languages)
    lang = profile.get("lang", "unknown").lower()
    lang_features = [0.0] * 4  # fr, en, es, other
    if lang == "fr":
        lang_features[0] = 1.0
    elif lang == "en":
        lang_features[1] = 1.0
    elif lang == "es":
        lang_features[2] = 1.0
    else:
        lang_features[3] = 1.0
    
    features.extend(lang_features)
    
    # Document type encoding (one-hot for common types)
    doc_type = profile.get("type", "unknown").lower()
    type_features = [0.0] * 5  # fiche_technique, rapport, article, document_court, other
    if doc_type == "fiche_technique":
        type_features[0] = 1.0
    elif doc_type == "rapport":
        type_features[1] = 1.0
    elif doc_type == "article":
        type_features[2] = 1.0
    elif doc_type == "document_court":
        type_features[3] = 1.0
    else:
        type_features[4] = 1.0
    
    features.extend(type_features)
    
    return np.array(features, dtype=np.float32)


def normalize_features(features: np.ndarray, scaler: Optional[StandardScaler] = None) -> np.ndarray:
    """
    Normalize features using StandardScaler.
    
    Args:
        features: Feature array
        scaler: Pre-fitted scaler (if None, will fit on the data)
        
    Returns:
        Normalized features
    """
    if scaler is None:
        scaler = StandardScaler()
        # Fit on the data (assuming features is 2D)
        if features.ndim == 1:
            features = features.reshape(1, -1)
        normalized = scaler.fit_transform(features)
    else:
        if features.ndim == 1:
            features = features.reshape(1, -1)
        normalized = scaler.transform(features)
    
    return normalized.flatten()


def get_feature_dimension(embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2") -> int:
    """
    Get the total dimension of combined features.
    
    Args:
        embedding_model: Name of the embedding model
        
    Returns:
        Total feature dimension
    """
    embedding_dim = get_embedding_dimension(embedding_model)
    structural_dim = 17  # 5 numeric + 3 boolean + 4 language + 5 type
    return embedding_dim + structural_dim


def create_feature_vector(
    text: str,
    profile: Dict[str, any],
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    normalize: bool = True,
    scaler: Optional[StandardScaler] = None
) -> np.ndarray:
    """
    Create a complete feature vector for ML training/prediction.
    
    Args:
        text: Input text
        profile: Document profile
        embedding_model: Name of the embedding model
        normalize: Whether to normalize features
        scaler: Pre-fitted scaler for normalization
        
    Returns:
        Complete feature vector
    """
    # Combine features
    features = combine_features(text, profile, embedding_model)
    
    # Normalize if requested
    if normalize:
        features = normalize_features(features, scaler)
    
    return features


def batch_create_feature_vectors(
    texts: List[str],
    profiles: List[Dict[str, any]],
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    normalize: bool = True
) -> np.ndarray:
    """
    Create feature vectors for a batch of texts and profiles.
    
    Args:
        texts: List of input texts
        profiles: List of document profiles
        embedding_model: Name of the embedding model
        normalize: Whether to normalize features
        
    Returns:
        Array of feature vectors
    """
    if len(texts) != len(profiles):
        raise ValueError("Number of texts must match number of profiles")
    
    feature_vectors = []
    
    for text, profile in zip(texts, profiles):
        features = create_feature_vector(text, profile, embedding_model, normalize=False)
        feature_vectors.append(features)
    
    feature_matrix = np.array(feature_vectors)
    
    # Normalize the entire batch if requested
    if normalize:
        scaler = StandardScaler()
        feature_matrix = scaler.fit_transform(feature_matrix)
    
    return feature_matrix


def encode_strategy_labels(strategies: List[str]) -> tuple[np.ndarray, LabelEncoder]:
    """
    Encode strategy labels for ML training.
    
    Args:
        strategies: List of strategy names
        
    Returns:
        Tuple of (encoded_labels, label_encoder)
    """
    encoder = LabelEncoder()
    encoded = encoder.fit_transform(strategies)
    return encoded, encoder


def decode_strategy_labels(encoded_labels: np.ndarray, encoder: LabelEncoder) -> List[str]:
    """
    Decode strategy labels from ML predictions.
    
    Args:
        encoded_labels: Encoded labels
        encoder: Label encoder used for encoding
        
    Returns:
        List of strategy names
    """
    return encoder.inverse_transform(encoded_labels).tolist()
