"""
Unit tests for ML training module.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from src.workflows.ml.training import (
    StrategyDataset,
    StrategyScorerModel,
    load_strategy_dataset,
    prepare_training_data,
)


@pytest.fixture
def sample_dataset_file():
    """Create a temporary dataset file for testing."""
    data = [
        {
            "doc_id": "doc1",
            "text": "This is a technical document with headings and tables.",
            "length_tokens": 100,
            "hierarchy_depth": 3,
            "has_tables": True,
            "has_headings": True,
            "structure_score": 0.8,
            "lang": "en",
            "best_strategy": "semantic",
        },
        {
            "doc_id": "doc2",
            "text": "Short news article without much structure.",
            "length_tokens": 30,
            "hierarchy_depth": 1,
            "has_tables": False,
            "has_headings": False,
            "structure_score": 0.2,
            "lang": "en",
            "best_strategy": "recursive",
        },
        {
            "doc_id": "doc3",
            "text": "A manual with clear hierarchical structure and code examples.",
            "length_tokens": 200,
            "hierarchy_depth": 4,
            "has_tables": True,
            "has_headings": True,
            "structure_score": 0.9,
            "lang": "en",
            "best_strategy": "late",
        },
        {
            "doc_id": "doc4",
            "text": "Another semantic document with good structure.",
            "length_tokens": 120,
            "hierarchy_depth": 2,
            "has_tables": False,
            "has_headings": True,
            "structure_score": 0.7,
            "lang": "fr",
            "best_strategy": "semantic",
        },
        {
            "doc_id": "doc5",
            "text": "Second recursive document for testing.",
            "length_tokens": 40,
            "hierarchy_depth": 1,
            "has_tables": False,
            "has_headings": False,
            "structure_score": 0.3,
            "lang": "fr",
            "best_strategy": "recursive",
        },
        {
            "doc_id": "doc6",
            "text": "Another late chunking document with code.",
            "length_tokens": 180,
            "hierarchy_depth": 3,
            "has_tables": True,
            "has_headings": True,
            "structure_score": 0.85,
            "lang": "en",
            "best_strategy": "late",
        },
        {
            "doc_id": "doc7",
            "text": "Third semantic document for better testing.",
            "length_tokens": 110,
            "hierarchy_depth": 2,
            "has_tables": True,
            "has_headings": True,
            "structure_score": 0.75,
            "lang": "en",
            "best_strategy": "semantic",
        },
        {
            "doc_id": "doc8",
            "text": "Third recursive article example.",
            "length_tokens": 35,
            "hierarchy_depth": 1,
            "has_tables": False,
            "has_headings": False,
            "structure_score": 0.25,
            "lang": "fr",
            "best_strategy": "recursive",
        },
        {
            "doc_id": "doc9",
            "text": "Third late chunking manual with technical content.",
            "length_tokens": 190,
            "hierarchy_depth": 4,
            "has_tables": True,
            "has_headings": True,
            "structure_score": 0.9,
            "lang": "en",
            "best_strategy": "late",
        },
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
        temp_path = f.name

    yield temp_path

    # Cleanup
    Path(temp_path).unlink()


def test_strategy_dataset_creation():
    """Test creating a StrategyDataset."""
    features = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    labels = np.array([0, 1], dtype=np.int64)

    dataset = StrategyDataset(features, labels)

    assert len(dataset) == 2

    # Check first item
    item = dataset[0]
    assert "input_values" in item
    assert "labels" in item
    assert torch.is_tensor(item["input_values"])
    assert torch.is_tensor(item["labels"])


def test_strategy_dataset_getitem():
    """Test getting items from StrategyDataset."""
    features = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    labels = np.array([0], dtype=np.int64)

    dataset = StrategyDataset(features, labels)

    item = dataset[0]

    assert torch.allclose(item["input_values"], torch.tensor([1.0, 2.0, 3.0]))
    assert item["labels"] == 0


def test_strategy_scorer_model_initialization():
    """Test initializing the StrategyScorerModel."""
    input_dim = 401  # Embedding (384) + structural (17)
    num_classes = 4  # semantic, recursive, late, parent_child
    hidden_dim = 128

    model = StrategyScorerModel(input_dim, num_classes, hidden_dim)

    assert model.input_dim == input_dim
    assert model.num_classes == num_classes
    assert model.layers is not None


def test_strategy_scorer_model_forward():
    """Test forward pass through the model."""
    input_dim = 10
    num_classes = 3
    batch_size = 4

    model = StrategyScorerModel(input_dim, num_classes, hidden_dim=16)

    # Create random input
    input_values = torch.randn(batch_size, input_dim)

    # Forward pass without labels
    output = model(input_values)

    assert "logits" in output
    assert output["logits"].shape == (batch_size, num_classes)
    assert output["loss"] is None


def test_strategy_scorer_model_forward_with_labels():
    """Test forward pass with labels for loss calculation."""
    input_dim = 10
    num_classes = 3
    batch_size = 4

    model = StrategyScorerModel(input_dim, num_classes, hidden_dim=16)

    # Create random input and labels
    input_values = torch.randn(batch_size, input_dim)
    labels = torch.randint(0, num_classes, (batch_size,))

    # Forward pass with labels
    output = model(input_values, labels=labels)

    assert "logits" in output
    assert "loss" in output
    assert output["logits"].shape == (batch_size, num_classes)
    assert output["loss"] is not None
    assert torch.is_tensor(output["loss"])
    assert output["loss"].item() > 0


def test_load_strategy_dataset(sample_dataset_file):
    """Test loading a strategy dataset from JSONL file."""
    dataset = load_strategy_dataset(sample_dataset_file)

    assert dataset is not None
    assert len(dataset) == 9  # Updated for 9 samples (3 of each class)

    # Check first example
    first_example = dataset[0]
    assert "text" in first_example
    assert "best_strategy" in first_example
    assert "length_tokens" in first_example
    assert first_example["doc_id"] == "doc1"


def test_load_strategy_dataset_missing_file():
    """Test loading a dataset from a non-existent file."""
    with pytest.raises(FileNotFoundError):
        load_strategy_dataset("nonexistent_file.jsonl")


def test_prepare_training_data(sample_dataset_file):
    """Test preparing training data from a dataset."""
    dataset = load_strategy_dataset(sample_dataset_file)

    X_train, X_test, y_train, y_test, scaler, label_encoder = prepare_training_data(
        dataset, test_size=0.33, random_state=42  # 33% of 9 = 3 test samples
    )

    # Check shapes (9 samples: 6 train, 3 test)
    assert X_train.shape[0] == 6  # ~67% of 9
    assert X_test.shape[0] == 3   # ~33% of 9
    assert X_train.shape[1] == 401  # Feature dimension
    assert X_test.shape[1] == 401

    # Check labels
    assert len(y_train) == 6
    assert len(y_test) == 3

    # Check scaler and encoder
    assert scaler is not None
    assert label_encoder is not None
    assert len(label_encoder.classes_) == 3  # semantic, recursive, late


def test_prepare_training_data_stratification(sample_dataset_file):
    """Test that stratification preserves class distribution."""
    dataset = load_strategy_dataset(sample_dataset_file)

    X_train, X_test, y_train, y_test, scaler, label_encoder = prepare_training_data(
        dataset, test_size=0.33, random_state=42
    )

    # Count classes in original data
    all_labels = np.concatenate([y_train, y_test])
    unique, counts = np.unique(all_labels, return_counts=True)

    # Should have 3 of each class (semantic, recursive, late)
    assert len(unique) == 3
    assert all(count == 3 for count in counts)  # Each class appears 3 times


def test_prepare_training_data_normalized(sample_dataset_file):
    """Test that features are normalized."""
    dataset = load_strategy_dataset(sample_dataset_file)

    X_train, X_test, y_train, y_test, scaler, label_encoder = prepare_training_data(
        dataset, test_size=0.33, random_state=42
    )

    # After normalization, each feature should have mean ~0 and std ~1
    # (approximately, given small sample size)
    train_means = X_train.mean(axis=0)
    train_stds = X_train.std(axis=0)

    # Most features should be relatively centered
    assert np.abs(train_means).mean() < 1.0
    # Stds should be around 1 (may vary due to small sample)
    assert 0.3 < train_stds.mean() < 2.0  # Relaxed bounds for small sample


def test_strategy_scorer_model_output_shape():
    """Test that model output has correct shape."""
    input_dim = 401
    num_classes = 4
    model = StrategyScorerModel(input_dim, num_classes)

    # Single example
    x = torch.randn(1, input_dim)
    output = model(x)

    assert output["logits"].shape == (1, num_classes)

    # Batch
    x_batch = torch.randn(8, input_dim)
    output_batch = model(x_batch)

    assert output_batch["logits"].shape == (8, num_classes)


def test_strategy_dataset_len():
    """Test __len__ method of StrategyDataset."""
    features = np.random.rand(10, 5).astype(np.float32)
    labels = np.random.randint(0, 3, 10).astype(np.int64)

    dataset = StrategyDataset(features, labels)

    assert len(dataset) == 10
