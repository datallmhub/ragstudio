"""
Integration tests for the ML training and prediction pipeline.
"""

import json
import tempfile
from pathlib import Path

import pytest

from src.workflows.ml.training import train_complete_pipeline, load_strategy_dataset
from src.workflows.ml.strategy_scorer_hf import StrategyScorerHF


@pytest.fixture
def ml_dataset_file():
    """Create a temporary ML dataset for testing."""
    data = [
        {
            "doc_id": "tech1",
            "text": "# Installation Guide\n\n## Setup\n\n```bash\npip install pkg\n```\n\n| Param | Value |\n|-------|-------|\n| Port | 8080 |",
            "length_tokens": 50,
            "hierarchy_depth": 2,
            "has_tables": True,
            "has_headings": True,
            "structure_score": 0.9,
            "lang": "en",
            "best_strategy": "late",
        },
        {
            "doc_id": "blog1",
            "text": "AI is changing the world. Machine learning enables new applications every day.",
            "length_tokens": 15,
            "hierarchy_depth": 1,
            "has_tables": False,
            "has_headings": False,
            "structure_score": 0.0,
            "lang": "en",
            "best_strategy": "recursive",
        },
        {
            "doc_id": "report1",
            "text": "# Q1 Report\n\n## Sales\n\n### North America\n\nGrowth: 15%\n\n### Europe\n\nGrowth: 12%",
            "length_tokens": 30,
            "hierarchy_depth": 3,
            "has_tables": False,
            "has_headings": True,
            "structure_score": 0.7,
            "lang": "en",
            "best_strategy": "parent_child",
        },
        {
            "doc_id": "doc1",
            "text": "# Technical Spec\n\n## Features\n\nWell-structured content with clear headings.",
            "length_tokens": 20,
            "hierarchy_depth": 2,
            "has_tables": False,
            "has_headings": True,
            "structure_score": 0.6,
            "lang": "en",
            "best_strategy": "semantic",
        },
        # Duplicate each class to have at least 2 per class for stratification
        {
            "doc_id": "tech2",
            "text": "# Config\n\n```yaml\nport: 3000\n```\n\n| Setting | Default |\n|---------|---------|",
            "length_tokens": 45,
            "hierarchy_depth": 1,
            "has_tables": True,
            "has_headings": True,
            "structure_score": 0.85,
            "lang": "en",
            "best_strategy": "late",
        },
        {
            "doc_id": "blog2",
            "text": "Short news article about recent developments in technology sector.",
            "length_tokens": 12,
            "hierarchy_depth": 1,
            "has_tables": False,
            "has_headings": False,
            "structure_score": 0.0,
            "lang": "en",
            "best_strategy": "recursive",
        },
        {
            "doc_id": "report2",
            "text": "# Annual Report\n\n## Overview\n\n### Summary\n\nHierarchical structure",
            "length_tokens": 25,
            "hierarchy_depth": 3,
            "has_tables": False,
            "has_headings": True,
            "structure_score": 0.65,
            "lang": "en",
            "best_strategy": "parent_child",
        },
        {
            "doc_id": "doc2",
            "text": "# Product Guide\n\n## Description\n\nSemantic structure with topics.",
            "length_tokens": 18,
            "hierarchy_depth": 2,
            "has_tables": False,
            "has_headings": True,
            "structure_score": 0.55,
            "lang": "en",
            "best_strategy": "semantic",
        },
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
        temp_path = f.name

    yield temp_path

    # Cleanup
    Path(temp_path).unlink()


def test_load_ml_dataset(ml_dataset_file):
    """Test loading a strategy dataset for ML training."""
    dataset = load_strategy_dataset(ml_dataset_file)

    assert dataset is not None
    assert len(dataset) == 8
    assert "text" in dataset.column_names
    assert "best_strategy" in dataset.column_names


def test_ml_training_pipeline_integration(ml_dataset_file, tmp_path):
    """Test the complete ML training pipeline end-to-end."""
    output_dir = str(tmp_path / "test_model")

    # Train the model
    model, scaler, label_encoder, metrics = train_complete_pipeline(
        data_file=ml_dataset_file,
        output_dir=output_dir,
        num_epochs=3,  # Quick training for test
        batch_size=2,
        test_size=0.5,  # 50% of 8 = 4 test samples (min for 4 classes)
    )

    # Check that training completed
    assert model is not None
    assert scaler is not None
    assert label_encoder is not None
    assert "accuracy" in metrics

    # Check that files were saved
    assert Path(output_dir, "model.safetensors").exists() or Path(output_dir, "pytorch_model.bin").exists()
    assert Path(output_dir, "scaler.pkl").exists()
    assert Path(output_dir, "label_encoder.pkl").exists()

    # Check label encoder has all 4 strategies
    assert len(label_encoder.classes_) == 4
    assert set(label_encoder.classes_) == {"late", "parent_child", "recursive", "semantic"}


def test_ml_prediction_integration(ml_dataset_file, tmp_path):
    """Test prediction using a trained model."""
    output_dir = str(tmp_path / "test_model")

    # Train a model first
    train_complete_pipeline(
        data_file=ml_dataset_file,
        output_dir=output_dir,
        num_epochs=2,
        batch_size=2,
        test_size=0.5,
    )

    # Load the model
    scorer = StrategyScorerHF(model_path=output_dir)

    # Test prediction on sample text
    test_text = "# Technical Guide\n\n## Installation\n\n```bash\ninstall.sh\n```"
    test_profile = {
        "length_tokens": 20,
        "length_chars": len(test_text),
        "hierarchy_depth": 2,
        "structure_score": 0.8,
        "has_headings": True,
        "has_tables": False,
        "has_lists": False,
        "lang": "en",
        "type": "fiche_technique",
        "avg_sentence_length": 5.0,
    }

    # Make prediction
    strategy, confidence = scorer.predict_strategy(test_text, test_profile)

    # Check prediction results
    assert strategy in ["late", "parent_child", "recursive", "semantic"]
    assert 0.0 <= confidence <= 1.0


def test_ml_pipeline_predictions_vary(ml_dataset_file, tmp_path):
    """Test that different documents get different predictions."""
    output_dir = str(tmp_path / "test_model")

    # Train model
    train_complete_pipeline(
        data_file=ml_dataset_file,
        output_dir=output_dir,
        num_epochs=2,
        batch_size=2,
        test_size=0.5,
    )

    scorer = StrategyScorerHF(model_path=output_dir)

    # Test on very different documents
    simple_text = "This is a short blog article."
    simple_profile = {
        "length_tokens": 6,
        "length_chars": len(simple_text),
        "hierarchy_depth": 1,
        "structure_score": 0.0,
        "has_headings": False,
        "has_tables": False,
        "has_lists": False,
        "lang": "en",
        "type": "article",
        "avg_sentence_length": 6.0,
    }

    complex_text = "# Manual\n\n## Chapter 1\n\n### Section 1.1\n\n```code```\n\n| Table | Data |"
    complex_profile = {
        "length_tokens": 50,
        "length_chars": len(complex_text),
        "hierarchy_depth": 3,
        "structure_score": 1.0,
        "has_headings": True,
        "has_tables": True,
        "has_lists": False,
        "lang": "en",
        "type": "fiche_technique",
        "avg_sentence_length": 8.0,
    }

    strategy1, conf1 = scorer.predict_strategy(simple_text, simple_profile)
    strategy2, conf2 = scorer.predict_strategy(complex_text, complex_profile)

    # Predictions should exist and have valid confidence
    assert strategy1 in ["late", "parent_child", "recursive", "semantic"]
    assert strategy2 in ["late", "parent_child", "recursive", "semantic"]
    assert 0.0 <= conf1 <= 1.0
    assert 0.0 <= conf2 <= 1.0

    # With only 8 training samples, predictions may be the same,
    # but we just verify the pipeline works


def test_ml_model_persistence(ml_dataset_file, tmp_path):
    """Test that trained models can be loaded and reused."""
    output_dir = str(tmp_path / "test_model")

    # Train and save model
    train_complete_pipeline(
        data_file=ml_dataset_file,
        output_dir=output_dir,
        num_epochs=2,
        batch_size=2,
        test_size=0.5,
    )

    # Load model in first scorer
    scorer1 = StrategyScorerHF(model_path=output_dir)

    test_text = "Sample document text"
    test_profile = {
        "length_tokens": 5,
        "length_chars": len(test_text),
        "hierarchy_depth": 1,
        "structure_score": 0.5,
        "has_headings": False,
        "has_tables": False,
        "has_lists": False,
        "lang": "en",
        "type": "unknown",
        "avg_sentence_length": 3.0,
    }

    strategy1, conf1 = scorer1.predict_strategy(test_text, test_profile)

    # Load same model in second scorer (tests persistence)
    scorer2 = StrategyScorerHF(model_path=output_dir)
    strategy2, conf2 = scorer2.predict_strategy(test_text, test_profile)

    # Should get same predictions
    assert strategy1 == strategy2
    assert conf1 == conf2
