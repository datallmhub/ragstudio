"""
Model training module for strategy scoring using Hugging Face Transformers.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import torch
from datasets import Dataset, load_dataset
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    AutoTokenizer,
)

from src.workflows.ml.feature_engineering import (
    batch_create_feature_vectors,
    encode_strategy_labels,
    get_feature_dimension,
)
from src.workflows.ml.embeddings import get_embedding_dimension

LOGGER = logging.getLogger(__name__)


class StrategyDataset(torch.utils.data.Dataset):
    """Custom dataset for strategy classification."""
    
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return {
            "input_values": self.features[idx],
            "labels": self.labels[idx]
        }


class StrategyScorerModel(torch.nn.Module):
    """Custom model for strategy scoring."""
    
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 256):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Simple feedforward network
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, input_values, labels=None):
        logits = self.layers(input_values)
        
        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        
        return {"loss": loss, "logits": logits}


def load_strategy_dataset(data_file: str = "data/strategy_samples.jsonl") -> Dataset:
    """
    Load the strategy dataset from JSONL file.
    
    Args:
        data_file: Path to the JSONL data file
        
    Returns:
        Hugging Face Dataset
    """
    LOGGER.info("Loading strategy dataset from %s", data_file)
    
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Dataset file not found: {data_file}")

    # Loading from local JSON file, not downloading from Hub
    dataset = load_dataset("json", data_files=data_file, split="train")  # nosec B615
    LOGGER.info("Loaded %d samples from dataset", len(dataset))
    
    return dataset


def prepare_training_data(
    dataset: Dataset,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, LabelEncoder]:
    """
    Prepare training data from the dataset.
    
    Args:
        dataset: Hugging Face Dataset
        embedding_model: Name of the embedding model
        test_size: Fraction of data to use for testing
        random_state: Random state for reproducibility
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, scaler, label_encoder)
    """
    LOGGER.info("Preparing training data...")
    
    # Extract texts and profiles
    texts = [example["text"] for example in dataset]
    profiles = [
        {
            "length_tokens": example["length_tokens"],
            "length_chars": len(example["text"]),
            "hierarchy_depth": example["hierarchy_depth"],
            "structure_score": example["structure_score"],
            "avg_sentence_length": len(example["text"]) / max(example["text"].count("."), 1),
            "has_headings": example["has_headings"],
            "has_tables": example["has_tables"],
            "has_lists": example.get("has_lists", False),
            "lang": example["lang"],
            "type": example.get("type", "unknown"),
        }
        for example in dataset
    ]
    strategies = [example["best_strategy"] for example in dataset]
    
    # Create feature vectors
    LOGGER.info("Creating feature vectors...")
    X = batch_create_feature_vectors(texts, profiles, embedding_model, normalize=False)
    
    # Encode labels
    y, label_encoder = encode_strategy_labels(strategies)
    
    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    LOGGER.info(
        "Training data prepared: train_size=%d, test_size=%d, feature_dim=%d, num_classes=%d",
        len(X_train), len(X_test), X.shape[1], len(label_encoder.classes_)
    )
    
    return X_train, X_test, y_train, y_test, scaler, label_encoder


def train_strategy_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    num_classes: int,
    output_dir: str = "data/models/strategy_scorer_hf",
    num_epochs: int = 10,
    learning_rate: float = 1e-3,
    batch_size: int = 8
) -> Tuple[StrategyScorerModel, Dict[str, float]]:
    """
    Train the strategy scoring model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        num_classes: Number of strategy classes
        output_dir: Directory to save the model
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        batch_size: Batch size
        
    Returns:
        Tuple of (trained_model, metrics)
    """
    LOGGER.info("Starting model training...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create datasets
    train_dataset = StrategyDataset(X_train, y_train)
    test_dataset = StrategyDataset(X_test, y_test)
    
    # Initialize model
    input_dim = X_train.shape[1]
    model = StrategyScorerModel(input_dim, num_classes)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",  # Changed from evaluation_strategy
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=None,  # Disable wandb/tensorboard
    )
    
    # Custom collate function
    def collate_fn(batch):
        input_values = torch.stack([item["input_values"] for item in batch])
        labels = torch.stack([item["labels"] for item in batch])
        return {"input_values": input_values, "labels": labels}
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=collate_fn,
    )
    
    # Train the model
    LOGGER.info("Training model for %d epochs...", num_epochs)
    trainer.train()
    
    # Evaluate the model
    LOGGER.info("Evaluating model...")
    eval_results = trainer.evaluate()
    
    # Get predictions for detailed metrics
    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    metrics = {
        "accuracy": accuracy,
        "eval_loss": eval_results["eval_loss"],
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
    }
    
    LOGGER.info("Training completed. Accuracy: %.3f", accuracy)
    
    # Save the model
    trainer.save_model()
    LOGGER.info("Model saved to %s", output_dir)
    
    return model, metrics


def train_complete_pipeline(
    data_file: str = "data/strategy_samples.jsonl",
    output_dir: str = "data/models/strategy_scorer_hf",
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    num_epochs: int = 10,
    learning_rate: float = 1e-3,
    batch_size: int = 8,
    test_size: float = 0.2
) -> Tuple[StrategyScorerModel, StandardScaler, LabelEncoder, Dict[str, float]]:
    """
    Complete training pipeline.

    Args:
        data_file: Path to the dataset file
        output_dir: Directory to save the model
        embedding_model: Name of the embedding model
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        batch_size: Batch size
        test_size: Fraction of data to use for testing

    Returns:
        Tuple of (model, scaler, label_encoder, metrics)
    """
    LOGGER.info("Starting complete training pipeline...")

    # Load dataset
    dataset = load_strategy_dataset(data_file)

    # Prepare training data
    X_train, X_test, y_train, y_test, scaler, label_encoder = prepare_training_data(
        dataset, embedding_model, test_size=test_size
    )
    
    # Train model
    model, metrics = train_strategy_model(
        X_train, y_train, X_test, y_test,
        num_classes=len(label_encoder.classes_),
        output_dir=output_dir,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size
    )
    
    # Save scaler and label encoder
    scaler_path = os.path.join(output_dir, "scaler.pkl")
    encoder_path = os.path.join(output_dir, "label_encoder.pkl")
    
    joblib.dump(scaler, scaler_path)
    joblib.dump(label_encoder, encoder_path)
    
    LOGGER.info("Scaler saved to %s", scaler_path)
    LOGGER.info("Label encoder saved to %s", encoder_path)
    
    return model, scaler, label_encoder, metrics


if __name__ == "__main__":
    # Train the model when run as script
    model, scaler, label_encoder, metrics = train_complete_pipeline()
    print(f"Training completed with accuracy: {metrics['accuracy']:.3f}")
