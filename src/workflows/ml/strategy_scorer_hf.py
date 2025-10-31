"""
Strategy scorer module using trained Hugging Face model for prediction.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.workflows.ml.embeddings import compute_text_embedding
from src.workflows.ml.feature_engineering import create_feature_vector
from src.workflows.ml.training import StrategyScorerModel

LOGGER = logging.getLogger(__name__)


class StrategyScorerHF:
    """Strategy scorer using trained Hugging Face model."""
    
    def __init__(
        self,
        model_path: str = "data/models/strategy_scorer_hf",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        """
        Initialize the strategy scorer.
        
        Args:
            model_path: Path to the trained model directory
            embedding_model: Name of the embedding model
        """
        self.model_path = model_path
        self.embedding_model = embedding_model
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        LOGGER.info("StrategyScorerHF initialized with device: %s", self.device)
    
    def load_model(self) -> None:
        """Load the trained model and preprocessing components."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model path not found: {self.model_path}")
        
        LOGGER.info("Loading model from %s", self.model_path)
        
        # Load scaler
        scaler_path = os.path.join(self.model_path, "scaler.pkl")
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            LOGGER.info("Scaler loaded from %s", scaler_path)
        else:
            LOGGER.warning("Scaler not found at %s", scaler_path)
        
        # Load label encoder
        encoder_path = os.path.join(self.model_path, "label_encoder.pkl")
        if os.path.exists(encoder_path):
            self.label_encoder = joblib.load(encoder_path)
            LOGGER.info("Label encoder loaded from %s", encoder_path)
        else:
            LOGGER.warning("Label encoder not found at %s", encoder_path)
        
        # Load model - try both .bin and .safetensors formats
        model_file_bin = os.path.join(self.model_path, "pytorch_model.bin")
        model_file_safetensors = os.path.join(self.model_path, "model.safetensors")

        if os.path.exists(model_file_safetensors):
            # Load from safetensors format
            from safetensors.torch import load_file
            state_dict = load_file(model_file_safetensors)
            model_file = model_file_safetensors
        elif os.path.exists(model_file_bin):
            # Load from pytorch bin format (weights_only=True for security)
            state_dict = torch.load(
                model_file_bin,
                map_location=self.device,
                weights_only=True
            )
            model_file = model_file_bin
        else:
            raise FileNotFoundError(
                f"Model file not found. Tried:\n"
                f"  - {model_file_bin}\n"
                f"  - {model_file_safetensors}"
            )

        # Determine input dimension and number of classes
        input_dim = len(self.scaler.mean_) if self.scaler else 768 + 17  # embedding + structural
        num_classes = len(self.label_encoder.classes_) if self.label_encoder else 4

        # Create model and load state dict
        self.model = StrategyScorerModel(input_dim, num_classes)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        LOGGER.info("Model loaded successfully from %s", model_file)
    
    def predict_strategy(
        self,
        text: str,
        profile: Dict[str, any],
        return_confidence: bool = True
    ) -> Tuple[str, float]:
        """
        Predict the best strategy for a document.
        
        Args:
            text: Document text
            profile: Document profile from analyze_document()
            return_confidence: Whether to return confidence score
            
        Returns:
            Tuple of (predicted_strategy, confidence)
        """
        if self.model is None:
            self.load_model()
        
        # Create feature vector
        features = create_feature_vector(
            text, profile, self.embedding_model, normalize=True, scaler=self.scaler
        )
        
        # Convert to tensor
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(features_tensor)
            logits = outputs["logits"]
            probabilities = torch.softmax(logits, dim=1)
            
            # Get predicted class
            predicted_class_idx = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class_idx].item()
            
            # Decode strategy name
            if self.label_encoder:
                predicted_strategy = self.label_encoder.inverse_transform([predicted_class_idx])[0]
            else:
                # Fallback strategy names
                strategy_names = ["recursive", "semantic", "parent_child", "late"]
                predicted_strategy = strategy_names[predicted_class_idx] if predicted_class_idx < len(strategy_names) else "recursive"
        
        LOGGER.debug(
            "Predicted strategy: %s (confidence=%.3f)",
            predicted_strategy, confidence
        )
        
        return predicted_strategy, confidence
    
    def predict_batch(
        self,
        texts: List[str],
        profiles: List[Dict[str, any]]
    ) -> List[Tuple[str, float]]:
        """
        Predict strategies for a batch of documents.
        
        Args:
            texts: List of document texts
            profiles: List of document profiles
            
        Returns:
            List of (strategy, confidence) tuples
        """
        if len(texts) != len(profiles):
            raise ValueError("Number of texts must match number of profiles")
        
        predictions = []
        for text, profile in zip(texts, profiles):
            strategy, confidence = self.predict_strategy(text, profile)
            predictions.append((strategy, confidence))
        
        return predictions
    
    def get_strategy_probabilities(
        self,
        text: str,
        profile: Dict[str, any]
    ) -> Dict[str, float]:
        """
        Get probability distribution over all strategies.
        
        Args:
            text: Document text
            profile: Document profile
            
        Returns:
            Dictionary mapping strategy names to probabilities
        """
        if self.model is None:
            self.load_model()
        
        # Create feature vector
        features = create_feature_vector(
            text, profile, self.embedding_model, normalize=True, scaler=self.scaler
        )
        
        # Convert to tensor
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(features_tensor)
            logits = outputs["logits"]
            probabilities = torch.softmax(logits, dim=1)
            
            # Convert to dictionary
            if self.label_encoder:
                strategy_names = self.label_encoder.classes_
            else:
                strategy_names = ["recursive", "semantic", "parent_child", "late"]
            
            prob_dict = {
                strategy_names[i]: probabilities[0][i].item()
                for i in range(len(strategy_names))
            }
        
        return prob_dict


# Global instance for efficiency
_SCORER_INSTANCE: Optional[StrategyScorerHF] = None


def get_strategy_scorer(
    model_path: str = "data/models/strategy_scorer_hf",
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> StrategyScorerHF:
    """
    Get or create the global strategy scorer instance.
    
    Args:
        model_path: Path to the trained model
        embedding_model: Name of the embedding model
        
    Returns:
        StrategyScorerHF instance
    """
    global _SCORER_INSTANCE
    
    if _SCORER_INSTANCE is None:
        _SCORER_INSTANCE = StrategyScorerHF(model_path, embedding_model)
        LOGGER.info("Created global strategy scorer instance")
    
    return _SCORER_INSTANCE


def predict_strategy(
    text: str,
    profile: Dict[str, any],
    model_path: str = "data/models/strategy_scorer_hf",
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> Tuple[str, float]:
    """
    Convenience function to predict strategy for a single document.
    
    Args:
        text: Document text
        profile: Document profile
        model_path: Path to the trained model
        embedding_model: Name of the embedding model
        
    Returns:
        Tuple of (predicted_strategy, confidence)
    """
    scorer = get_strategy_scorer(model_path, embedding_model)
    return scorer.predict_strategy(text, profile)


def clear_strategy_scorer() -> None:
    """Clear the global strategy scorer instance to free memory."""
    global _SCORER_INSTANCE
    _SCORER_INSTANCE = None
    LOGGER.info("Strategy scorer instance cleared from memory")
