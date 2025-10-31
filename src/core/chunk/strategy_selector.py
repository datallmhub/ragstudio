"""
Strategy selector for adaptive chunking based on document analysis.

This module implements the decision logic for selecting the most appropriate
chunking strategy based on document characteristics.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from src.core.config import chunking

LOGGER = logging.getLogger(__name__)


def select_chunking_strategy(profile: Dict[str, any]) -> Dict[str, any]:
    """
    Select the most appropriate chunking strategy based on document profile.
    
    Args:
        profile: Document profile from analyze_document()
        
    Returns:
        Dictionary containing strategy configuration
    """
    strategy_config = {
        "strategy": "recursive",  # Default fallback
        "max_tokens": chunking.max_tokens,
        "overlap": chunking.overlap,
        "reason": "default_fallback"
    }
    
    # Decision tree based on document characteristics
    if profile["length_tokens"] < 1000:
        strategy_config = {
            "strategy": "recursive",
            "max_tokens": 300,
            "overlap": 30,
            "reason": "short_document"
        }
    
    elif profile["has_headings"] and profile["hierarchy_depth"] >= 2:
        strategy_config = {
            "strategy": "parent_child",
            "max_tokens": 400,
            "overlap": 50,
            "reason": "structured_document"
        }
    
    elif profile["type"] in ["fiche_technique", "rapport"]:
        strategy_config = {
            "strategy": "semantic",
            "max_tokens": 500,
            "overlap": 60,
            "reason": "technical_document"
        }
    
    elif profile["has_tables"]:
        strategy_config = {
            "strategy": "late",
            "max_tokens": 300,
            "overlap": 50,
            "reason": "tabular_content"
        }
    
    elif profile["structure_score"] > 0.6:
        strategy_config = {
            "strategy": "parent_child",
            "max_tokens": 400,
            "overlap": 50,
            "reason": "high_structure_score"
        }
    
    elif profile["length_tokens"] > 5000:
        strategy_config = {
            "strategy": "semantic",
            "max_tokens": 600,
            "overlap": 80,
            "reason": "long_document"
        }
    
    LOGGER.info(
        "📘 Strategy selected: %s (reason=%s, tokens=%d, type=%s)",
        strategy_config["strategy"],
        strategy_config["reason"],
        profile["length_tokens"],
        profile["type"]
    )
    
    return strategy_config


def get_strategy_explanation(profile: Dict[str, any], strategy_config: Dict[str, any]) -> str:
    """
    Generate a human-readable explanation of why this strategy was chosen.
    
    Args:
        profile: Document profile
        strategy_config: Selected strategy configuration
        
    Returns:
        Explanation string
    """
    strategy = strategy_config["strategy"]
    reason = strategy_config["reason"]
    
    explanations = {
        "short_document": f"Document is short ({profile['length_tokens']} tokens), using recursive chunking for simplicity",
        "structured_document": f"Document has headings and hierarchy (depth={profile['hierarchy_depth']}), using parent-child chunking",
        "technical_document": f"Document type '{profile['type']}' benefits from semantic chunking for better context",
        "tabular_content": f"Document contains tables, using late chunking to preserve table structure",
        "high_structure_score": f"High structure score ({profile['structure_score']:.2f}), using parent-child chunking",
        "long_document": f"Long document ({profile['length_tokens']} tokens), using semantic chunking for better context",
        "default_fallback": "Using default recursive strategy as fallback"
    }
    
    return explanations.get(reason, f"Strategy {strategy} selected for reason: {reason}")


def validate_strategy_config(strategy_config: Dict[str, any]) -> bool:
    """
    Validate that the strategy configuration is valid.
    
    Args:
        strategy_config: Strategy configuration to validate
        
    Returns:
        True if valid, False otherwise
    """
    required_keys = ["strategy", "max_tokens", "overlap", "reason"]
    
    # Check required keys
    if not all(key in strategy_config for key in required_keys):
        LOGGER.error("Strategy config missing required keys: %s", required_keys)
        return False
    
    # Validate strategy
    valid_strategies = ["recursive", "semantic", "parent_child", "late"]
    if strategy_config["strategy"] not in valid_strategies:
        LOGGER.error("Invalid strategy: %s. Must be one of: %s", 
                    strategy_config["strategy"], valid_strategies)
        return False
    
    # Validate numeric values
    if not isinstance(strategy_config["max_tokens"], int) or strategy_config["max_tokens"] <= 0:
        LOGGER.error("Invalid max_tokens: %s", strategy_config["max_tokens"])
        return False
    
    if not isinstance(strategy_config["overlap"], int) or strategy_config["overlap"] < 0:
        LOGGER.error("Invalid overlap: %s", strategy_config["overlap"])
        return False
    
    # Check overlap is not greater than max_tokens
    if strategy_config["overlap"] >= strategy_config["max_tokens"]:
        LOGGER.error("Overlap (%d) must be less than max_tokens (%d)", 
                    strategy_config["overlap"], strategy_config["max_tokens"])
        return False
    
    return True


def get_strategy_stats() -> Dict[str, int]:
    """
    Get statistics about strategy usage (for monitoring).
    
    Returns:
        Dictionary with strategy usage counts
    """
    # This would typically read from a log file or database
    # For now, return empty stats
    return {
        "recursive": 0,
        "semantic": 0,
        "parent_child": 0,
        "late": 0
    }
