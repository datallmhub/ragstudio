"""
Unit tests for strategy_selector module.
"""

import pytest
from src.core.chunk.strategy_selector import (
    select_chunking_strategy,
    get_strategy_explanation,
    validate_strategy_config,
    get_strategy_stats
)


class TestSelectChunkingStrategy:
    """Tests for select_chunking_strategy function."""

    def test_short_document_strategy(self):
        """Test strategy selection for short documents (< 1000 tokens)."""
        profile = {
            "length_tokens": 500,
            "has_headings": False,
            "hierarchy_depth": 0,
            "type": "article",
            "has_tables": False,
            "structure_score": 0.3
        }

        config = select_chunking_strategy(profile)

        assert config["strategy"] == "recursive"
        assert config["max_tokens"] == 300
        assert config["overlap"] == 30
        assert config["reason"] == "short_document"

    def test_structured_document_with_headings(self):
        """Test parent_child strategy for structured documents with headings."""
        profile = {
            "length_tokens": 2000,
            "has_headings": True,
            "hierarchy_depth": 3,
            "type": "article",
            "has_tables": False,
            "structure_score": 0.5
        }

        config = select_chunking_strategy(profile)

        assert config["strategy"] == "parent_child"
        assert config["max_tokens"] == 400
        assert config["overlap"] == 50
        assert config["reason"] == "structured_document"

    def test_technical_document_type(self):
        """Test semantic strategy for technical document types."""
        profile = {
            "length_tokens": 2000,
            "has_headings": False,
            "hierarchy_depth": 0,
            "type": "fiche_technique",
            "has_tables": False,
            "structure_score": 0.3
        }

        config = select_chunking_strategy(profile)

        assert config["strategy"] == "semantic"
        assert config["max_tokens"] == 500
        assert config["overlap"] == 60
        assert config["reason"] == "technical_document"

    def test_rapport_document_type(self):
        """Test semantic strategy for rapport document type."""
        profile = {
            "length_tokens": 2000,
            "has_headings": False,
            "hierarchy_depth": 0,
            "type": "rapport",
            "has_tables": False,
            "structure_score": 0.3
        }

        config = select_chunking_strategy(profile)

        assert config["strategy"] == "semantic"
        assert config["reason"] == "technical_document"

    def test_tabular_content_strategy(self):
        """Test late chunking strategy for documents with tables."""
        profile = {
            "length_tokens": 2000,
            "has_headings": False,
            "hierarchy_depth": 0,
            "type": "article",
            "has_tables": True,
            "structure_score": 0.3
        }

        config = select_chunking_strategy(profile)

        assert config["strategy"] == "late"
        assert config["max_tokens"] == 300
        assert config["overlap"] == 50
        assert config["reason"] == "tabular_content"

    def test_high_structure_score(self):
        """Test parent_child strategy for high structure score."""
        profile = {
            "length_tokens": 2000,
            "has_headings": False,
            "hierarchy_depth": 0,
            "type": "article",
            "has_tables": False,
            "structure_score": 0.75
        }

        config = select_chunking_strategy(profile)

        assert config["strategy"] == "parent_child"
        assert config["reason"] == "high_structure_score"

    def test_long_document_strategy(self):
        """Test semantic strategy for long documents (> 5000 tokens)."""
        profile = {
            "length_tokens": 10000,
            "has_headings": False,
            "hierarchy_depth": 0,
            "type": "article",
            "has_tables": False,
            "structure_score": 0.3
        }

        config = select_chunking_strategy(profile)

        assert config["strategy"] == "semantic"
        assert config["max_tokens"] == 600
        assert config["overlap"] == 80
        assert config["reason"] == "long_document"

    def test_default_fallback(self):
        """Test default fallback strategy for medium documents with no special features."""
        profile = {
            "length_tokens": 3000,
            "has_headings": False,
            "hierarchy_depth": 0,
            "type": "article",
            "has_tables": False,
            "structure_score": 0.3
        }

        config = select_chunking_strategy(profile)

        assert config["strategy"] == "recursive"
        assert config["reason"] == "default_fallback"

    def test_priority_order_headings_before_tables(self):
        """Test that structured documents (headings) have priority over tables."""
        profile = {
            "length_tokens": 2000,
            "has_headings": True,
            "hierarchy_depth": 2,
            "type": "article",
            "has_tables": True,  # Both headings and tables
            "structure_score": 0.5
        }

        config = select_chunking_strategy(profile)

        # Headings (parent_child) should take priority over tables (late)
        assert config["strategy"] == "parent_child"
        assert config["reason"] == "structured_document"


class TestGetStrategyExplanation:
    """Tests for get_strategy_explanation function."""

    def test_short_document_explanation(self):
        """Test explanation for short document strategy."""
        profile = {
            "length_tokens": 500,
            "type": "article",
            "has_headings": False,
            "hierarchy_depth": 0,
            "has_tables": False,
            "structure_score": 0.3
        }
        config = {"strategy": "recursive", "reason": "short_document"}

        explanation = get_strategy_explanation(profile, config)

        assert "short" in explanation.lower()
        assert "500" in explanation
        assert "tokens" in explanation.lower()

    def test_structured_document_explanation(self):
        """Test explanation for structured document strategy."""
        profile = {
            "hierarchy_depth": 3,
            "type": "article",
            "length_tokens": 2000,
            "has_headings": True,
            "has_tables": False,
            "structure_score": 0.5
        }
        config = {"strategy": "parent_child", "reason": "structured_document"}

        explanation = get_strategy_explanation(profile, config)

        assert "headings" in explanation.lower() or "hierarchy" in explanation.lower()
        assert "3" in explanation

    def test_technical_document_explanation(self):
        """Test explanation for technical document strategy."""
        profile = {
            "type": "fiche_technique",
            "length_tokens": 2000,
            "has_headings": False,
            "hierarchy_depth": 0,
            "has_tables": False,
            "structure_score": 0.3
        }
        config = {"strategy": "semantic", "reason": "technical_document"}

        explanation = get_strategy_explanation(profile, config)

        assert "fiche_technique" in explanation.lower()
        assert "semantic" in explanation.lower()

    def test_high_structure_score_explanation(self):
        """Test explanation for high structure score."""
        profile = {
            "structure_score": 0.85,
            "type": "article",
            "length_tokens": 2000,
            "has_headings": False,
            "hierarchy_depth": 0,
            "has_tables": False
        }
        config = {"strategy": "parent_child", "reason": "high_structure_score"}

        explanation = get_strategy_explanation(profile, config)

        assert "0.85" in explanation or "structure" in explanation.lower()

    def test_unknown_reason_fallback(self):
        """Test explanation handles unknown reasons gracefully."""
        profile = {
            "type": "article",
            "length_tokens": 2000,
            "has_headings": False,
            "hierarchy_depth": 0,
            "has_tables": False,
            "structure_score": 0.3
        }
        config = {"strategy": "custom", "reason": "unknown_reason"}

        explanation = get_strategy_explanation(profile, config)

        assert "custom" in explanation.lower()
        assert "unknown_reason" in explanation.lower()


class TestValidateStrategyConfig:
    """Tests for validate_strategy_config function."""

    def test_valid_configuration(self):
        """Test validation of a valid strategy configuration."""
        config = {
            "strategy": "recursive",
            "max_tokens": 400,
            "overlap": 50,
            "reason": "test"
        }

        assert validate_strategy_config(config) is True

    def test_missing_required_keys(self):
        """Test validation fails when required keys are missing."""
        config = {
            "strategy": "recursive",
            "max_tokens": 400
            # Missing overlap and reason
        }

        assert validate_strategy_config(config) is False

    def test_invalid_strategy_name(self):
        """Test validation fails for invalid strategy names."""
        config = {
            "strategy": "invalid_strategy",
            "max_tokens": 400,
            "overlap": 50,
            "reason": "test"
        }

        assert validate_strategy_config(config) is False

    def test_valid_strategy_names(self):
        """Test all valid strategy names are accepted."""
        valid_strategies = ["recursive", "semantic", "parent_child", "late"]

        for strategy in valid_strategies:
            config = {
                "strategy": strategy,
                "max_tokens": 400,
                "overlap": 50,
                "reason": "test"
            }
            assert validate_strategy_config(config) is True

    def test_negative_max_tokens(self):
        """Test validation fails for negative max_tokens."""
        config = {
            "strategy": "recursive",
            "max_tokens": -100,
            "overlap": 50,
            "reason": "test"
        }

        assert validate_strategy_config(config) is False

    def test_zero_max_tokens(self):
        """Test validation fails for zero max_tokens."""
        config = {
            "strategy": "recursive",
            "max_tokens": 0,
            "overlap": 50,
            "reason": "test"
        }

        assert validate_strategy_config(config) is False

    def test_negative_overlap(self):
        """Test validation fails for negative overlap."""
        config = {
            "strategy": "recursive",
            "max_tokens": 400,
            "overlap": -10,
            "reason": "test"
        }

        assert validate_strategy_config(config) is False

    def test_zero_overlap_is_valid(self):
        """Test that zero overlap is valid."""
        config = {
            "strategy": "recursive",
            "max_tokens": 400,
            "overlap": 0,
            "reason": "test"
        }

        assert validate_strategy_config(config) is True

    def test_overlap_greater_than_max_tokens(self):
        """Test validation fails when overlap >= max_tokens."""
        config = {
            "strategy": "recursive",
            "max_tokens": 100,
            "overlap": 100,  # Equal to max_tokens
            "reason": "test"
        }

        assert validate_strategy_config(config) is False

    def test_overlap_exceeds_max_tokens(self):
        """Test validation fails when overlap > max_tokens."""
        config = {
            "strategy": "recursive",
            "max_tokens": 100,
            "overlap": 150,
            "reason": "test"
        }

        assert validate_strategy_config(config) is False

    def test_non_integer_max_tokens(self):
        """Test validation fails for non-integer max_tokens."""
        config = {
            "strategy": "recursive",
            "max_tokens": "400",  # String instead of int
            "overlap": 50,
            "reason": "test"
        }

        assert validate_strategy_config(config) is False

    def test_non_integer_overlap(self):
        """Test validation fails for non-integer overlap."""
        config = {
            "strategy": "recursive",
            "max_tokens": 400,
            "overlap": 50.5,  # Float instead of int
            "reason": "test"
        }

        assert validate_strategy_config(config) is False


class TestGetStrategyStats:
    """Tests for get_strategy_stats function."""

    def test_returns_dictionary(self):
        """Test that get_strategy_stats returns a dictionary."""
        stats = get_strategy_stats()

        assert isinstance(stats, dict)

    def test_contains_all_strategies(self):
        """Test that stats contain all strategy types."""
        stats = get_strategy_stats()

        assert "recursive" in stats
        assert "semantic" in stats
        assert "parent_child" in stats
        assert "late" in stats

    def test_all_values_are_integers(self):
        """Test that all stat values are integers."""
        stats = get_strategy_stats()

        for key, value in stats.items():
            assert isinstance(value, int)
