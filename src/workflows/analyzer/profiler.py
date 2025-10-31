"""
Document profiler for adaptive chunking strategy selection.

This module analyzes documents to extract key characteristics that help
determine the most appropriate chunking strategy.
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set

from src.workflows.io.schema import Document

LOGGER = logging.getLogger(__name__)


def analyze_document(document: Document) -> Dict[str, any]:
    """
    Analyze a document to extract characteristics for strategy selection.
    
    Args:
        document: The document to analyze
        
    Returns:
        Dictionary containing document profile metrics
    """
    profile = {
        "type": _detect_document_type(document),
        "lang": document.language or "unknown",
        "length_tokens": _estimate_token_count(document.text),
        "length_chars": len(document.text),
        "has_headings": _has_headings(document),
        "has_tables": _has_tables(document),
        "hierarchy_depth": _calculate_hierarchy_depth(document),
        "avg_sentence_length": _calculate_avg_sentence_length(document.text),
        "has_lists": _has_lists(document),
        "structure_score": 0.0,  # Will be calculated
    }
    
    # Calculate structure score based on multiple factors
    profile["structure_score"] = _calculate_structure_score(profile)
    
    LOGGER.info(
        "Document analyzed: %s (type=%s, tokens=%d, structure_score=%.2f)",
        Path(document.source_path).name,
        profile["type"],
        profile["length_tokens"],
        profile["structure_score"]
    )
    
    return profile


def _detect_document_type(document: Document) -> str:
    """Detect the type of document based on filename and content patterns."""
    # Handle both Path and string source_path
    if hasattr(document.source_path, 'name'):
        filename = document.source_path.name.lower()
    else:
        filename = Path(document.source_path).name.lower()
    
    text = document.text.lower()
    
    # File extension patterns
    if filename.endswith(('.pdf', '.docx', '.doc')):
        if any(keyword in filename for keyword in ['fiche', 'spec', 'manual', 'guide']):
            return "fiche_technique"
        elif any(keyword in filename for keyword in ['rapport', 'report', 'analyse']):
            return "rapport"
        elif any(keyword in filename for keyword in ['article', 'blog', 'news']):
            return "article"
    
    # Content patterns
    if any(keyword in text for keyword in ['table des matières', 'sommaire', 'chapitre']):
        return "rapport"
    elif any(keyword in text for keyword in ['caractéristiques', 'spécifications', 'paramètres']):
        return "fiche_technique"
    elif any(keyword in text for keyword in ['introduction', 'conclusion', 'résumé']):
        return "article"
    
    # Default based on length
    if len(document.text) > 5000:
        return "rapport"
    elif len(document.text) > 1000:
        return "article"
    else:
        return "document_court"


def _estimate_token_count(text: str) -> int:
    """Estimate token count using a simple heuristic."""
    # Rough estimation: 1 token ≈ 4 characters for most languages
    return len(text) // 4


def _has_headings(document: Document) -> bool:
    """Check if document has heading-like structures."""
    text = document.text
    heading_patterns = [
        # Common heading patterns
        r'^\s*#{1,6}\s+',  # Markdown headers
        r'^\s*\d+\.\s+',   # Numbered sections
        r'^\s*[A-Z][A-Z\s]+$',  # ALL CAPS lines
        r'^\s*[IVX]+\.\s+',  # Roman numerals
    ]
    
    import re
    for pattern in heading_patterns:
        if re.search(pattern, text, re.MULTILINE):
            return True
    
    return False


def _has_tables(document: Document) -> bool:
    """Check if document contains table-like structures."""
    text = document.text
    
    # Look for table indicators
    table_indicators = [
        '|',  # Pipe separators
        '\t',  # Tab separators
        'colonne',  # French table headers
        'ligne',   # French table rows
    ]
    
    # Count occurrences of table indicators
    table_score = sum(text.count(indicator) for indicator in table_indicators)
    
    # Consider it a table if we have multiple indicators
    return table_score > 5


def _calculate_hierarchy_depth(document: Document) -> int:
    """Calculate the depth of document hierarchy based on heading patterns."""
    text = document.text
    import re
    
    # Find different heading levels
    levels = set()
    
    # Markdown headers (# ## ###)
    markdown_headers = re.findall(r'^(#{1,6})\s+', text, re.MULTILINE)
    levels.update(len(header) for header in markdown_headers)
    
    # Numbered sections (1. 1.1. 1.1.1.)
    numbered_sections = re.findall(r'^(\d+(?:\.\d+)*)\.\s+', text, re.MULTILINE)
    levels.update(len(section.split('.')) for section in numbered_sections)
    
    return max(levels) if levels else 1


def _calculate_avg_sentence_length(text: str) -> float:
    """Calculate average sentence length in characters."""
    import re
    
    # Split into sentences (simple approach)
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return 0.0
    
    total_length = sum(len(sentence) for sentence in sentences)
    return total_length / len(sentences)


def _has_lists(document: Document) -> bool:
    """Check if document contains list-like structures."""
    text = document.text
    
    # Look for list indicators
    list_patterns = [
        r'^\s*[-*•]\s+',  # Bullet points
        r'^\s*\d+\.\s+',  # Numbered lists
        r'^\s*[a-z]\.\s+',  # Letter lists
    ]
    
    import re
    for pattern in list_patterns:
        if re.search(pattern, text, re.MULTILINE):
            return True
    
    return False


def _calculate_structure_score(profile: Dict[str, any]) -> float:
    """
    Calculate a structure score (0-1) indicating how structured the document is.
    Higher scores indicate more structured documents that benefit from parent_child chunking.
    """
    score = 0.0
    
    # Base score from hierarchy depth
    if profile["hierarchy_depth"] >= 3:
        score += 0.4
    elif profile["hierarchy_depth"] >= 2:
        score += 0.2
    
    # Bonus for having headings
    if profile["has_headings"]:
        score += 0.3
    
    # Bonus for having tables (indicates structured data)
    if profile["has_tables"]:
        score += 0.2
    
    # Bonus for having lists
    if profile["has_lists"]:
        score += 0.1
    
    # Penalty for very short documents
    if profile["length_tokens"] < 500:
        score -= 0.2
    
    # Bonus for longer documents (more likely to benefit from structure)
    if profile["length_tokens"] > 2000:
        score += 0.1
    
    return min(1.0, max(0.0, score))


def get_document_summary(profile: Dict[str, any]) -> str:
    """Generate a human-readable summary of the document profile."""
    return (
        f"Document: {profile['type']} | "
        f"Tokens: {profile['length_tokens']} | "
        f"Structure: {profile['structure_score']:.2f} | "
        f"Headings: {profile['has_headings']} | "
        f"Tables: {profile['has_tables']} | "
        f"Hierarchy: {profile['hierarchy_depth']}"
    )
