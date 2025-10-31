"""
LangChain-based chunker for Atlas-RAG v3.0

Replaces Chonkie with proper LangChain text splitters.
Provides clean chunking without word breaks, with rich metadata.
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Dict, List, Optional

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)

from src.workflows.io.schema import Chunk, Document

LOGGER = logging.getLogger(__name__)


class TextPreprocessor:
    """Pre-process text before chunking to fix common PDF extraction errors."""

    @staticmethod
    def fix_extraction_errors(text: str) -> tuple[str, int]:
        """
        Fix common PDF extraction errors (missing spaces, etc.).

        Returns:
            tuple: (cleaned_text, num_fixes)
        """
        original = text
        fixes = 0

        # Fix missing spaces
        space_fixes = {
            r'\bapermis\b': 'a permis',
            r'\bAla\b': 'A la',
            r'\basouvent\b': 'a souvent',
            r'\bapresque\b': 'a presque',
            r"ad'ailleurs": "a d'ailleurs",
            r'\bDela\b': 'De la',
            r'\beouvert\b': 'e ouvert',
            r'»va\b': '» va',
            r'»de\b': '» de',
            r'\bouvent(?=[A-ZÀÉÈ])': 'ouvent ',  # "ouventL'" → "ouvent L'"
        }

        for pattern, replacement in space_fixes.items():
            matches = re.findall(pattern, text)
            if matches:
                fixes += len(matches)
            text = re.sub(pattern, replacement, text)

        return text, fixes

    @staticmethod
    def remove_page_numbers(text: str) -> tuple[str, int]:
        """
        Remove isolated page numbers that appear between paragraphs.

        Returns:
            tuple: (cleaned_text, num_removed)
        """
        # Pattern: \n<1-3 digit number>\n<Text starting with capital>
        pattern = r'\n(\d{1,3})\n(?=[A-ZÀÉÈÊ])'
        matches = len(re.findall(pattern, text))
        text = re.sub(pattern, '\n', text)
        return text, matches

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """Normalize excessive whitespace."""
        # Multiple spaces → single space
        text = re.sub(r' {2,}', ' ', text)
        # Multiple newlines → max 2
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    @classmethod
    def preprocess(cls, text: str) -> tuple[str, Dict[str, int]]:
        """
        Full preprocessing pipeline.

        Returns:
            tuple: (cleaned_text, stats_dict)
        """
        stats = {}

        # Fix extraction errors
        text, fixes = cls.fix_extraction_errors(text)
        stats['extraction_fixes'] = fixes

        # Remove page numbers
        text, removed = cls.remove_page_numbers(text)
        stats['page_numbers_removed'] = removed

        # Normalize whitespace
        text = cls.normalize_whitespace(text)

        return text, stats


class AtlasChunker:
    """
    Clean chunker using LangChain text splitters.

    Provides:
    - No word breaks
    - Rich metadata (chunk_index, char_start/end, token_count, etc.)
    - Quality validation
    - Multiple strategies
    """

    def __init__(
        self,
        strategy: str = "recursive",
        chunk_size: int = 400,
        chunk_overlap: int = 50,
        length_function: str = "tokens",
        **kwargs
    ):
        """
        Initialize AtlasChunker.

        Args:
            strategy: Chunking strategy ("recursive", "token")
            chunk_size: Maximum size per chunk (in tokens or chars)
            chunk_overlap: Overlap between chunks
            length_function: How to measure size ("tokens" or "chars")
        """
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function

        # Create appropriate splitter
        if strategy == "recursive":
            # Best strategy: respects document structure
            # Approximation: 1 token ≈ 4 chars for French
            char_chunk_size = chunk_size * 4 if length_function == "tokens" else chunk_size
            char_overlap = chunk_overlap * 4 if length_function == "tokens" else chunk_overlap

            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=char_chunk_size,
                chunk_overlap=char_overlap,
                length_function=len,
                separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
                keep_separator=True,
            )

        elif strategy == "token":
            # Token-based (requires tiktoken)
            self.splitter = TokenTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        LOGGER.info(
            f"AtlasChunker initialized: strategy={strategy}, "
            f"chunk_size={chunk_size}, overlap={chunk_overlap}"
        )

    def chunk_document(
        self,
        document: Document,
        preprocess: bool = True,
        additional_metadata: Optional[Dict] = None,
    ) -> List[Chunk]:
        """
        Chunk a document with rich metadata.

        Args:
            document: Document to chunk
            preprocess: Whether to preprocess text (fix PDF errors)
            additional_metadata: Additional metadata to add to chunks

        Returns:
            List of Chunk objects with complete metadata
        """
        if not document.text:
            return []

        # 1. Preprocessing
        text_to_chunk = document.text
        preprocess_stats = {}

        if preprocess:
            text_to_chunk, preprocess_stats = TextPreprocessor.preprocess(document.text)

            if preprocess_stats.get('extraction_fixes', 0) > 0:
                LOGGER.info(
                    f"Preprocessing: fixed {preprocess_stats['extraction_fixes']} extraction errors, "
                    f"removed {preprocess_stats['page_numbers_removed']} page numbers"
                )

        # 2. Split text
        try:
            chunk_texts = self.splitter.split_text(text_to_chunk)
        except Exception as e:
            LOGGER.error(f"Chunking failed: {e}")
            raise

        if not chunk_texts:
            LOGGER.warning("No chunks created")
            return []

        LOGGER.info(f"Created {len(chunk_texts)} chunks")

        # 3. Create Chunk objects with metadata
        chunks = []
        char_position = 0

        for idx, chunk_text in enumerate(chunk_texts):
            # Find exact position in original text
            # Use first 50 chars to locate chunk
            search_text = chunk_text[:min(50, len(chunk_text))]
            char_start = text_to_chunk.find(search_text, char_position)

            if char_start == -1:
                # Fallback: use last position
                char_start = char_position

            char_end = char_start + len(chunk_text)

            # Generate deterministic ID (not for security)
            chunk_id = hashlib.md5(
                f"{document.source_path}:{idx}:{chunk_text[:100]}".encode(),
                usedforsecurity=False
            ).hexdigest()[:12]

            # Count sentences (approximate)
            sentence_count = len(re.findall(r'[.!?]+', chunk_text))

            # Estimate tokens (1 token ≈ 4 chars)
            token_count = len(chunk_text) // 4

            # Build metadata
            chunk_metadata = {
                "source": str(document.source_path) if document.source_path else "unknown",
                "chunk_index": idx,
                "total_chunks": len(chunk_texts),
                "char_start": char_start,
                "char_end": char_end,
                "char_length": len(chunk_text),
                "token_count": token_count,
                "sentence_count": sentence_count,
                "chunking_strategy": self.strategy,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                **preprocess_stats,  # Add preprocessing stats
            }

            # Add document metadata
            if document.metadata:
                chunk_metadata.update(document.metadata)

            # Add additional metadata
            if additional_metadata:
                chunk_metadata.update(additional_metadata)

            # Create Chunk
            chunk = Chunk(
                document_id=document.id,
                text=chunk_text,
                metadata=chunk_metadata,
                id=f"chunk_{chunk_id}",
            )

            chunks.append(chunk)
            char_position = char_end

        # 4. Validate quality
        validation_issues = self._validate_chunks(chunks)
        if validation_issues:
            LOGGER.warning(f"Quality issues detected: {len(validation_issues)}")
            for issue in validation_issues[:3]:  # Log first 3
                LOGGER.warning(f"  - {issue}")

        return chunks

    def _validate_chunks(self, chunks: List[Chunk]) -> List[str]:
        """
        Validate chunk quality.

        Returns:
            List of issue descriptions (empty if no issues)
        """
        issues = []

        # Check for very short chunks (< 50 chars)
        short_chunks = [c for c in chunks if len(c.text) < 50]
        if short_chunks:
            issues.append(f"{len(short_chunks)} chunks < 50 chars")

        # Check for word breaks at end
        for chunk in chunks[:10]:  # Check first 10
            last_char = chunk.text[-1] if chunk.text else ""
            # Should end with punctuation or newline ideally
            if last_char.isalpha():
                issues.append(
                    f"Chunk {chunk.metadata.get('chunk_index', '?')} ends mid-word: "
                    f"'{chunk.text[-20:]}'"
                )

        # Check for exact duplicates
        seen_texts = set()
        for chunk in chunks:
            text_hash = hashlib.md5(chunk.text.encode(), usedforsecurity=False).hexdigest()
            if text_hash in seen_texts:
                issues.append(f"Duplicate chunk detected: {chunk.id}")
            seen_texts.add(text_hash)

        return issues


def chunk_document_langchain(
    document: Document,
    strategy: str = "recursive",
    max_tokens: int = 400,
    overlap: int = 50,
    additional_metadata: Optional[Dict] = None,
    preprocess: bool = True,
) -> List[Chunk]:
    """
    Convenience function to chunk a document with LangChain.

    Args:
        document: Document to chunk
        strategy: Chunking strategy
        max_tokens: Maximum tokens per chunk
        overlap: Token overlap between chunks
        additional_metadata: Additional metadata
        preprocess: Whether to preprocess text

    Returns:
        List of Chunk objects
    """
    chunker = AtlasChunker(
        strategy=strategy,
        chunk_size=max_tokens,
        chunk_overlap=overlap,
        length_function="tokens",
    )

    return chunker.chunk_document(
        document,
        preprocess=preprocess,
        additional_metadata=additional_metadata,
    )