"""
Unit tests for LangChain-based Atlas Chunker.

Tests:
- Text preprocessing
- Chunk creation
- Metadata completeness
- Quality validation (no word breaks, no duplicates)
"""

import pytest
from pathlib import Path

from src.core.chunk.langchain_chunker import (
    AtlasChunker,
    TextPreprocessor,
    chunk_document_langchain,
)
from src.workflows.io.schema import Document


class TestTextPreprocessor:
    """Test text preprocessing functions."""

    def test_fix_extraction_errors(self):
        """Test fixing common PDF extraction errors."""
        text = "Cela apermis de distinguer Ala fin eouvert"
        cleaned, fixes = TextPreprocessor.fix_extraction_errors(text)

        assert "a permis" in cleaned
        assert "A la" in cleaned
        assert "e ouvert" in cleaned
        assert "apermis" not in cleaned
        assert fixes >= 3

    def test_remove_page_numbers(self):
        """Test removing isolated page numbers."""
        text = "End of page\n42\nNew Chapter"
        cleaned, removed = TextPreprocessor.remove_page_numbers(text)

        assert "\n42\n" not in cleaned
        assert "New Chapter" in cleaned
        assert removed == 1

    def test_normalize_whitespace(self):
        """Test whitespace normalization."""
        text = "Text  with    multiple   spaces\n\n\n\nand newlines"
        cleaned = TextPreprocessor.normalize_whitespace(text)

        assert "  " not in cleaned
        assert "\n\n\n" not in cleaned

    def test_preprocess_full_pipeline(self):
        """Test full preprocessing pipeline."""
        text = "Cela apermis\n5\nDe continuer Ala page"
        cleaned, stats = TextPreprocessor.preprocess(text)

        assert "a permis" in cleaned
        assert "A la" in cleaned
        assert "\n5\n" not in cleaned
        assert stats['extraction_fixes'] >= 2
        assert stats['page_numbers_removed'] == 1


class TestAtlasChunker:
    """Test AtlasChunker class."""

    @pytest.fixture
    def sample_document(self):
        """Create a sample document."""
        text = """COURS DE GRAMMAIRE FRANÇAISE

L'orthographe française a une longue histoire. Les scribes du Moyen Âge
ont contribué à sa formation.

Renaissance: les imprimeurs ont standardisé l'orthographe. Les grammairiens
ont codifié les règles.

Époque moderne: l'Académie française fixe les normes. L'enseignement
obligatoire diffuse l'orthographe standard."""

        return Document(
            text=text,
            source_path=Path("test.pdf"),
            metadata={"format": "pdf"}
        )

    def test_chunker_initialization(self):
        """Test chunker initialization."""
        chunker = AtlasChunker(
            strategy="recursive",
            chunk_size=200,
            chunk_overlap=20,
        )

        assert chunker.strategy == "recursive"
        assert chunker.chunk_size == 200
        assert chunker.chunk_overlap == 20

    def test_chunk_document_basic(self, sample_document):
        """Test basic chunking."""
        chunker = AtlasChunker(chunk_size=50, chunk_overlap=10)
        chunks = chunker.chunk_document(sample_document)

        # Should create multiple chunks
        assert len(chunks) > 1

        # All chunks should have text
        assert all(len(c.text) > 0 for c in chunks)

        # All chunks should have metadata
        assert all(c.metadata is not None for c in chunks)

    def test_chunk_metadata_complete(self, sample_document):
        """Test that chunks have complete metadata."""
        chunker = AtlasChunker(chunk_size=150)
        chunks = chunker.chunk_document(sample_document)

        for chunk in chunks:
            metadata = chunk.metadata

            # Required fields
            assert "chunk_index" in metadata
            assert "total_chunks" in metadata
            assert "char_start" in metadata
            assert "char_end" in metadata
            assert "token_count" in metadata
            assert "chunking_strategy" in metadata

            # Types
            assert isinstance(metadata["chunk_index"], int)
            assert isinstance(metadata["char_start"], int)
            assert isinstance(metadata["char_end"], int)

            # Values
            assert metadata["chunk_index"] >= 0
            assert metadata["char_start"] >= 0
            assert metadata["char_end"] > metadata["char_start"]

    def test_chunk_no_word_breaks(self, sample_document):
        """Test that chunks don't break words."""
        chunker = AtlasChunker(chunk_size=100)
        chunks = chunker.chunk_document(sample_document)

        for chunk in chunks:
            # Chunk should not end with isolated letter (mid-word break)
            # Allow ending with punctuation or space
            last_char = chunk.text[-1]

            # If ends with letter, should be followed by space or punctuation in next chunk
            if last_char.isalpha() and chunk.metadata["chunk_index"] < len(chunks) - 1:
                # Check if it's a proper word boundary
                # This is acceptable if next chunk starts with space/punctuation
                next_chunk = chunks[chunk.metadata["chunk_index"] + 1]
                first_char = next_chunk.text[0] if next_chunk.text else ""

                # Should start with space, newline, or be continuation
                assert first_char in [" ", "\n"] or not first_char.isalpha(), \
                    f"Chunk {chunk.metadata['chunk_index']} breaks word: '{chunk.text[-20:]}'"

    def test_chunk_no_duplicates(self, sample_document):
        """Test that there are no exact duplicate chunks."""
        chunker = AtlasChunker(chunk_size=150)
        chunks = chunker.chunk_document(sample_document)

        chunk_texts = [c.text for c in chunks]
        unique_texts = set(chunk_texts)

        assert len(chunk_texts) == len(unique_texts), "Found duplicate chunks"

    def test_chunk_overlap_works(self, sample_document):
        """Test that overlap creates continuity between chunks."""
        chunker = AtlasChunker(chunk_size=150, chunk_overlap=50)
        chunks = chunker.chunk_document(sample_document)

        if len(chunks) > 1:
            # Check overlap between consecutive chunks
            for i in range(len(chunks) - 1):
                chunk1 = chunks[i]
                chunk2 = chunks[i + 1]

                # There should be some text overlap
                # Check if end of chunk1 appears in start of chunk2
                end_text = chunk1.text[-40:].strip()
                start_text = chunk2.text[:80].strip()

                # Overlap should exist
                has_overlap = any(
                    word in start_text
                    for word in end_text.split()[-3:]  # Last 3 words
                    if len(word) > 3
                )

                assert has_overlap, f"No overlap between chunks {i} and {i+1}"

    def test_preprocessing_enabled(self):
        """Test that preprocessing fixes errors."""
        doc = Document(
            text="Cela apermis de faire Ala page",
            source_path=Path("test.pdf")
        )

        chunker = AtlasChunker(chunk_size=100)
        chunks = chunker.chunk_document(doc, preprocess=True)

        # Check that errors are fixed
        all_text = " ".join([c.text for c in chunks])
        assert "a permis" in all_text
        assert "A la" in all_text
        assert "apermis" not in all_text

    def test_preprocessing_disabled(self):
        """Test that preprocessing can be disabled."""
        doc = Document(
            text="Cela apermis de faire",
            source_path=Path("test.pdf")
        )

        chunker = AtlasChunker(chunk_size=100)
        chunks = chunker.chunk_document(doc, preprocess=False)

        # Errors should remain
        all_text = " ".join([c.text for c in chunks])
        assert "apermis" in all_text

    def test_additional_metadata(self, sample_document):
        """Test adding additional metadata to chunks."""
        chunker = AtlasChunker(chunk_size=150)
        chunks = chunker.chunk_document(
            sample_document,
            additional_metadata={"custom_field": "test_value"}
        )

        for chunk in chunks:
            assert "custom_field" in chunk.metadata
            assert chunk.metadata["custom_field"] == "test_value"


class TestConvenienceFunction:
    """Test the convenience function."""

    def test_chunk_document_langchain(self):
        """Test the convenience function."""
        doc = Document(
            text="This is a test document. It has multiple sentences. "
                 "We will chunk it using the convenience function.",
            source_path=Path("test.txt")
        )

        chunks = chunk_document_langchain(
            doc,
            strategy="recursive",
            max_tokens=50,
            overlap=10
        )

        assert len(chunks) > 0
        assert all(c.metadata["chunking_strategy"] == "recursive" for c in chunks)


class TestEdgeCases:
    """Test edge cases."""

    def test_empty_document(self):
        """Test chunking empty document."""
        doc = Document(text="", source_path=Path("empty.pdf"))
        chunker = AtlasChunker()
        chunks = chunker.chunk_document(doc)

        assert len(chunks) == 0

    def test_very_short_document(self):
        """Test chunking very short document."""
        doc = Document(text="Hello", source_path=Path("short.txt"))
        chunker = AtlasChunker(chunk_size=100)
        chunks = chunker.chunk_document(doc)

        assert len(chunks) == 1
        assert chunks[0].text == "Hello"

    def test_very_long_document(self):
        """Test chunking very long document."""
        # Create 10,000 char document
        text = "This is a sentence. " * 500  # ~10,000 chars

        doc = Document(text=text, source_path=Path("long.txt"))
        chunker = AtlasChunker(chunk_size=200)
        chunks = chunker.chunk_document(doc)

        # Should create many chunks
        assert len(chunks) > 5

        # All chunks should be reasonably sized
        for chunk in chunks:
            assert len(chunk.text) < 1000  # Max ~200 tokens * 4 chars

    def test_special_characters(self):
        """Test handling special characters."""
        text = "Texte avec accents: é è ê à ù. Caractères spéciaux: « » © ®."
        doc = Document(text=text, source_path=Path("special.txt"))

        chunker = AtlasChunker(chunk_size=100)
        chunks = chunker.chunk_document(doc)

        assert len(chunks) > 0
        # Special chars should be preserved
        all_text = " ".join([c.text for c in chunks])
        assert "é" in all_text
        assert "«" in all_text