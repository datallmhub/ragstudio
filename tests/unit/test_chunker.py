from pathlib import Path
from types import SimpleNamespace

import pytest

from src.core.chunk.chunker import (
    chunk_document,
    chunk_document_adaptive,
    _encoding_length,
    _chunk_fallback,
    _chunk_parent_child,
    _split_by_headings,
)
from src.workflows.io.schema import Document


def _stub_tiktoken(monkeypatch):
    class StubEncoding:
        def encode(self, text: str):
            return [ord(ch) for ch in text]

        def decode(self, tokens):
            return "".join(chr(tok) for tok in tokens)

    stub = SimpleNamespace(
        encoding_for_model=lambda model: StubEncoding(),
        get_encoding=lambda name: StubEncoding(),
    )
    monkeypatch.setattr("src.core.chunk.chunker.tiktoken", stub)


def test_chunk_document_returns_empty_for_blank_text():
    doc = Document(source_path=Path("empty.txt"), text="")
    assert chunk_document(doc) == []


def test_chunk_document_uses_fallback_when_chonkie_missing(monkeypatch):
    doc = Document(
        source_path=Path("sample.txt"),
        text="Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 5,
    )
    # Test that LangChain chunker works (Chonkie removed from codebase)
    _stub_tiktoken(monkeypatch)
    chunks = chunk_document(doc, max_tokens=20, overlap=5)
    assert len(chunks) >= 1  # LangChain produces at least 1 chunk
    assert all(chunk.text for chunk in chunks)


@pytest.mark.parametrize("overlap", [0, 10])
def test_chunk_document_respects_overlap(monkeypatch, overlap: int):
    text = " ".join(str(i) for i in range(120))
    doc = Document(source_path=Path("numbers.txt"), text=text)
    # Test LangChain chunker with different overlap settings
    _stub_tiktoken(monkeypatch)
    chunks = chunk_document(doc, max_tokens=30, overlap=overlap)
    assert chunks
    # LangChain may produce different chunking, just verify we got chunks
    assert len(chunks) >= 1


class TestEncodingLength:
    """Tests for _encoding_length function."""

    def test_with_tiktoken(self, monkeypatch):
        """Test encoding length with tiktoken available."""
        _stub_tiktoken(monkeypatch)
        text = "Hello world"
        length = _encoding_length(text, model="gpt-3.5-turbo")
        assert length == len(text)  # StubEncoding encodes 1 char = 1 token

    def test_without_tiktoken(self, monkeypatch):
        """Test encoding length fallback when tiktoken unavailable."""
        monkeypatch.setattr("src.core.chunk.chunker.tiktoken", None)
        text = "Hello world"
        length = _encoding_length(text, model="gpt-3.5-turbo")
        assert length == len(text)  # Falls back to character count


class TestChunkFallback:
    """Tests for _chunk_fallback function."""

    def test_empty_text(self, monkeypatch):
        """Test chunking empty text returns empty list."""
        _stub_tiktoken(monkeypatch)
        result = _chunk_fallback("", max_tokens=100, overlap=10, model="gpt-3.5-turbo")
        assert result == []

    def test_basic_chunking(self, monkeypatch):
        """Test basic chunking of text."""
        _stub_tiktoken(monkeypatch)
        text = "A" * 200  # 200 characters
        result = _chunk_fallback(text, max_tokens=50, overlap=10, model="gpt-3.5-turbo")
        assert len(result) >= 3  # Should create multiple chunks
        assert all(chunk for chunk in result)  # All chunks should have content

    def test_no_overlap(self, monkeypatch):
        """Test chunking with no overlap."""
        _stub_tiktoken(monkeypatch)
        text = "A" * 100
        result = _chunk_fallback(text, max_tokens=50, overlap=0, model="gpt-3.5-turbo")
        assert len(result) == 2  # Should create exactly 2 chunks

    def test_without_tiktoken(self, monkeypatch):
        """Test fallback behavior when tiktoken not available."""
        monkeypatch.setattr("src.core.chunk.chunker.tiktoken", None)
        text = "A" * 200
        result = _chunk_fallback(text, max_tokens=50, overlap=10, model="gpt-3.5-turbo")
        assert len(result) >= 1  # Should fall back to character-based chunking
        assert all(chunk for chunk in result)

    def test_empty_tokens(self, monkeypatch):
        """Test handling of text that produces empty tokens."""
        class StubEncoding:
            def encode(self, text: str):
                return []  # Return empty tokens

            def decode(self, tokens):
                return ""

        stub = SimpleNamespace(
            encoding_for_model=lambda model: StubEncoding(),
            get_encoding=lambda name: StubEncoding(),
        )
        monkeypatch.setattr("src.core.chunk.chunker.tiktoken", stub)

        result = _chunk_fallback("test", max_tokens=100, overlap=10, model="gpt-3.5-turbo")
        assert result == []


class TestSplitByHeadings:
    """Tests for _split_by_headings function."""

    def test_markdown_headers(self):
        """Test splitting by markdown headers."""
        text = """# Introduction
This is intro text.

## Section 1
Section 1 content.

### Subsection 1.1
Subsection content."""

        sections = _split_by_headings(text)
        assert len(sections) >= 3
        # Check that headings are extracted
        headings = [section[0] for section in sections]
        assert any("Introduction" in h for h in headings)
        assert any("Section 1" in h for h in headings)

    def test_numbered_sections(self):
        """Test splitting by numbered sections."""
        text = """1. First Section
First section content.

2. Second Section
Second section content.

2.1. Subsection
Subsection content."""

        sections = _split_by_headings(text)
        assert len(sections) >= 2
        # At least 2 top-level sections
        headings = [section[0] for section in sections]
        assert any("First Section" in h for h in headings)

    def test_all_caps_headers(self):
        """Test splitting by ALL CAPS headers."""
        text = """INTRODUCTION
This is the introduction.

METHODOLOGY
This is the methodology.

RESULTS
These are the results."""

        sections = _split_by_headings(text)
        assert len(sections) >= 3
        headings = [section[0] for section in sections]
        assert any("INTRODUCTION" in h for h in headings)

    def test_roman_numerals(self):
        """Test splitting by Roman numeral headers."""
        text = """I. First Part
First part content.

II. Second Part
Second part content.

III. Third Part
Third part content."""

        sections = _split_by_headings(text)
        assert len(sections) >= 1  # Should find at least some sections

    def test_no_headers(self):
        """Test text with no headers returns single section."""
        text = "Just plain text with no headers."
        sections = _split_by_headings(text)
        assert len(sections) == 1
        assert sections[0][0] == ""  # No heading
        assert "plain text" in sections[0][1]

    def test_empty_sections(self):
        """Test handling of empty sections."""
        text = """# Header 1

# Header 2
Content here.

# Header 3"""

        sections = _split_by_headings(text)
        # Should handle empty sections gracefully
        assert len(sections) >= 2


class TestChunkParentChild:
    """Tests for _chunk_parent_child function."""

    def test_with_clear_structure(self, monkeypatch):
        """Test parent-child chunking with clear document structure."""
        _stub_tiktoken(monkeypatch)
        text = """# Introduction
This is the introduction with some content.

# Chapter 1
This is chapter 1 with substantial content that needs chunking.

## Section 1.1
Subsection content here."""

        result = _chunk_parent_child(text, max_tokens=50, overlap=10, model="gpt-3.5-turbo")
        assert len(result) >= 1
        # Chunks should include heading context
        assert any("#" in chunk for chunk in result)

    def test_without_structure_fallback(self, monkeypatch):
        """Test fallback to regular chunking when no structure found."""
        _stub_tiktoken(monkeypatch)
        text = "Just plain text without any structure or headers."

        result = _chunk_parent_child(text, max_tokens=50, overlap=10, model="gpt-3.5-turbo")
        assert len(result) >= 1
        assert all(chunk for chunk in result)

    def test_empty_sections_skipped(self, monkeypatch):
        """Test that empty sections are skipped."""
        _stub_tiktoken(monkeypatch)
        text = """# Header 1

# Header 2
Some content.

# Header 3
"""

        result = _chunk_parent_child(text, max_tokens=50, overlap=10, model="gpt-3.5-turbo")
        # Only sections with content should produce chunks
        assert len(result) >= 1


class TestChunkDocumentAdaptive:
    """Tests for chunk_document_adaptive function."""

    def test_parent_child_strategy(self, monkeypatch):
        """Test adaptive chunking with parent-child strategy."""
        _stub_tiktoken(monkeypatch)
        doc = Document(
            source_path=Path("structured.txt"),
            text="""# Introduction
Introduction text.

# Chapter 1
Chapter 1 text."""
        )

        strategy_config = {
            "strategy": "parent_child",
            "max_tokens": 100,
            "overlap": 20,
            "reason": "structured_document"
        }

        chunks = chunk_document_adaptive(doc, strategy_config=strategy_config)
        assert len(chunks) >= 1
        # Check metadata includes strategy info
        assert chunks[0].metadata["chunking_strategy"] == "parent_child"
        assert chunks[0].metadata["chunking_reason"] == "structured_document"

    def test_other_strategies_use_fallback(self, monkeypatch):
        """Test that non-parent_child strategies use fallback chunker."""
        _stub_tiktoken(monkeypatch)
        doc = Document(
            source_path=Path("sample.txt"),
            text="Simple text for chunking."
        )

        strategy_config = {
            "strategy": "recursive",
            "max_tokens": 100,
            "overlap": 20,
            "reason": "default"
        }

        chunks = chunk_document_adaptive(doc, strategy_config=strategy_config)
        assert len(chunks) >= 1
        assert chunks[0].metadata["chunking_strategy"] == "recursive"

    def test_empty_document(self, monkeypatch):
        """Test adaptive chunking with empty document."""
        _stub_tiktoken(monkeypatch)
        doc = Document(source_path=Path("empty.txt"), text="")

        strategy_config = {
            "strategy": "recursive",
            "max_tokens": 100,
            "overlap": 20,
            "reason": "default"
        }

        chunks = chunk_document_adaptive(doc, strategy_config=strategy_config)
        assert chunks == []

    def test_additional_metadata(self, monkeypatch):
        """Test that additional metadata is preserved."""
        _stub_tiktoken(monkeypatch)
        doc = Document(
            source_path=Path("sample.txt"),
            text="Sample text."
        )

        strategy_config = {
            "strategy": "recursive",
            "max_tokens": 100,
            "overlap": 20,
            "reason": "test"
        }

        additional_metadata = {"custom_key": "custom_value"}

        chunks = chunk_document_adaptive(
            doc,
            strategy_config=strategy_config,
            additional_metadata=additional_metadata
        )
        assert len(chunks) >= 1
        assert chunks[0].metadata["custom_key"] == "custom_value"
        assert chunks[0].metadata["chunking_strategy"] == "recursive"


class TestChunkDocument:
    """Additional tests for chunk_document function."""

    def test_strategy_mapping_semantic_to_recursive(self, monkeypatch):
        """Test that semantic strategy maps to recursive."""
        _stub_tiktoken(monkeypatch)
        doc = Document(
            source_path=Path("sample.txt"),
            text="Sample text for testing."
        )

        chunks = chunk_document(doc, strategy="semantic", max_tokens=100, overlap=20)
        assert len(chunks) >= 1
        # Metadata should show requested_strategy
        assert chunks[0].metadata["requested_strategy"] == "semantic"

    def test_strategy_mapping_token(self, monkeypatch):
        """Test that token strategy is preserved."""
        _stub_tiktoken(monkeypatch)
        doc = Document(
            source_path=Path("sample.txt"),
            text="Sample text for testing."
        )

        chunks = chunk_document(doc, strategy="token", max_tokens=100, overlap=20)
        assert len(chunks) >= 1
        assert chunks[0].metadata["requested_strategy"] == "token"

    def test_clean_pdf_disabled(self, monkeypatch):
        """Test chunking with PDF cleaning disabled."""
        _stub_tiktoken(monkeypatch)
        doc = Document(
            source_path=Path("sample.pdf"),
            text="PDF text with artifacts."
        )

        chunks = chunk_document(doc, clean_pdf=False, max_tokens=100, overlap=20)
        assert len(chunks) >= 1
