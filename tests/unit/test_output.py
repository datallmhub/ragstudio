"""
Tests for CLI output format utilities.

Tests all output functions in src/core/cli/utils/output.py.
"""

import pytest
import json
import csv
from pathlib import Path

from src.core.cli.utils.output import (
    OutputFormat,
    save_chunks_json,
    save_chunks_jsonl,
    save_chunks_csv,
    save_chunks,
    detect_format_from_extension,
)


@pytest.fixture
def sample_chunks():
    """Sample chunks for testing."""
    return [
        {
            "id": "chunk001",
            "text": "First chunk text",
            "metadata": {"source": "doc1.txt", "page": 1}
        },
        {
            "id": "chunk002",
            "text": "Second chunk text",
            "metadata": {"source": "doc1.txt", "page": 2}
        },
        {
            "id": "chunk003",
            "text": "Third chunk with\nnewlines",
            "metadata": {"source": "doc2.txt", "page": 1, "confidence": 0.95}
        },
    ]


class TestOutputFormat:
    """Test OutputFormat enum."""

    def test_output_format_values(self):
        """Test that OutputFormat has correct values."""
        assert OutputFormat.json == "json"
        assert OutputFormat.jsonl == "jsonl"
        assert OutputFormat.csv == "csv"

    def test_output_format_is_string_enum(self):
        """Test that OutputFormat values are strings."""
        assert isinstance(OutputFormat.json.value, str)
        assert isinstance(OutputFormat.jsonl.value, str)
        assert isinstance(OutputFormat.csv.value, str)


class TestSaveChunksJson:
    """Test save_chunks_json function."""

    def test_save_chunks_json_basic(self, tmp_path, sample_chunks):
        """Test saving chunks to JSON format."""
        output_file = tmp_path / "output.json"

        save_chunks_json(sample_chunks, output_file)

        assert output_file.exists()
        loaded = json.loads(output_file.read_text())
        assert len(loaded) == 3
        assert loaded[0]["id"] == "chunk001"
        assert loaded[1]["text"] == "Second chunk text"

    def test_save_chunks_json_empty_list(self, tmp_path):
        """Test saving empty chunks list to JSON."""
        output_file = tmp_path / "empty.json"

        save_chunks_json([], output_file)

        assert output_file.exists()
        loaded = json.loads(output_file.read_text())
        assert loaded == []

    def test_save_chunks_json_unicode(self, tmp_path):
        """Test saving chunks with Unicode characters."""
        chunks = [
            {"id": "unicode1", "text": "Texte en français avec accents: é è ê"},
            {"id": "unicode2", "text": "日本語テキスト"},
            {"id": "unicode3", "text": "Emoji test 🎉🚀✨"},
        ]
        output_file = tmp_path / "unicode.json"

        save_chunks_json(chunks, output_file)

        assert output_file.exists()
        loaded = json.loads(output_file.read_text())
        assert loaded[0]["text"] == "Texte en français avec accents: é è ê"
        assert loaded[1]["text"] == "日本語テキスト"
        assert loaded[2]["text"] == "Emoji test 🎉🚀✨"


class TestSaveChunksJsonl:
    """Test save_chunks_jsonl function."""

    def test_save_chunks_jsonl_basic(self, tmp_path, sample_chunks):
        """Test saving chunks to JSONL format."""
        output_file = tmp_path / "output.jsonl"

        save_chunks_jsonl(sample_chunks, output_file)

        assert output_file.exists()
        lines = output_file.read_text().strip().split("\n")
        assert len(lines) == 3

        # Each line should be valid JSON
        chunk1 = json.loads(lines[0])
        chunk2 = json.loads(lines[1])
        assert chunk1["id"] == "chunk001"
        assert chunk2["id"] == "chunk002"

    def test_save_chunks_jsonl_empty_list(self, tmp_path):
        """Test saving empty chunks list to JSONL."""
        output_file = tmp_path / "empty.jsonl"

        save_chunks_jsonl([], output_file)

        assert output_file.exists()
        content = output_file.read_text()
        assert content == "\n"

    def test_save_chunks_jsonl_unicode(self, tmp_path):
        """Test JSONL with Unicode characters."""
        chunks = [
            {"id": "u1", "text": "Français"},
            {"id": "u2", "text": "日本語"},
        ]
        output_file = tmp_path / "unicode.jsonl"

        save_chunks_jsonl(chunks, output_file)

        lines = output_file.read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["text"] == "Français"
        assert json.loads(lines[1])["text"] == "日本語"


class TestSaveChunksCsv:
    """Test save_chunks_csv function."""

    def test_save_chunks_csv_basic(self, tmp_path, sample_chunks):
        """Test saving chunks to CSV format."""
        output_file = tmp_path / "output.csv"

        save_chunks_csv(sample_chunks, output_file)

        assert output_file.exists()

        with open(output_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 3
        assert rows[0]["id"] == "chunk001"
        assert rows[0]["text"] == "First chunk text"
        assert rows[0]["source"] == "doc1.txt"

    def test_save_chunks_csv_empty_list(self, tmp_path):
        """Test saving empty chunks list to CSV."""
        output_file = tmp_path / "empty.csv"

        save_chunks_csv([], output_file)

        assert output_file.exists()
        content = output_file.read_text()
        assert content == "id,text,metadata\n"

    def test_save_chunks_csv_newlines_removed(self, tmp_path, sample_chunks):
        """Test that newlines in text are replaced with spaces."""
        output_file = tmp_path / "newlines.csv"

        save_chunks_csv(sample_chunks, output_file)

        with open(output_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # chunk003 has "Third chunk with\nnewlines"
        assert rows[2]["text"] == "Third chunk with newlines"

    def test_save_chunks_csv_metadata_flattening(self, tmp_path):
        """Test that metadata is flattened into CSV columns."""
        chunks = [
            {
                "id": "c1",
                "text": "Text",
                "metadata": {
                    "str_field": "value",
                    "int_field": 42,
                    "float_field": 3.14,
                    "bool_field": True,
                }
            }
        ]
        output_file = tmp_path / "metadata.csv"

        save_chunks_csv(chunks, output_file)

        with open(output_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert rows[0]["str_field"] == "value"
        assert rows[0]["int_field"] == "42"
        assert rows[0]["float_field"] == "3.14"
        assert rows[0]["bool_field"] == "True"

    def test_save_chunks_csv_complex_metadata(self, tmp_path):
        """Test CSV with complex metadata (lists, dicts)."""
        chunks = [
            {
                "id": "c1",
                "text": "Text",
                "metadata": {
                    "simple": "value",
                    "list_field": [1, 2, 3],
                    "dict_field": {"nested": "data"},
                }
            }
        ]
        output_file = tmp_path / "complex.csv"

        save_chunks_csv(chunks, output_file)

        with open(output_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Complex types should be JSON-serialized
        assert rows[0]["simple"] == "value"
        assert json.loads(rows[0]["list_field"]) == [1, 2, 3]
        assert json.loads(rows[0]["dict_field"]) == {"nested": "data"}

    def test_save_chunks_csv_variable_metadata_keys(self, tmp_path):
        """Test CSV with chunks having different metadata keys."""
        chunks = [
            {"id": "c1", "text": "T1", "metadata": {"key1": "v1", "key2": "v2"}},
            {"id": "c2", "text": "T2", "metadata": {"key1": "v3", "key3": "v4"}},
            {"id": "c3", "text": "T3", "metadata": {"key2": "v5", "key3": "v6"}},
        ]
        output_file = tmp_path / "variable.csv"

        save_chunks_csv(chunks, output_file)

        with open(output_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            rows = list(reader)

        # All metadata keys should be in headers
        assert "key1" in headers
        assert "key2" in headers
        assert "key3" in headers

    def test_save_chunks_csv_no_metadata(self, tmp_path):
        """Test CSV with chunks without metadata."""
        chunks = [
            {"id": "c1", "text": "Text 1"},
            {"id": "c2", "text": "Text 2"},
        ]
        output_file = tmp_path / "no_metadata.csv"

        save_chunks_csv(chunks, output_file)

        with open(output_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 2
        assert rows[0]["id"] == "c1"
        assert rows[0]["text"] == "Text 1"


class TestSaveChunks:
    """Test save_chunks dispatcher function."""

    def test_save_chunks_json_format(self, tmp_path, sample_chunks):
        """Test save_chunks with JSON format."""
        output_file = tmp_path / "test.json"

        save_chunks(sample_chunks, output_file, format=OutputFormat.json)

        assert output_file.exists()
        loaded = json.loads(output_file.read_text())
        assert len(loaded) == 3

    def test_save_chunks_jsonl_format(self, tmp_path, sample_chunks):
        """Test save_chunks with JSONL format."""
        output_file = tmp_path / "test.jsonl"

        save_chunks(sample_chunks, output_file, format=OutputFormat.jsonl)

        assert output_file.exists()
        lines = output_file.read_text().strip().split("\n")
        assert len(lines) == 3

    def test_save_chunks_csv_format(self, tmp_path, sample_chunks):
        """Test save_chunks with CSV format."""
        output_file = tmp_path / "test.csv"

        save_chunks(sample_chunks, output_file, format=OutputFormat.csv)

        assert output_file.exists()
        with open(output_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 3

    def test_save_chunks_default_format(self, tmp_path, sample_chunks):
        """Test save_chunks with default format (JSON)."""
        output_file = tmp_path / "default.json"

        save_chunks(sample_chunks, output_file)  # No format specified

        assert output_file.exists()
        loaded = json.loads(output_file.read_text())
        assert len(loaded) == 3

    def test_save_chunks_invalid_format(self, tmp_path, sample_chunks):
        """Test save_chunks with invalid format."""
        output_file = tmp_path / "test.txt"

        with pytest.raises(ValueError, match="Unsupported format"):
            save_chunks(sample_chunks, output_file, format="invalid_format")


class TestDetectFormatFromExtension:
    """Test detect_format_from_extension function."""

    def test_detect_json_extension(self, tmp_path):
        """Test detecting JSON format from .json extension."""
        path = tmp_path / "file.json"

        format = detect_format_from_extension(path)

        assert format == OutputFormat.json

    def test_detect_jsonl_extension(self, tmp_path):
        """Test detecting JSONL format from .jsonl extension."""
        path = tmp_path / "file.jsonl"

        format = detect_format_from_extension(path)

        assert format == OutputFormat.jsonl

    def test_detect_csv_extension(self, tmp_path):
        """Test detecting CSV format from .csv extension."""
        path = tmp_path / "file.csv"

        format = detect_format_from_extension(path)

        assert format == OutputFormat.csv

    def test_detect_uppercase_extension(self, tmp_path):
        """Test detection with uppercase extensions."""
        assert detect_format_from_extension(tmp_path / "file.JSON") == OutputFormat.json
        assert detect_format_from_extension(tmp_path / "file.JSONL") == OutputFormat.jsonl
        assert detect_format_from_extension(tmp_path / "file.CSV") == OutputFormat.csv

    def test_detect_default_for_unknown_extension(self, tmp_path):
        """Test that unknown extensions default to JSON."""
        assert detect_format_from_extension(tmp_path / "file.txt") == OutputFormat.json
        assert detect_format_from_extension(tmp_path / "file.xyz") == OutputFormat.json
        assert detect_format_from_extension(tmp_path / "file") == OutputFormat.json

    def test_detect_no_extension(self, tmp_path):
        """Test detection with no file extension."""
        path = tmp_path / "file_without_extension"

        format = detect_format_from_extension(path)

        assert format == OutputFormat.json
