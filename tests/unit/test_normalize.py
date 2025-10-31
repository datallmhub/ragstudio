from pathlib import Path

import pytest

from src.workflows.ingest.normalize import clean_text, detect_language, normalize_document
from src.workflows.io.schema import Document


def test_clean_text_removes_extra_spaces():
    raw = "- Page 1 -   Bonjour   le   monde"
    assert clean_text(raw) == "Bonjour le monde"


def test_clean_text_handles_empty_string():
    assert clean_text("") == ""


@pytest.mark.parametrize(
    "text,expected_lang",
    [
        ("La voiture est électrique.", "fr"),
        ("The car is electric.", "en"),
    ],
)
def test_detect_language_basic(text: str, expected_lang: str):
    detected = detect_language(text)
    assert detected and detected.startswith(expected_lang)


def test_normalize_document_filters_languages(tmp_path: Path):
    doc = Document(source_path=tmp_path / "sample.txt", text="Hallo Welt!")
    assert normalize_document(doc, allowed_languages=["fr", "en"]) is None


def test_normalize_document_updates_metadata(sample_document: Document):
    normalized = normalize_document(sample_document, allowed_languages=None)
    assert normalized is not None
    assert normalized.metadata["language"] in ("fr", "fr-FR")
