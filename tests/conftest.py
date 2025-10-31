from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterator

import pytest

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.workflows.io.schema import Document


@pytest.fixture
def sample_text() -> str:
    return (
        "La Peugeot 208 2024 propose un moteur hybride 48V, une version électrique "
        "et de nouveaux systèmes d'aide à la conduite."
    )


@pytest.fixture
def sample_document(tmp_path: Path, sample_text: str) -> Document:
    source = tmp_path / "sample.txt"
    source.write_text(sample_text, encoding="utf8")
    return Document(source_path=source, text=sample_text, metadata={"source_name": source.name})


@pytest.fixture
def data_dir() -> Path:
    return Path("tests/data").resolve()
