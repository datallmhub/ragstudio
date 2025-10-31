"""
PDF text cleaning utilities to fix extraction artifacts.

Common artifacts from PDF extraction:
- Vertical text (letters separated by newlines): "e m m a G" -> "emmaG"
- Broken words: "Equipe-\nment" -> "Equipement"
- Multiple spaces/newlines
- Control characters
"""

import re
import logging
from typing import List

LOGGER = logging.getLogger(__name__)


def clean_pdf_text(text: str, aggressive: bool = False) -> str:
    """
    Clean PDF extraction artifacts.

    Args:
        text: Raw extracted text from PDF
        aggressive: If True, apply more aggressive cleaning (may remove valid content)

    Returns:
        Cleaned text
    """
    if not text:
        return text

    original_length = len(text)

    # 1. Fix vertical text (single letters separated by newlines)
    # Pattern: "e\nm\nm\na\nG" -> "emmaG"
    text = _fix_vertical_text(text)

    # 2. Fix hyphenated words broken across lines
    # Pattern: "Equipe-\nment" -> "Equipement"
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)

    # 3. Remove excessive newlines (keep max 2)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # 4. Fix multiple spaces
    text = re.sub(r'  +', ' ', text)

    # 5. Remove trailing/leading whitespace per line
    lines = text.split('\n')
    lines = [line.strip() for line in lines]
    text = '\n'.join(lines)

    if aggressive:
        # 6. Remove lines with only special characters (table artifacts)
        lines = [line for line in lines if _is_meaningful_line(line)]
        text = '\n'.join(lines)

        # 7. Remove isolated single characters (except meaningful ones)
        text = re.sub(r'\b[a-z]\b', '', text)

    cleaned_length = len(text)
    removed = original_length - cleaned_length

    if removed > 0:
        LOGGER.info(f"PDF cleaning removed {removed} characters ({removed/original_length*100:.1f}%)")

    return text


def _fix_vertical_text(text: str) -> str:
    """
    Fix vertical text artifacts where letters are separated by spaces or newlines.

    Patterns fixed:
    - "e m m a G" -> "emmaG"
    - "e\nm\nm\na\nG" -> "emmaG"
    - Mixed: "e m m\na G" -> "emmaG"
    """
    # Pattern 1: Letters separated by spaces on same line
    # "e m m a G" -> "emmaG"
    text = re.sub(r'\b([a-z])\s+([a-z])\s+([a-z])\s+([a-z])', r'\1\2\3\4', text, flags=re.IGNORECASE)
    text = re.sub(r'\b([a-z])\s+([a-z])\s+([a-z])', r'\1\2\3', text, flags=re.IGNORECASE)
    text = re.sub(r'\b([a-z])\s+([a-z])', r'\1\2', text, flags=re.IGNORECASE)

    # Pattern 2: Single letters on their own lines
    lines = text.split('\n')
    fixed_lines = []
    buffer = []

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Check if this looks like vertical text (single char or very short)
        if len(line) <= 2 and line and not line.isspace():
            # Start collecting potential vertical text
            buffer.append(line)

            # Look ahead to see if pattern continues
            j = i + 1
            while j < len(lines) and j < i + 15:  # Max 15 chars vertically
                next_line = lines[j].strip()
                if len(next_line) <= 2 and next_line and not next_line.isspace():
                    buffer.append(next_line)
                    j += 1
                else:
                    break

            # If we collected 3+ single chars, it's likely vertical text
            if len(buffer) >= 3:
                # Join them horizontally with no spaces
                merged = ''.join(buffer)
                fixed_lines.append(merged)
                i = j
                buffer = []
            else:
                # False alarm, keep as-is
                fixed_lines.extend(buffer)
                i += len(buffer)
                buffer = []
        else:
            fixed_lines.append(line)
            i += 1

    return '\n'.join(fixed_lines)


def _is_meaningful_line(line: str) -> bool:
    """
    Check if a line contains meaningful content.

    Lines with only special chars, numbers, or very short are filtered.
    """
    if not line or len(line) < 2:
        return False

    # Count alphanumeric vs special chars
    alpha_count = sum(c.isalnum() for c in line)

    # Line must have at least 30% alphanumeric
    return alpha_count / len(line) >= 0.3


def clean_table_artifacts(text: str) -> str:
    """
    Remove table extraction artifacts like repeated separators.

    Common patterns:
    - "| | | |" -> ""
    - "─────────" -> ""
    - "=========" -> ""
    """
    # Remove lines with only table separators
    lines = text.split('\n')

    cleaned = []
    for line in lines:
        # Skip lines with only separators
        if re.match(r'^[\s\|\-_=+]+$', line):
            continue
        cleaned.append(line)

    return '\n'.join(cleaned)


def preprocess_before_chunking(text: str, source_type: str = "pdf") -> str:
    """
    Main preprocessing pipeline before chunking.

    Args:
        text: Raw extracted text
        source_type: Document type ("pdf", "docx", "html", etc.)

    Returns:
        Cleaned text ready for chunking
    """
    if source_type == "pdf":
        # PDF-specific cleaning
        text = clean_pdf_text(text, aggressive=False)
        text = clean_table_artifacts(text)
    elif source_type in ("html", "docx"):
        # HTML/DOCX usually have cleaner extraction
        text = clean_pdf_text(text, aggressive=False)

    return text


__all__ = [
    "clean_pdf_text",
    "clean_table_artifacts",
    "preprocess_before_chunking",
]
