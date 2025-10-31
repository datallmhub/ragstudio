"""
Text quality checking utilities.

Used to detect if extracted text is readable or if OCR/extraction failed completely.
"""

import re
from typing import Dict


def _has_mixed_case_chaos(word: str) -> bool:
    """Check if a word has chaotic mixed case (sign of bad OCR)."""
    if len(word) < 4:
        return False

    # Count transitions between upper and lower case
    transitions = 0
    for i in range(len(word) - 1):
        if word[i].isalpha() and word[i+1].isalpha():
            if word[i].isupper() != word[i+1].isupper():
                transitions += 1

    # If more than 2 transitions in a short word, it's probably garbage
    return transitions > 2


def _is_valid_word(word: str) -> bool:
    """
    Check if a word looks like a valid word (not random chars).

    Bad OCR produces things like:
    - sjuaweoejdep (no vowel pattern)
    - aJANe0 (mixed case chaos + numbers)
    - UONeIOeNEJ (too many uppercase transitions)
    - JUSWEWLIOJUOD (too long, all uppercase)
    """
    if len(word) < 2:
        return True  # Short words OK

    # Filter out words with numbers mixed with letters (usually OCR artifacts)
    has_digit = any(c.isdigit() for c in word)
    has_letter = any(c.isalpha() for c in word)
    if has_digit and has_letter:
        return False

    # Filter out words with mixed case chaos
    if _has_mixed_case_chaos(word):
        return False

    # Filter out very long words (likely OCR errors)
    if len(word) > 15:
        return False

    # Filter out all-uppercase words longer than 5 chars (except known acronyms)
    if word.isupper() and len(word) > 5:
        return False

    # Count vowels and consonants
    vowels = 'aeiouAEIOUéèêëàâäïîôöùûüÿæœ'
    consonants = 'bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ'

    vowel_count = sum(1 for c in word if c in vowels)
    consonant_count = sum(1 for c in word if c in consonants)
    total_letters = vowel_count + consonant_count

    if total_letters == 0:
        return False

    # Words must have at least one vowel
    if vowel_count == 0:
        return False

    # Too many consonants in a row is suspicious
    max_consonant_run = 0
    current_run = 0
    for c in word:
        if c in consonants:
            current_run += 1
            max_consonant_run = max(max_consonant_run, current_run)
        else:
            current_run = 0

    if max_consonant_run > 5:  # More than 5 consonants in a row
        return False

    return True


def check_text_quality(text: str, min_valid_word_ratio: float = 0.4) -> Dict[str, any]:
    """
    Check if extracted text is of sufficient quality to be usable.

    This detects cases where OCR completely failed and produced garbage like:
    - ".S89IN0SXe"
    - "~QuaWeanesedus| Jeu900 B,)"
    - "wedinbZ,| e a}UasaYU"
    - "sjuaweoejdep sap aJANe0"

    Args:
        text: The extracted text to check
        min_valid_word_ratio: Minimum ratio of valid words (default: 0.4 = 40%)

    Returns:
        Dict with:
        - is_readable (bool): True if text quality is sufficient
        - valid_word_ratio (float): Ratio of valid-looking words
        - total_chars (int): Total character count
        - reason (str): Reason if not readable
    """
    if not text or len(text.strip()) == 0:
        return {
            "is_readable": False,
            "valid_word_ratio": 0.0,
            "total_chars": 0,
            "reason": "Texte vide"
        }

    # Remove whitespace for analysis
    text_no_space = text.replace(" ", "").replace("\n", "").replace("\t", "")

    if len(text_no_space) == 0:
        return {
            "is_readable": False,
            "valid_word_ratio": 0.0,
            "total_chars": len(text),
            "reason": "Contient uniquement des espaces"
        }

    # Extract words (sequences of alphanumeric characters)
    words = re.findall(r'[a-zA-Z0-9éèêëàâäïîôöùûüÿæœÉÈÊËÀÂÄÏÎÔÖÙÛÜŸÆŒ]+', text)

    if len(words) == 0:
        return {
            "is_readable": False,
            "valid_word_ratio": 0.0,
            "total_chars": len(text),
            "reason": "Aucun mot détecté"
        }

    # Filter out very short words (less than 2 chars) for analysis
    words_for_analysis = [w for w in words if len(w) >= 2]

    # If we have ANY words, even if very short, accept the text
    # This allows single-word or very short valid documents
    if len(words_for_analysis) == 0:
        # No words >= 2 chars, but if we have single-char words, check if they're valid
        if len(words) > 0:
            # Accept text with at least one word (even single char like "a", "I")
            return {
                "is_readable": True,
                "valid_word_ratio": 1.0,
                "total_chars": len(text),
                "reason": None
            }
        return {
            "is_readable": False,
            "valid_word_ratio": 0.0,
            "total_chars": len(text),
            "reason": "Aucun mot de longueur suffisante"
        }

    # Count valid-looking words
    valid_words = sum(1 for w in words_for_analysis if _is_valid_word(w))
    valid_word_ratio = valid_words / len(words_for_analysis)

    # Check if valid word ratio is too low
    if valid_word_ratio < min_valid_word_ratio:
        return {
            "is_readable": False,
            "valid_word_ratio": valid_word_ratio,
            "total_chars": len(text),
            "reason": f"Trop de mots invalides ({valid_word_ratio:.1%} valides, minimum {min_valid_word_ratio:.0%})"
        }

    # Text passes quality checks
    return {
        "is_readable": True,
        "valid_word_ratio": valid_word_ratio,
        "total_chars": len(text),
        "reason": None
    }


def check_chunks_quality(chunks_data: list, min_readable_ratio: float = 0.5) -> Dict[str, any]:
    """
    Check if chunks overall are of sufficient quality.

    Args:
        chunks_data: List of chunk dictionaries with 'text' field
        min_readable_ratio: Minimum ratio of readable chunks (default: 0.5 = 50%)

    Returns:
        Dict with:
        - is_readable (bool): True if enough chunks are readable
        - readable_count (int): Number of readable chunks
        - total_count (int): Total number of chunks
        - readable_ratio (float): Ratio of readable chunks
        - reason (str): Reason if not readable
    """
    if not chunks_data:
        return {
            "is_readable": False,
            "readable_count": 0,
            "total_count": 0,
            "readable_ratio": 0.0,
            "reason": "Aucun chunk créé"
        }

    total_count = len(chunks_data)
    readable_count = 0

    for chunk in chunks_data:
        text = chunk.get("text", "")
        quality = check_text_quality(text, min_valid_word_ratio=0.3)
        if quality["is_readable"]:
            readable_count += 1

    readable_ratio = readable_count / total_count if total_count > 0 else 0

    if readable_ratio < min_readable_ratio:
        return {
            "is_readable": False,
            "readable_count": readable_count,
            "total_count": total_count,
            "readable_ratio": readable_ratio,
            "reason": f"Trop peu de chunks lisibles ({readable_count}/{total_count} = {readable_ratio:.0%}, minimum {min_readable_ratio:.0%})"
        }

    return {
        "is_readable": True,
        "readable_count": readable_count,
        "total_count": total_count,
        "readable_ratio": readable_ratio,
        "reason": None
    }
