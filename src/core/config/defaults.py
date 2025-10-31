"""
Centralised configuration values for the ChunkForge pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkingConfig:
    strategy: str = "semantic"  # semantic (best), sentence (fast), or token (basic)
    max_tokens: int = 400
    overlap: int = 50  # Must be < max_tokens for Chonkie
    model: str = "gpt-3.5-turbo"  # For tiktoken encoding


@dataclass(frozen=True)
class LanguageConfig:
    allowed_languages: tuple[str, ...] = ("fr", "en")
    ocr_languages: tuple[str, ...] = ("fra", "eng")


@dataclass(frozen=True)
class PipelinePaths:
    raw_dir: str = "data/raw_documents"
    output_path: str = "processed_data/chunks.jsonl"
    report_path: str = "processed_data/ingestion_report.json"
    prefect_output_dir: str = "processed_data/prefect"
    prefect_report_path: str = "processed_data/prefect_run_report.json"


chunking = ChunkingConfig()
languages = LanguageConfig()
paths = PipelinePaths()


__all__ = ["chunking", "languages", "paths", "ChunkingConfig", "LanguageConfig", "PipelinePaths"]
