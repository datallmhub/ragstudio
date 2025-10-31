"""
Atlas-RAG Workflow Modules

Advanced modules used by Prefect workflows for batch processing,
document analysis, ML scoring, OCR, and complex ingestion pipelines.
"""

__version__ = "0.1.0"

# Workflow modules for Prefect orchestration
__all__ = [
    "analyzer",
    "ingest",
    "io",
    "ml",
    "ocr",
    "router",
]