"""Pipeline management for Atlas-RAG.

This module provides components for robust pipeline execution including:
- Retry mechanisms with exponential backoff
- Interactive error handling with user prompts
- Run history tracking
- Status management
"""

from src.core.pipeline.status import (
    FileStatus,
    PipelineStatus,
    PipelineStats,
    format_status,
    format_pipeline_status,
)

from src.core.pipeline.retry import (
    RetryConfig,
    RetryStrategy,
    RetryableError,
    FatalError,
    retry_with_backoff,
    retry,
    get_retry_config,
    RETRY_PRESETS,
)

from src.core.pipeline.interactive import (
    UserDecision,
    ExecutionMode,
    InteractivePipelineManager,
    create_pipeline_manager,
    prompt_user_decision,
)

from src.core.pipeline.history import (
    FileResult,
    PipelineRun,
    HistoryManager,
    DEFAULT_HISTORY_DIR,
)

__all__ = [
    # Status
    "FileStatus",
    "PipelineStatus",
    "PipelineStats",
    "format_status",
    "format_pipeline_status",
    # Retry
    "RetryConfig",
    "RetryStrategy",
    "RetryableError",
    "FatalError",
    "retry_with_backoff",
    "retry",
    "get_retry_config",
    "RETRY_PRESETS",
    # Interactive
    "UserDecision",
    "ExecutionMode",
    "InteractivePipelineManager",
    "create_pipeline_manager",
    "prompt_user_decision",
    # History
    "FileResult",
    "PipelineRun",
    "HistoryManager",
    "DEFAULT_HISTORY_DIR",
]
