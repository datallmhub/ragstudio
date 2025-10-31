"""Retry mechanism for Atlas-RAG pipeline.

This module provides a robust retry mechanism with exponential backoff
for handling transient failures during document processing.
"""

from dataclasses import dataclass
from typing import Callable, Optional, Any, TypeVar
from enum import Enum
import time
import logging

LOGGER = logging.getLogger(__name__)

T = TypeVar('T')


class RetryStrategy(str, Enum):
    """Retry strategies."""

    EXPONENTIAL = "exponential"  # 1s, 2s, 4s, 8s, ...
    LINEAR = "linear"            # 1s, 1s, 1s, 1s, ...
    FIXED = "fixed"              # user-defined delay


@dataclass
class RetryConfig:
    """Configuration for retry mechanism."""

    max_attempts: int = 3              # 1 initial + 2 retries
    initial_delay: float = 1.0         # seconds
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    backoff_factor: float = 2.0        # for exponential strategy
    max_delay: float = 30.0            # cap at 30 seconds
    jitter: bool = False               # add random jitter to prevent thundering herd

    def __post_init__(self):
        """Validate configuration."""
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.initial_delay < 0:
            raise ValueError("initial_delay must be >= 0")
        if self.backoff_factor < 1:
            raise ValueError("backoff_factor must be >= 1")
        if self.max_delay < self.initial_delay:
            raise ValueError("max_delay must be >= initial_delay")


class RetryableError(Exception):
    """Exception that triggers a retry.

    Use this for transient errors that might succeed on retry
    (network issues, temporary unavailability, etc.)
    """
    pass


class FatalError(Exception):
    """Exception that stops immediately (no retry).

    Use this for permanent errors that won't be fixed by retrying
    (invalid file format, missing permissions, etc.)
    """
    pass


def calculate_delay(
    attempt: int,
    config: RetryConfig,
) -> float:
    """
    Calculate delay for given attempt number.

    Args:
        attempt: Current attempt number (1-indexed)
        config: Retry configuration

    Returns:
        Delay in seconds
    """
    if config.strategy == RetryStrategy.EXPONENTIAL:
        delay = config.initial_delay * (config.backoff_factor ** (attempt - 1))
    elif config.strategy == RetryStrategy.LINEAR:
        delay = config.initial_delay
    elif config.strategy == RetryStrategy.FIXED:
        delay = config.initial_delay
    else:
        delay = config.initial_delay

    # Cap at max_delay
    delay = min(delay, config.max_delay)

    # Add jitter if enabled (random 0-10% variation)
    if config.jitter:
        import random
        jitter_factor = 1.0 + (random.random() * 0.1 - 0.05)  # nosec B311 - Jitter for retry backoff, not crypto
        delay *= jitter_factor

    return delay


def retry_with_backoff(
    func: Callable[[], T],
    config: Optional[RetryConfig] = None,
    error_handler: Optional[Callable[[Exception, int], None]] = None,
    operation_name: Optional[str] = None,
) -> T:
    """
    Execute function with retry logic.

    Args:
        func: Function to execute (must take no arguments)
        config: Retry configuration (uses defaults if None)
        error_handler: Optional callback for each error (error, attempt)
        operation_name: Optional name for logging

    Returns:
        Function result if successful

    Raises:
        FatalError: If function raises FatalError
        RetryableError: If max retries exceeded
        Exception: Last exception if max retries exceeded

    Example:
        >>> def fetch_data():
        ...     # might fail transiently
        ...     return requests.get("https://api.example.com/data").json()
        ...
        >>> config = RetryConfig(max_attempts=3, initial_delay=1.0)
        >>> result = retry_with_backoff(fetch_data, config=config)
    """
    if config is None:
        config = RetryConfig()

    operation = operation_name or getattr(func, '__name__', 'operation')
    last_error = None

    for attempt in range(1, config.max_attempts + 1):
        try:
            LOGGER.info(f"{operation}: Attempt {attempt}/{config.max_attempts}")
            result = func()

            if attempt > 1:
                LOGGER.info(f"{operation}: Success after {attempt} attempts")

            return result

        except FatalError as e:
            LOGGER.error(f"{operation}: Fatal error encountered: {e}")
            raise

        except Exception as e:
            last_error = e
            LOGGER.warning(f"{operation}: Attempt {attempt} failed: {e}")

            # Call error handler if provided
            if error_handler:
                try:
                    error_handler(e, attempt)
                except Exception as handler_error:
                    LOGGER.error(f"Error handler failed: {handler_error}")

            # If this was the last attempt, don't sleep
            if attempt >= config.max_attempts:
                break

            # Calculate delay and sleep
            delay = calculate_delay(attempt, config)
            LOGGER.info(f"{operation}: Retrying in {delay:.1f}s...")
            time.sleep(delay)

    # Max retries exceeded
    LOGGER.error(f"{operation}: Failed after {config.max_attempts} attempts")
    raise RetryableError(
        f"{operation} failed after {config.max_attempts} attempts: {last_error}"
    ) from last_error


def retry(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL,
) -> Callable:
    """
    Decorator for retry logic.

    Args:
        max_attempts: Maximum number of attempts
        initial_delay: Initial delay in seconds
        strategy: Retry strategy

    Returns:
        Decorated function

    Example:
        >>> @retry(max_attempts=3, initial_delay=2.0)
        ... def fetch_data():
        ...     return requests.get("https://api.example.com/data").json()
        ...
        >>> result = fetch_data()  # will retry automatically
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        def wrapper(*args, **kwargs) -> T:
            config = RetryConfig(
                max_attempts=max_attempts,
                initial_delay=initial_delay,
                strategy=strategy,
            )
            return retry_with_backoff(
                lambda: func(*args, **kwargs),
                config=config,
                operation_name=func.__name__,
            )
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    return decorator


# Preset configurations for common scenarios
RETRY_PRESETS = {
    'quick': RetryConfig(
        max_attempts=2,
        initial_delay=0.5,
        strategy=RetryStrategy.LINEAR,
    ),
    'standard': RetryConfig(
        max_attempts=3,
        initial_delay=1.0,
        strategy=RetryStrategy.EXPONENTIAL,
    ),
    'aggressive': RetryConfig(
        max_attempts=5,
        initial_delay=1.0,
        strategy=RetryStrategy.EXPONENTIAL,
        backoff_factor=2.0,
    ),
    'patient': RetryConfig(
        max_attempts=3,
        initial_delay=2.0,
        strategy=RetryStrategy.EXPONENTIAL,
        backoff_factor=3.0,
        max_delay=60.0,
    ),
}


def get_retry_config(preset: str = 'standard') -> RetryConfig:
    """
    Get a preset retry configuration.

    Args:
        preset: Name of preset ('quick', 'standard', 'aggressive', 'patient')

    Returns:
        RetryConfig instance

    Raises:
        ValueError: If preset not found
    """
    if preset not in RETRY_PRESETS:
        raise ValueError(
            f"Unknown preset: {preset}. "
            f"Available: {', '.join(RETRY_PRESETS.keys())}"
        )
    return RETRY_PRESETS[preset]
