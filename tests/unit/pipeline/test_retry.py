"""Unit tests for src.core.pipeline.retry module."""

import pytest
import time
from unittest.mock import Mock, call

from src.core.pipeline.retry import (
    RetryStrategy,
    RetryConfig,
    RetryableError,
    FatalError,
    retry_with_backoff,
    retry,
    calculate_delay,
    get_retry_config,
    RETRY_PRESETS,
)


class TestRetryStrategy:
    """Tests for RetryStrategy enum."""

    def test_all_strategies_exist(self):
        """Test that all expected strategies are defined."""
        assert RetryStrategy.EXPONENTIAL == "exponential"
        assert RetryStrategy.LINEAR == "linear"
        assert RetryStrategy.FIXED == "fixed"


class TestRetryConfig:
    """Tests for RetryConfig dataclass."""

    def test_default_initialization(self):
        """Test default configuration."""
        config = RetryConfig()
        assert config.max_attempts == 3
        assert config.initial_delay == 1.0
        assert config.strategy == RetryStrategy.EXPONENTIAL
        assert config.backoff_factor == 2.0
        assert config.max_delay == 30.0
        assert config.jitter is False

    def test_custom_initialization(self):
        """Test custom configuration."""
        config = RetryConfig(
            max_attempts=5,
            initial_delay=2.0,
            strategy=RetryStrategy.LINEAR,
            backoff_factor=3.0,
            max_delay=60.0,
            jitter=True,
        )
        assert config.max_attempts == 5
        assert config.initial_delay == 2.0
        assert config.strategy == RetryStrategy.LINEAR
        assert config.backoff_factor == 3.0
        assert config.max_delay == 60.0
        assert config.jitter is True

    def test_validation_max_attempts(self):
        """Test validation for max_attempts."""
        with pytest.raises(ValueError, match="max_attempts must be >= 1"):
            RetryConfig(max_attempts=0)

    def test_validation_initial_delay(self):
        """Test validation for initial_delay."""
        with pytest.raises(ValueError, match="initial_delay must be >= 0"):
            RetryConfig(initial_delay=-1.0)

    def test_validation_backoff_factor(self):
        """Test validation for backoff_factor."""
        with pytest.raises(ValueError, match="backoff_factor must be >= 1"):
            RetryConfig(backoff_factor=0.5)

    def test_validation_max_delay(self):
        """Test validation for max_delay."""
        with pytest.raises(ValueError, match="max_delay must be >= initial_delay"):
            RetryConfig(initial_delay=5.0, max_delay=2.0)


class TestCalculateDelay:
    """Tests for calculate_delay function."""

    def test_exponential_backoff(self):
        """Test exponential backoff calculation."""
        config = RetryConfig(
            initial_delay=1.0,
            strategy=RetryStrategy.EXPONENTIAL,
            backoff_factor=2.0,
        )

        assert calculate_delay(1, config) == 1.0  # 1.0 * 2^0
        assert calculate_delay(2, config) == 2.0  # 1.0 * 2^1
        assert calculate_delay(3, config) == 4.0  # 1.0 * 2^2
        assert calculate_delay(4, config) == 8.0  # 1.0 * 2^3

    def test_linear_backoff(self):
        """Test linear backoff (constant delay)."""
        config = RetryConfig(
            initial_delay=2.0,
            strategy=RetryStrategy.LINEAR,
        )

        assert calculate_delay(1, config) == 2.0
        assert calculate_delay(2, config) == 2.0
        assert calculate_delay(3, config) == 2.0

    def test_fixed_backoff(self):
        """Test fixed backoff (constant delay)."""
        config = RetryConfig(
            initial_delay=3.0,
            strategy=RetryStrategy.FIXED,
        )

        assert calculate_delay(1, config) == 3.0
        assert calculate_delay(2, config) == 3.0
        assert calculate_delay(3, config) == 3.0

    def test_max_delay_cap(self):
        """Test that delay is capped at max_delay."""
        config = RetryConfig(
            initial_delay=1.0,
            strategy=RetryStrategy.EXPONENTIAL,
            backoff_factor=2.0,
            max_delay=5.0,
        )

        assert calculate_delay(1, config) == 1.0
        assert calculate_delay(2, config) == 2.0
        assert calculate_delay(3, config) == 4.0
        assert calculate_delay(4, config) == 5.0  # Capped at max_delay
        assert calculate_delay(5, config) == 5.0  # Still capped

    def test_jitter(self):
        """Test that jitter adds randomness."""
        config = RetryConfig(
            initial_delay=10.0,
            strategy=RetryStrategy.LINEAR,
            jitter=True,
        )

        # With jitter, delays should vary slightly
        delays = [calculate_delay(1, config) for _ in range(10)]

        # All delays should be close to 10.0 but not all exactly 10.0
        assert all(9.5 <= d <= 10.5 for d in delays)
        assert len(set(delays)) > 1  # Should have some variation


class TestRetryWithBackoff:
    """Tests for retry_with_backoff function."""

    def test_success_first_attempt(self):
        """Test successful execution on first attempt."""
        mock_func = Mock(return_value="success")

        result = retry_with_backoff(mock_func, config=RetryConfig(max_attempts=3))

        assert result == "success"
        assert mock_func.call_count == 1

    def test_success_after_failures(self):
        """Test success after some failures."""
        mock_func = Mock(side_effect=[Exception("fail"), Exception("fail"), "success"])

        result = retry_with_backoff(
            mock_func,
            config=RetryConfig(max_attempts=3, initial_delay=0.01)  # Fast for testing
        )

        assert result == "success"
        assert mock_func.call_count == 3

    def test_max_retries_exceeded(self):
        """Test that RetryableError is raised after max retries."""
        mock_func = Mock(side_effect=Exception("persistent error"))

        with pytest.raises(RetryableError, match="failed after 3 attempts"):
            retry_with_backoff(
                mock_func,
                config=RetryConfig(max_attempts=3, initial_delay=0.01)
            )

        assert mock_func.call_count == 3

    def test_fatal_error_no_retry(self):
        """Test that FatalError stops immediately without retry."""
        mock_func = Mock(side_effect=FatalError("fatal"))

        with pytest.raises(FatalError, match="fatal"):
            retry_with_backoff(
                mock_func,
                config=RetryConfig(max_attempts=3)
            )

        assert mock_func.call_count == 1  # Only one attempt

    def test_error_handler_called(self):
        """Test that error handler is called for each error."""
        mock_func = Mock(side_effect=[Exception("error1"), Exception("error2"), "success"])
        error_handler = Mock()

        result = retry_with_backoff(
            mock_func,
            config=RetryConfig(max_attempts=3, initial_delay=0.01),
            error_handler=error_handler,
        )

        assert result == "success"
        assert error_handler.call_count == 2
        # Check that error handler was called with errors and attempt numbers
        call_args_list = error_handler.call_args_list
        assert len(call_args_list) == 2
        # First call: error with "error1", attempt 1
        assert str(call_args_list[0][0][0]) == "error1"
        assert call_args_list[0][0][1] == 1
        # Second call: error with "error2", attempt 2
        assert str(call_args_list[1][0][0]) == "error2"
        assert call_args_list[1][0][1] == 2

    def test_operation_name_in_logs(self, caplog):
        """Test that operation name appears in logs."""
        mock_func = Mock(return_value="success")

        retry_with_backoff(
            mock_func,
            config=RetryConfig(max_attempts=3),
            operation_name="TestOperation"
        )

        assert "TestOperation" in caplog.text

    def test_delays_are_applied(self):
        """Test that delays are actually applied between retries."""
        mock_func = Mock(side_effect=[Exception("fail"), "success"])

        start_time = time.time()
        result = retry_with_backoff(
            mock_func,
            config=RetryConfig(max_attempts=3, initial_delay=0.1)
        )
        elapsed_time = time.time() - start_time

        assert result == "success"
        # Should have at least one delay of ~0.1 seconds
        assert elapsed_time >= 0.1


class TestRetryDecorator:
    """Tests for @retry decorator."""

    def test_decorator_success(self):
        """Test decorator with successful function."""
        @retry(max_attempts=3, initial_delay=0.01)
        def successful_func():
            return "success"

        result = successful_func()
        assert result == "success"

    def test_decorator_with_failures(self):
        """Test decorator with failures then success."""
        call_count = {"count": 0}

        @retry(max_attempts=3, initial_delay=0.01)
        def flaky_func():
            call_count["count"] += 1
            if call_count["count"] < 2:
                raise Exception("fail")
            return "success"

        result = flaky_func()
        assert result == "success"
        assert call_count["count"] == 2

    def test_decorator_max_retries(self):
        """Test decorator raises after max retries."""
        @retry(max_attempts=2, initial_delay=0.01)
        def always_fails():
            raise Exception("always fails")

        with pytest.raises(RetryableError):
            always_fails()

    def test_decorator_preserves_function_metadata(self):
        """Test that decorator preserves function name and docstring."""
        @retry(max_attempts=3)
        def my_function():
            """My docstring."""
            return "result"

        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ == "My docstring."


class TestRetryPresets:
    """Tests for retry presets."""

    def test_all_presets_exist(self):
        """Test that all expected presets are defined."""
        assert "quick" in RETRY_PRESETS
        assert "standard" in RETRY_PRESETS
        assert "aggressive" in RETRY_PRESETS
        assert "patient" in RETRY_PRESETS

    def test_quick_preset(self):
        """Test quick preset configuration."""
        config = RETRY_PRESETS["quick"]
        assert config.max_attempts == 2
        assert config.initial_delay == 0.5
        assert config.strategy == RetryStrategy.LINEAR

    def test_standard_preset(self):
        """Test standard preset configuration."""
        config = RETRY_PRESETS["standard"]
        assert config.max_attempts == 3
        assert config.initial_delay == 1.0
        assert config.strategy == RetryStrategy.EXPONENTIAL

    def test_aggressive_preset(self):
        """Test aggressive preset configuration."""
        config = RETRY_PRESETS["aggressive"]
        assert config.max_attempts == 5
        assert config.initial_delay == 1.0
        assert config.strategy == RetryStrategy.EXPONENTIAL

    def test_patient_preset(self):
        """Test patient preset configuration."""
        config = RETRY_PRESETS["patient"]
        assert config.max_attempts == 3
        assert config.initial_delay == 2.0
        assert config.strategy == RetryStrategy.EXPONENTIAL
        assert config.max_delay == 60.0

    def test_get_retry_config_valid(self):
        """Test get_retry_config with valid preset."""
        config = get_retry_config("standard")
        assert config.max_attempts == 3
        assert config.initial_delay == 1.0

    def test_get_retry_config_invalid(self):
        """Test get_retry_config with invalid preset."""
        with pytest.raises(ValueError, match="Unknown preset"):
            get_retry_config("nonexistent")


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_zero_delay(self):
        """Test retry with zero delay."""
        mock_func = Mock(side_effect=[Exception("fail"), "success"])

        result = retry_with_backoff(
            mock_func,
            config=RetryConfig(max_attempts=3, initial_delay=0.0)
        )

        assert result == "success"
        assert mock_func.call_count == 2

    def test_single_attempt(self):
        """Test with max_attempts=1 (no retries)."""
        mock_func = Mock(side_effect=Exception("fail"))

        with pytest.raises(RetryableError):
            retry_with_backoff(
                mock_func,
                config=RetryConfig(max_attempts=1)
            )

        assert mock_func.call_count == 1

    def test_error_handler_exception(self):
        """Test that exception in error handler doesn't break retry."""
        mock_func = Mock(side_effect=[Exception("fail"), "success"])
        error_handler = Mock(side_effect=Exception("handler error"))

        # Should still succeed despite error handler failing
        result = retry_with_backoff(
            mock_func,
            config=RetryConfig(max_attempts=3, initial_delay=0.01),
            error_handler=error_handler,
        )

        assert result == "success"

    def test_very_large_backoff(self):
        """Test that very large backoff is capped by max_delay."""
        config = RetryConfig(
            initial_delay=1.0,
            strategy=RetryStrategy.EXPONENTIAL,
            backoff_factor=10.0,
            max_delay=5.0,
        )

        # Even with large backoff factor, should be capped
        assert calculate_delay(10, config) == 5.0
