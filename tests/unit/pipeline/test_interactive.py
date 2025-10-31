"""Unit tests for src.core.pipeline.interactive module."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from src.core.pipeline.interactive import (
    UserDecision,
    ExecutionMode,
    InteractivePipelineManager,
    create_pipeline_manager,
    get_user_decision_for_mode,
)


class TestUserDecision:
    """Tests for UserDecision enum."""

    def test_all_decisions_exist(self):
        """Test that all expected decisions are defined."""
        assert UserDecision.CONTINUE == "continue"
        assert UserDecision.STOP == "stop"
        assert UserDecision.IGNORE == "ignore"

    def test_from_char_continue(self):
        """Test from_char with 'C'."""
        assert UserDecision.from_char("C") == UserDecision.CONTINUE
        assert UserDecision.from_char("c") == UserDecision.CONTINUE

    def test_from_char_stop(self):
        """Test from_char with 'S'."""
        assert UserDecision.from_char("S") == UserDecision.STOP
        assert UserDecision.from_char("s") == UserDecision.STOP

    def test_from_char_ignore(self):
        """Test from_char with 'I'."""
        assert UserDecision.from_char("I") == UserDecision.IGNORE
        assert UserDecision.from_char("i") == UserDecision.IGNORE

    def test_from_char_invalid(self):
        """Test from_char with invalid character."""
        with pytest.raises(ValueError, match="Invalid decision character"):
            UserDecision.from_char("X")


class TestExecutionMode:
    """Tests for ExecutionMode enum."""

    def test_all_modes_exist(self):
        """Test that all expected modes are defined."""
        assert ExecutionMode.INTERACTIVE == "interactive"
        assert ExecutionMode.AUTO_CONTINUE == "auto_continue"
        assert ExecutionMode.AUTO_STOP == "auto_stop"
        assert ExecutionMode.AUTO_SKIP == "auto_skip"

    def test_is_interactive(self):
        """Test is_interactive() method."""
        assert ExecutionMode.INTERACTIVE.is_interactive()
        assert not ExecutionMode.AUTO_CONTINUE.is_interactive()
        assert not ExecutionMode.AUTO_STOP.is_interactive()
        assert not ExecutionMode.AUTO_SKIP.is_interactive()

    def test_get_auto_decision_continue(self):
        """Test get_auto_decision for AUTO_CONTINUE."""
        decision = ExecutionMode.AUTO_CONTINUE.get_auto_decision()
        assert decision == UserDecision.CONTINUE

    def test_get_auto_decision_stop(self):
        """Test get_auto_decision for AUTO_STOP."""
        decision = ExecutionMode.AUTO_STOP.get_auto_decision()
        assert decision == UserDecision.STOP

    def test_get_auto_decision_skip(self):
        """Test get_auto_decision for AUTO_SKIP."""
        decision = ExecutionMode.AUTO_SKIP.get_auto_decision()
        assert decision == UserDecision.IGNORE

    def test_get_auto_decision_interactive_raises(self):
        """Test get_auto_decision raises for INTERACTIVE mode."""
        with pytest.raises(ValueError):
            ExecutionMode.INTERACTIVE.get_auto_decision()


class TestGetUserDecisionForMode:
    """Tests for get_user_decision_for_mode function."""

    @patch('src.core.pipeline.interactive.prompt_user_decision')
    def test_interactive_mode_prompts(self, mock_prompt):
        """Test that interactive mode prompts the user."""
        mock_prompt.return_value = UserDecision.CONTINUE
        error = Exception("test error")
        file_path = Path("test.txt")

        decision = get_user_decision_for_mode(
            ExecutionMode.INTERACTIVE,
            error,
            file_path,
            attempt=3,
        )

        assert decision == UserDecision.CONTINUE
        mock_prompt.assert_called_once()

    def test_auto_continue_mode(self):
        """Test AUTO_CONTINUE mode returns CONTINUE."""
        error = Exception("test error")
        file_path = Path("test.txt")

        decision = get_user_decision_for_mode(
            ExecutionMode.AUTO_CONTINUE,
            error,
            file_path,
            attempt=3,
        )

        assert decision == UserDecision.CONTINUE

    def test_auto_stop_mode(self):
        """Test AUTO_STOP mode returns STOP."""
        error = Exception("test error")
        file_path = Path("test.txt")

        decision = get_user_decision_for_mode(
            ExecutionMode.AUTO_STOP,
            error,
            file_path,
            attempt=3,
        )

        assert decision == UserDecision.STOP

    def test_auto_skip_mode(self):
        """Test AUTO_SKIP mode returns IGNORE."""
        error = Exception("test error")
        file_path = Path("test.txt")

        decision = get_user_decision_for_mode(
            ExecutionMode.AUTO_SKIP,
            error,
            file_path,
            attempt=3,
        )

        assert decision == UserDecision.IGNORE


class TestInteractivePipelineManager:
    """Tests for InteractivePipelineManager class."""

    def test_initialization_default(self):
        """Test default initialization."""
        manager = InteractivePipelineManager()
        assert manager.mode == ExecutionMode.INTERACTIVE
        assert manager.decisions_history == []
        assert not manager._stopped

    def test_initialization_custom_mode(self):
        """Test initialization with custom mode."""
        manager = InteractivePipelineManager(mode=ExecutionMode.AUTO_CONTINUE)
        assert manager.mode == ExecutionMode.AUTO_CONTINUE

    @patch('src.core.pipeline.interactive.get_user_decision_for_mode')
    def test_handle_error(self, mock_get_decision):
        """Test handle_error method."""
        mock_get_decision.return_value = UserDecision.CONTINUE

        manager = InteractivePipelineManager()
        error = Exception("test error")
        file_path = Path("test.txt")

        decision = manager.handle_error(error, file_path, attempt=3)

        assert decision == UserDecision.CONTINUE
        assert len(manager.decisions_history) == 1

        history_entry = manager.decisions_history[0]
        assert history_entry["file"] == str(file_path)
        assert history_entry["error"] == str(error)
        assert history_entry["attempt"] == 3
        assert history_entry["decision"] == UserDecision.CONTINUE.value

    @patch('src.core.pipeline.interactive.get_user_decision_for_mode')
    def test_handle_error_with_context(self, mock_get_decision):
        """Test handle_error with context."""
        mock_get_decision.return_value = UserDecision.IGNORE

        manager = InteractivePipelineManager()
        error = Exception("OCR failed")
        file_path = Path("scan.pdf")
        context = {"step": "OCR", "reason": "Low confidence"}

        decision = manager.handle_error(error, file_path, attempt=2, context=context)

        assert decision == UserDecision.IGNORE
        assert manager.decisions_history[0]["context"] == context

    @patch('src.core.pipeline.interactive.get_user_decision_for_mode')
    def test_handle_error_stop_sets_flag(self, mock_get_decision):
        """Test that STOP decision sets _stopped flag."""
        mock_get_decision.return_value = UserDecision.STOP

        manager = InteractivePipelineManager()
        error = Exception("test error")
        file_path = Path("test.txt")

        decision = manager.handle_error(error, file_path, attempt=3)

        assert decision == UserDecision.STOP
        assert manager._stopped

    def test_should_continue_with_continue_decision(self):
        """Test should_continue returns True for CONTINUE."""
        manager = InteractivePipelineManager()
        assert manager.should_continue(UserDecision.CONTINUE)

    def test_should_continue_with_ignore_decision(self):
        """Test should_continue returns True for IGNORE."""
        manager = InteractivePipelineManager()
        assert manager.should_continue(UserDecision.IGNORE)

    def test_should_continue_with_stop_decision(self):
        """Test should_continue returns False for STOP."""
        manager = InteractivePipelineManager()
        assert not manager.should_continue(UserDecision.STOP)

    def test_should_continue_after_stopped(self):
        """Test should_continue returns False after pipeline stopped."""
        manager = InteractivePipelineManager()
        manager._stopped = True
        assert not manager.should_continue()

    def test_is_stopped(self):
        """Test is_stopped method."""
        manager = InteractivePipelineManager()
        assert not manager.is_stopped()

        manager._stopped = True
        assert manager.is_stopped()

    @patch('src.core.pipeline.interactive.get_user_decision_for_mode')
    def test_get_decisions_summary_empty(self, mock_get_decision):
        """Test get_decisions_summary with no decisions."""
        manager = InteractivePipelineManager()
        summary = manager.get_decisions_summary()

        assert summary["total"] == 0
        assert summary["continue"] == 0
        assert summary["stop"] == 0
        assert summary["ignore"] == 0

    @patch('src.core.pipeline.interactive.get_user_decision_for_mode')
    def test_get_decisions_summary_with_decisions(self, mock_get_decision):
        """Test get_decisions_summary with multiple decisions."""
        manager = InteractivePipelineManager()

        # Simulate multiple errors
        mock_get_decision.side_effect = [
            UserDecision.CONTINUE,
            UserDecision.CONTINUE,
            UserDecision.IGNORE,
            UserDecision.STOP,
        ]

        for i in range(4):
            manager.handle_error(
                Exception(f"error{i}"),
                Path(f"file{i}.txt"),
                attempt=3
            )

        summary = manager.get_decisions_summary()

        assert summary["total"] == 4
        assert summary["continue"] == 2
        assert summary["stop"] == 1
        assert summary["ignore"] == 1
        assert len(summary["history"]) == 4


class TestCreatePipelineManager:
    """Tests for create_pipeline_manager factory function."""

    def test_create_interactive_default(self):
        """Test creating interactive manager (default)."""
        manager = create_pipeline_manager()
        assert manager.mode == ExecutionMode.INTERACTIVE

    def test_create_interactive_explicit(self):
        """Test creating interactive manager explicitly."""
        manager = create_pipeline_manager(interactive=True)
        assert manager.mode == ExecutionMode.INTERACTIVE

    def test_create_auto_continue(self):
        """Test creating auto-continue manager."""
        manager = create_pipeline_manager(auto_continue=True)
        assert manager.mode == ExecutionMode.AUTO_CONTINUE

    def test_create_auto_stop(self):
        """Test creating auto-stop manager."""
        manager = create_pipeline_manager(auto_stop=True)
        assert manager.mode == ExecutionMode.AUTO_STOP

    def test_create_auto_skip(self):
        """Test creating auto-skip manager."""
        manager = create_pipeline_manager(auto_skip=True)
        assert manager.mode == ExecutionMode.AUTO_SKIP

    def test_priority_auto_stop_over_others(self):
        """Test that auto_stop has priority."""
        manager = create_pipeline_manager(
            auto_continue=True,
            auto_stop=True,
            auto_skip=True,
        )
        assert manager.mode == ExecutionMode.AUTO_STOP

    def test_priority_auto_skip_over_continue(self):
        """Test that auto_skip has priority over auto_continue."""
        manager = create_pipeline_manager(
            auto_continue=True,
            auto_skip=True,
        )
        assert manager.mode == ExecutionMode.AUTO_SKIP


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @patch('src.core.pipeline.interactive.get_user_decision_for_mode')
    def test_multiple_errors_same_file(self, mock_get_decision):
        """Test handling multiple errors for the same file."""
        mock_get_decision.return_value = UserDecision.CONTINUE

        manager = InteractivePipelineManager()
        file_path = Path("problematic.txt")

        for i in range(3):
            manager.handle_error(
                Exception(f"error{i}"),
                file_path,
                attempt=i + 1
            )

        assert len(manager.decisions_history) == 3
        # All should be for the same file
        assert all(d["file"] == str(file_path) for d in manager.decisions_history)

    @patch('src.core.pipeline.interactive.get_user_decision_for_mode')
    def test_error_types_recorded(self, mock_get_decision):
        """Test that error types are recorded correctly."""
        mock_get_decision.return_value = UserDecision.CONTINUE

        manager = InteractivePipelineManager()

        # Different error types
        errors = [
            ValueError("value error"),
            TypeError("type error"),
            RuntimeError("runtime error"),
        ]

        for error in errors:
            manager.handle_error(error, Path("test.txt"), attempt=1)

        # Check error types are recorded
        error_types = [d["error_type"] for d in manager.decisions_history]
        assert "ValueError" in error_types
        assert "TypeError" in error_types
        assert "RuntimeError" in error_types

    @patch('src.core.pipeline.interactive.get_user_decision_for_mode')
    def test_empty_context(self, mock_get_decision):
        """Test handle_error with None context."""
        mock_get_decision.return_value = UserDecision.CONTINUE

        manager = InteractivePipelineManager()
        manager.handle_error(
            Exception("error"),
            Path("test.txt"),
            attempt=1,
            context=None
        )

        assert manager.decisions_history[0]["context"] == {}
