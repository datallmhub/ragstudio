"""Interactive prompts for user decisions during pipeline execution.

This module provides interactive prompts and execution modes for handling
errors during batch processing.
"""

from enum import Enum
from typing import Optional, Dict, Any
from pathlib import Path
import sys

from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel

console = Console()


class UserDecision(str, Enum):
    """User decisions during interactive mode."""

    CONTINUE = "continue"    # [C] Continue execution with next file
    STOP = "stop"           # [S] Stop pipeline immediately
    IGNORE = "ignore"       # [I] Ignore this error (mark as skipped)

    @classmethod
    def from_char(cls, char: str) -> 'UserDecision':
        """Create decision from single character."""
        char_upper = char.upper()
        if char_upper == 'C':
            return cls.CONTINUE
        elif char_upper == 'S':
            return cls.STOP
        elif char_upper == 'I':
            return cls.IGNORE
        else:
            raise ValueError(f"Invalid decision character: {char}")


class ExecutionMode(str, Enum):
    """Execution modes for pipeline."""

    INTERACTIVE = "interactive"      # Ask user for each error
    AUTO_CONTINUE = "auto_continue"  # Continue automatically
    AUTO_STOP = "auto_stop"          # Stop on first error
    AUTO_SKIP = "auto_skip"          # Skip failed files automatically

    def is_interactive(self) -> bool:
        """Check if mode requires user interaction."""
        return self == ExecutionMode.INTERACTIVE

    def get_auto_decision(self) -> UserDecision:
        """Get automatic decision for non-interactive modes."""
        if self == ExecutionMode.AUTO_CONTINUE:
            return UserDecision.CONTINUE
        elif self == ExecutionMode.AUTO_STOP:
            return UserDecision.STOP
        elif self == ExecutionMode.AUTO_SKIP:
            return UserDecision.IGNORE
        else:
            raise ValueError(f"Mode {self} does not have auto decision")


def prompt_user_decision(
    error: Exception,
    file_path: Path,
    attempt: int,
    context: Optional[Dict[str, Any]] = None,
) -> UserDecision:
    """
    Prompt user for decision after error.

    Args:
        error: The exception that occurred
        file_path: Path to the file being processed
        attempt: Current attempt number
        context: Optional context dict with additional info

    Returns:
        User's decision

    Example:
        >>> decision = prompt_user_decision(
        ...     error=ValueError("OCR failed"),
        ...     file_path=Path("scan.pdf"),
        ...     attempt=3,
        ...     context={"step": "OCR extraction"}
        ... )
    """
    console.print()

    # Create error panel
    error_lines = [
        f"[yellow]Fichier:[/yellow] {file_path.name}",
        f"[yellow]Erreur:[/yellow] {error}",
    ]

    if context:
        if "step" in context:
            error_lines.append(f"[yellow]Étape:[/yellow] {context['step']}")
        if "reason" in context:
            error_lines.append(f"[yellow]Raison:[/yellow] {context['reason']}")

    error_lines.append(f"[dim]Tentatives effectuées: {attempt}[/dim]")

    panel = Panel(
        "\n".join(error_lines),
        title="⚠️  Échec de traitement",
        border_style="red",
        expand=False,
    )
    console.print(panel)
    console.print()

    # Show options
    console.print("[bold cyan]👉 Que souhaitez-vous faire ?[/bold cyan]")
    console.print("   [bold green][C][/bold green] Continuer  - Continue avec le fichier suivant")
    console.print("   [bold red][S][/bold red] Stopper     - Arrête immédiatement le pipeline")
    console.print("   [bold yellow][I][/bold yellow] Ignorer    - Marque comme 'skipped' et continue")
    console.print()

    # Get user input
    while True:
        try:
            choice = Prompt.ask(
                "Votre choix",
                choices=["C", "c", "S", "s", "I", "i"],
                default="C",
                show_choices=False,
            ).upper()

            decision = UserDecision.from_char(choice)

            # Show confirmation
            if decision == UserDecision.CONTINUE:
                console.print("[green]✓[/green] Continue avec le fichier suivant...")
            elif decision == UserDecision.STOP:
                console.print("[red]⏹[/red] Arrêt du pipeline demandé")
            elif decision == UserDecision.IGNORE:
                console.print("[yellow]⏭[/yellow] Fichier ignoré (skipped)")

            console.print()
            return decision

        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]⚠️[/yellow] Interruption détectée, arrêt du pipeline")
            return UserDecision.STOP
        except Exception as e:
            console.print(f"[red]Erreur:[/red] {e}")
            console.print("Veuillez choisir C, S ou I")


def get_user_decision_for_mode(
    mode: ExecutionMode,
    error: Exception,
    file_path: Path,
    attempt: int,
    context: Optional[Dict[str, Any]] = None,
) -> UserDecision:
    """
    Get user decision based on execution mode.

    For interactive mode, prompts the user.
    For auto modes, returns automatic decision and logs it.

    Args:
        mode: Execution mode
        error: The exception
        file_path: File path
        attempt: Attempt number
        context: Optional context

    Returns:
        Decision to take
    """
    if mode.is_interactive():
        return prompt_user_decision(error, file_path, attempt, context)

    # Auto modes
    decision = mode.get_auto_decision()

    # Log the auto decision
    if decision == UserDecision.CONTINUE:
        console.print(
            f"[yellow]⚠️[/yellow] Erreur sur {file_path.name}, "
            f"continue automatiquement (mode: {mode.value})"
        )
    elif decision == UserDecision.STOP:
        console.print(
            f"[red]⏹[/red] Erreur sur {file_path.name}, "
            f"arrêt immédiat (mode: {mode.value})"
        )
    elif decision == UserDecision.IGNORE:
        console.print(
            f"[yellow]⏭[/yellow] Erreur sur {file_path.name}, "
            f"fichier ignoré (mode: {mode.value})"
        )

    return decision


class InteractivePipelineManager:
    """Manages interactive pipeline execution with error handling."""

    def __init__(self, mode: ExecutionMode = ExecutionMode.INTERACTIVE):
        """
        Initialize manager.

        Args:
            mode: Execution mode
        """
        self.mode = mode
        self.decisions_history: list[Dict[str, Any]] = []
        self._stopped = False

    def handle_error(
        self,
        error: Exception,
        file_path: Path,
        attempt: int,
        context: Optional[Dict[str, Any]] = None,
    ) -> UserDecision:
        """
        Handle an error during pipeline execution.

        Args:
            error: The exception
            file_path: File being processed
            attempt: Attempt number
            context: Optional context

        Returns:
            User's decision
        """
        # Get decision
        decision = get_user_decision_for_mode(
            self.mode, error, file_path, attempt, context
        )

        # Record decision
        self.decisions_history.append({
            "file": str(file_path),
            "error": str(error),
            "error_type": type(error).__name__,
            "attempt": attempt,
            "decision": decision.value,
            "context": context or {},
        })

        # Mark as stopped if user chose STOP
        if decision == UserDecision.STOP:
            self._stopped = True

        return decision

    def should_continue(self, decision: Optional[UserDecision] = None) -> bool:
        """
        Check if pipeline should continue.

        Args:
            decision: Optional decision to check (uses internal state if None)

        Returns:
            True if should continue, False if should stop
        """
        if self._stopped:
            return False

        if decision is not None:
            return decision != UserDecision.STOP

        return True

    def is_stopped(self) -> bool:
        """Check if pipeline has been stopped."""
        return self._stopped

    def get_decisions_summary(self) -> Dict[str, Any]:
        """Get summary of all decisions made."""
        total = len(self.decisions_history)
        if total == 0:
            return {
                "total": 0,
                "continue": 0,
                "stop": 0,
                "ignore": 0,
            }

        continue_count = sum(
            1 for d in self.decisions_history
            if d["decision"] == UserDecision.CONTINUE.value
        )
        stop_count = sum(
            1 for d in self.decisions_history
            if d["decision"] == UserDecision.STOP.value
        )
        ignore_count = sum(
            1 for d in self.decisions_history
            if d["decision"] == UserDecision.IGNORE.value
        )

        return {
            "total": total,
            "continue": continue_count,
            "stop": stop_count,
            "ignore": ignore_count,
            "history": self.decisions_history,
        }


def create_pipeline_manager(
    interactive: bool = True,
    auto_continue: bool = False,
    auto_stop: bool = False,
    auto_skip: bool = False,
) -> InteractivePipelineManager:
    """
    Create pipeline manager based on CLI flags.

    Args:
        interactive: Enable interactive mode (default)
        auto_continue: Auto-continue mode
        auto_stop: Auto-stop mode
        auto_skip: Auto-skip mode

    Returns:
        InteractivePipelineManager instance

    Example:
        >>> # From CLI args
        >>> manager = create_pipeline_manager(
        ...     auto_continue=args.auto_continue,
        ...     auto_stop=args.auto_stop,
        ... )
    """
    # Determine mode from flags (priority: stop > skip > continue > interactive)
    if auto_stop:
        mode = ExecutionMode.AUTO_STOP
    elif auto_skip:
        mode = ExecutionMode.AUTO_SKIP
    elif auto_continue:
        mode = ExecutionMode.AUTO_CONTINUE
    else:
        mode = ExecutionMode.INTERACTIVE

    return InteractivePipelineManager(mode=mode)
