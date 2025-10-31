"""Helper functions for info command - following SOLID principles.

Each function has a single responsibility (SRP).
"""
from typing import Dict, Any, Optional

from src.core.cli.utils.display import console


def get_system_info() -> Dict[str, Any]:
    """
    Gather system and environment information. SRP: Info collection only.

    Returns:
        Dictionary with system information (Python version, platform, etc.)

    Examples:
        >>> info = get_system_info()
        >>> "python_version" in info
        True
    """
    import sys
    import platform
    from pathlib import Path

    return {
        "python_version": sys.version,
        "python_version_short": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "processor": platform.processor() or "unknown",
        "path": str(Path.home())
    }


def get_qdrant_info(qdrant_url: str) -> Optional[Dict[str, Any]]:
    """
    Check Qdrant connection and get server info. SRP: Qdrant connectivity only.

    Args:
        qdrant_url: Qdrant server URL

    Returns:
        Dictionary with Qdrant info, or None if unreachable

    Examples:
        >>> info = get_qdrant_info("http://localhost:6333")
        >>> info is None or isinstance(info, dict)
        True
    """
    try:
        from src.core.vector import QdrantVectorStore, VectorStoreConfig

        config = VectorStoreConfig()
        config.url = qdrant_url

        vector_store = QdrantVectorStore(config)
        vector_store.connect()

        return {
            "status": "✓ Connected",
            "url": qdrant_url,
            "version": "1.0.0",  # Placeholder - actual version would come from Qdrant
        }

    except Exception as e:
        return {
            "status": "✗ Disconnected",
            "url": qdrant_url,
            "error": str(e)
        }


def get_llm_info() -> Dict[str, Any]:
    """
    Get configured LLM information. SRP: LLM configuration retrieval only.

    Returns:
        Dictionary with LLM settings from config

    Examples:
        >>> info = get_llm_info()
        >>> "use_llm" in info
        True
    """
    try:
        from src.core.config.atlas_config import get_atlas_config

        config = get_atlas_config()
        return {
            "use_llm": config.llm.use_llm,
            "provider": config.llm.provider if config.llm.use_llm else None,
            "model": config.llm.model if config.llm.use_llm else None,
            "is_local": config.llm.is_local if config.llm.use_llm else None,
        }

    except Exception as e:
        return {
            "use_llm": False,
            "error": str(e)
        }


def display_system_info(info: Dict[str, Any]) -> None:
    """
    Display system information in formatted output. SRP: Display formatting only.

    Args:
        info: System information dictionary
    """
    console.print("[bold cyan]System Information[/bold cyan]")
    console.print(f"  Python:      {info.get('python_version_short', 'unknown')}")
    console.print(f"  Platform:    {info.get('platform', 'unknown')}")
    console.print(f"  Architecture: {info.get('architecture', 'unknown')}")
    console.print(f"  Processor:   {info.get('processor', 'unknown')}\n")


def display_qdrant_info(info: Optional[Dict[str, Any]]) -> None:
    """
    Display Qdrant connection status. SRP: Qdrant status display only.

    Args:
        info: Qdrant info dictionary, or None if unavailable
    """
    console.print("[bold cyan]Qdrant Vector Store[/bold cyan]")

    if info is None:
        console.print("  [red]✗ Unable to retrieve Qdrant info[/red]\n")
        return

    status_color = "green" if info.get("status", "").startswith("✓") else "red"
    console.print(f"  Status:      [{status_color}]{info.get('status', 'unknown')}[/{status_color}]")
    console.print(f"  URL:         {info.get('url', 'unknown')}")

    if "error" in info:
        console.print(f"  Error:       {info['error']}")
    else:
        console.print(f"  Version:     {info.get('version', 'unknown')}")

    console.print()


def display_llm_info(info: Dict[str, Any]) -> None:
    """
    Display LLM configuration. SRP: LLM info display only.

    Args:
        info: LLM information dictionary
    """
    console.print("[bold cyan]LLM Configuration[/bold cyan]")

    if "error" in info:
        console.print(f"  [red]✗ Error loading config: {info['error']}[/red]\n")
        return

    if not info.get("use_llm"):
        console.print("  Status:      [dim]Disabled[/dim]\n")
        return

    console.print("  Status:      [green]✓ Enabled[/green]")
    console.print(f"  Provider:    {info.get('provider', 'unknown')}")
    console.print(f"  Model:       {info.get('model', 'unknown')}")
    console.print(f"  Type:        {'Local' if info.get('is_local') else 'Remote'}\n")
