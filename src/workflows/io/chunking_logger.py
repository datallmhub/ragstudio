"""
CSV logging system for chunking decisions and strategy tracking.
"""

from __future__ import annotations

import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

LOGGER = logging.getLogger(__name__)


def log_chunking_decision(
    document_path: str,
    profile: Dict[str, any],
    strategy_config: Dict[str, any],
    log_file: str = "logs/chunking_decisions.csv"
) -> None:
    """
    Log a chunking decision to CSV file.
    
    Args:
        document_path: Path to the document
        profile: Document profile from analyze_document()
        strategy_config: Strategy configuration from select_chunking_strategy()
        log_file: Path to the CSV log file
    """
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare log entry
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "document": Path(document_path).name,
        "document_path": str(document_path),
        "type": profile.get("type", "unknown"),
        "lang": profile.get("lang", "unknown"),
        "tokens": profile.get("length_tokens", 0),
        "chars": profile.get("length_chars", 0),
        "has_headings": profile.get("has_headings", False),
        "has_tables": profile.get("has_tables", False),
        "hierarchy_depth": profile.get("hierarchy_depth", 1),
        "structure_score": profile.get("structure_score", 0.0),
        "chosen_strategy": strategy_config.get("strategy", "unknown"),
        "strategy_reason": strategy_config.get("reason", "unknown"),
        "max_tokens": strategy_config.get("max_tokens", 0),
        "overlap": strategy_config.get("overlap", 0),
    }
    
    # Write to CSV
    file_exists = log_path.exists()
    
    try:
        with open(log_path, 'a', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                "timestamp", "document", "document_path", "type", "lang",
                "tokens", "chars", "has_headings", "has_tables", "hierarchy_depth",
                "structure_score", "chosen_strategy", "strategy_reason",
                "max_tokens", "overlap"
            ]
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # Write header if file is new
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(log_entry)
            
        LOGGER.info("Chunking decision logged to %s", log_file)
        
    except Exception as exc:
        LOGGER.error("Failed to log chunking decision: %s", exc)


def get_strategy_stats(log_file: str = "logs/chunking_decisions.csv") -> Dict[str, int]:
    """
    Get statistics about strategy usage from the log file.
    
    Args:
        log_file: Path to the CSV log file
        
    Returns:
        Dictionary with strategy usage counts
    """
    log_path = Path(log_file)
    
    if not log_path.exists():
        return {
            "recursive": 0,
            "semantic": 0,
            "parent_child": 0,
            "late": 0,
            "total": 0
        }
    
    stats = {
        "recursive": 0,
        "semantic": 0,
        "parent_child": 0,
        "late": 0,
        "total": 0
    }
    
    try:
        with open(log_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            
            for row in reader:
                strategy = row.get("chosen_strategy", "unknown")
                if strategy in stats:
                    stats[strategy] += 1
                stats["total"] += 1
                
    except Exception as exc:
        LOGGER.error("Failed to read strategy stats: %s", exc)
    
    return stats


def get_recent_decisions(
    log_file: str = "logs/chunking_decisions.csv",
    limit: int = 10
) -> List[Dict[str, any]]:
    """
    Get recent chunking decisions from the log file.
    
    Args:
        log_file: Path to the CSV log file
        limit: Maximum number of recent decisions to return
        
    Returns:
        List of recent decisions
    """
    log_path = Path(log_file)
    
    if not log_path.exists():
        return []
    
    decisions = []
    
    try:
        with open(log_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            
            # Read all rows and sort by timestamp
            rows = list(reader)
            rows.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            
            # Return the most recent ones
            decisions = rows[:limit]
            
    except Exception as exc:
        LOGGER.error("Failed to read recent decisions: %s", exc)
    
    return decisions


def cleanup_old_logs(
    log_file: str = "logs/chunking_decisions.csv",
    days_to_keep: int = 30
) -> None:
    """
    Clean up old log entries to prevent the file from growing too large.
    
    Args:
        log_file: Path to the CSV log file
        days_to_keep: Number of days of logs to keep
    """
    log_path = Path(log_file)
    
    if not log_path.exists():
        return
    
    cutoff_date = datetime.now().timestamp() - (days_to_keep * 24 * 60 * 60)
    
    try:
        # Read all entries
        with open(log_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            rows = list(reader)
        
        # Filter entries newer than cutoff
        filtered_rows = []
        for row in rows:
            timestamp_str = row.get("timestamp", "")
            try:
                timestamp = datetime.fromisoformat(timestamp_str).timestamp()
                if timestamp >= cutoff_date:
                    filtered_rows.append(row)
            except ValueError:
                # Keep rows with invalid timestamps
                filtered_rows.append(row)
        
        # Write back filtered entries
        with open(log_path, 'w', newline='', encoding='utf-8') as csvfile:
            if filtered_rows:
                fieldnames = filtered_rows[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(filtered_rows)
        
        LOGGER.info("Cleaned up old log entries, kept %d entries", len(filtered_rows))
        
    except Exception as exc:
        LOGGER.error("Failed to cleanup old logs: %s", exc)
