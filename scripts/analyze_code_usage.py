#!/usr/bin/env python3
"""
Analyze code usage to identify potentially dead code.

This script identifies:
1. Modules that are never imported
2. Functions/classes never referenced
3. Code with 0% coverage that may be unused
"""

import ast
import os
from pathlib import Path
from collections import defaultdict
import re


def find_all_python_files(root_dir):
    """Find all Python files in the project."""
    return list(Path(root_dir).rglob("*.py"))


def extract_imports(file_path):
    """Extract all imports from a Python file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read(), filename=str(file_path))

        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)

        return imports
    except Exception as e:
        return []


def analyze_module_usage(src_dir="src"):
    """Analyze which modules are actually imported/used."""
    all_files = find_all_python_files(".")

    # Track all imports across the project
    all_imports = defaultdict(list)

    for file_path in all_files:
        imports = extract_imports(file_path)
        for imp in imports:
            all_imports[imp].append(str(file_path))

    # Modules with 0% coverage (from previous analysis)
    zero_coverage_modules = [
        "src.core.cache.redis_cache",
        "src.core.cli.app",
        "src.core.cli.commands.batch",
        "src.core.cli.commands.eval",
        "src.core.cli.commands.info",
        "src.core.cli.commands.ingest",
        "src.core.cli.commands.retry",
        "src.core.cli.commands.search",
        "src.core.cli.utils.quality_check",
        "src.core.rag.chain",
        "src.core.rag.feedback_loop",
        "src.core.rag.hybrid_search",
        "src.core.rag.query_expansion",
        "src.core.rag.reranker",
        "src.core.rag.retrievers",
        "src.core.vector.base",
        "src.core.vector.embeddings",
        "src.core.vector.qdrant_store",
    ]

    print("=" * 80)
    print("CODE USAGE ANALYSIS")
    print("=" * 80)
    print()

    # Analyze each zero-coverage module
    actively_used = []
    potentially_dead = []

    for module in zero_coverage_modules:
        # Check if this module is imported anywhere
        imported_by = []
        for imp, files in all_imports.items():
            if module in imp or imp in module:
                imported_by.extend(files)

        if imported_by:
            actively_used.append((module, len(imported_by), imported_by[:3]))
        else:
            potentially_dead.append(module)

    print("✅ ACTIVELY USED MODULES (0% coverage but imported):")
    print("-" * 80)
    for module, count, files in sorted(actively_used, key=lambda x: -x[1]):
        print(f"  {module}")
        print(f"    → Imported {count} times in: {', '.join(files)}")
    print()

    print("⚠️  POTENTIALLY DEAD CODE (0% coverage, no imports found):")
    print("-" * 80)
    for module in potentially_dead:
        print(f"  {module}")
    print()

    # Calculate priority for testing
    print("🎯 TESTING PRIORITY (actively used modules with 0% coverage):")
    print("-" * 80)
    print()

    # Prioritize by usage frequency and module importance
    cli_modules = [m for m, _, _ in actively_used if "cli" in m]
    rag_modules = [m for m, _, _ in actively_used if "rag" in m]
    other_modules = [m for m, _, _ in actively_used if "cli" not in m and "rag" not in m]

    print("1. CLI Commands (user-facing, critical):")
    for i, (module, count, _) in enumerate([m for m in actively_used if m[0] in cli_modules], 1):
        print(f"   {i}. {module[0]} - {count} imports")
    print()

    print("2. RAG/Vector Modules (core functionality):")
    for i, (module, count, _) in enumerate([m for m in actively_used if m[0] in rag_modules], 1):
        print(f"   {i}. {module[0]} - {count} imports")
    print()

    print("3. Other Infrastructure:")
    for i, (module, count, _) in enumerate([m for m in actively_used if m[0] in other_modules], 1):
        print(f"   {i}. {module[0]} - {count} imports")
    print()

    # Summary stats
    total_zero_cov = len(zero_coverage_modules)
    total_used = len(actively_used)
    total_dead = len(potentially_dead)

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total modules with 0% coverage: {total_zero_cov}")
    print(f"  ✅ Actively used (need tests): {total_used} ({total_used/total_zero_cov*100:.1f}%)")
    print(f"  ⚠️  Potentially dead code: {total_dead} ({total_dead/total_zero_cov*100:.1f}%)")
    print()

    return {
        'used': actively_used,
        'dead': potentially_dead,
        'total': total_zero_cov
    }


if __name__ == "__main__":
    analyze_module_usage()
