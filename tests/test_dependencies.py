"""
Dependency and environment check for CineSense
Can be run directly (python tests/test_dependencies.py) or via pytest.
Does not call sys.exit during pytest collection.
"""

import sys
import importlib
import pytest

CORE_CHECKS = [
    ("Flask", "flask"),
    ("MySQL Connector", "mysql.connector"),
    ("NumPy", "numpy"),
    ("scikit-learn", "sklearn"),
    ("Requests", "requests"),
]

OPTIONAL_CHECKS = [
    ("PyTorch", "torch"),
    ("Transformers", "transformers"),
    ("Sentence Transformers", "sentence_transformers"),
    ("FAISS", "faiss"),
    ("Pillow (PIL)", "PIL"),
    ("Redis", "redis"),
    ("SHAP", "shap"),
]


def check_module(module_name: str) -> bool:
    """Check if a module can be imported without crashing."""
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False


def test_core_dependencies():
    """Verify that essential core modules required to run CineSense are installed."""
    missing = []
    for display_name, mod in CORE_CHECKS:
        if not check_module(mod):
            missing.append(f"{display_name} ({mod})")
    assert not missing, f"Missing required core dependencies: {', '.join(missing)}"


@pytest.mark.parametrize("display_name,module_name", OPTIONAL_CHECKS)
def test_optional_dependencies(display_name, module_name):
    """Check optional dependencies with targeted skips instead of failing test collection."""
    if not check_module(module_name):
        pytest.skip(f"Optional dependency not installed: {display_name} ({module_name})")


def run_cli_diagnostic():
    """CLI diagnostic output when run as a standalone script."""
    print("\n" + "=" * 60)
    print("CineSense Dependency & Environment Diagnostic")
    print("=" * 60 + "\n")

    print("Core Dependencies:")
    print("-" * 60)
    core_ok = 0
    for display, mod in CORE_CHECKS:
        if check_module(mod):
            print(f"✓ {display:<30} INSTALLED")
            core_ok += 1
        else:
            print(f"✗ {display:<30} MISSING")

    print("\nOptional Dependencies:")
    print("-" * 60)
    opt_ok = 0
    for display, mod in OPTIONAL_CHECKS:
        if check_module(mod):
            print(f"✓ {display:<30} INSTALLED")
            opt_ok += 1
        else:
            print(f"⚠ {display:<30} NOT INSTALLED (optional)")

    print("\n" + "=" * 60)
    print(f"Core: {core_ok}/{len(CORE_CHECKS)} installed")
    print(f"Optional: {opt_ok}/{len(OPTIONAL_CHECKS)} installed")

    # Optional database check if MySQL is configured
    try:
        from database.db_manager import DatabaseManager
        db = DatabaseManager()
        if db.conn and db.conn.is_connected():
            cursor = db.conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM movies")
            count = cursor.fetchone()[0]
            cursor.close()
            print(f"✓ Database connected successfully ({count:,} movies)")
        else:
            print("⚠ Database not connected (using fallback/in-memory mode)")
    except Exception as e:
        print(f"⚠ Database check skipped/offline: {e}")

    print("=" * 60 + "\n")
    return 0 if core_ok == len(CORE_CHECKS) else 1


if __name__ == "__main__":
    sys.exit(run_cli_diagnostic())
