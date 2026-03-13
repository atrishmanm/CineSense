"""
Test if all dependencies are available
Run this before starting the full application
"""

import sys

print("\n" + "=" * 60)
print("CineSense Dependency Check")
print("=" * 60 + "\n")

# Core dependencies
checks = [
    ("Flask", "flask"),
    ("MySQL Connector", "mysql.connector"),
    ("PyTorch", "torch"),
    ("Transformers", "transformers"),
    ("Sentence Transformers", "sentence_transformers"),
    ("FAISS", "faiss"),
    ("NumPy", "numpy"),
    ("scikit-learn", "sklearn"),
    ("Pillow (PIL)", "PIL"),
    ("Requests", "requests"),
]

# Optional dependencies
optional_checks = [
    ("Redis", "redis"),
    ("SHAP", "shap"),
]

passed = 0
failed = 0
optional_failed = 0

print("Core Dependencies:")
print("-" * 60)
for name, module in checks:
    try:
        __import__(module)
        print(f"✓ {name:<30} INSTALLED")
        passed += 1
    except ImportError:
        print(f"✗ {name:<30} MISSING")
        failed += 1

print("\nOptional Dependencies:")
print("-" * 60)
for name, module in optional_checks:
    try:
        __import__(module)
        print(f"✓ {name:<30} INSTALLED")
    except ImportError:
        print(f"⚠ {name:<30} NOT INSTALLED (optional)")
        optional_failed += 1

print("\n" + "=" * 60)
print(f"Core: {passed}/{len(checks)} passed")
if failed > 0:
    print(f"⚠ {failed} core dependencies missing")
    print("Run: pip install -r requirements.txt")
else:
    print("✅ All core dependencies installed!")

if optional_failed > 0:
    print(f"\n{optional_failed} optional dependencies not installed")
    print("These are not required but provide additional features:")
    print("  • Redis: For caching (96% faster)")
    print("  • SHAP: For explainable AI")
print("=" * 60 + "\n")

# Test database connection
print("Testing Database Connection:")
print("-" * 60)
try:
    from database.db_manager import DatabaseManager
    db = DatabaseManager()
    cursor = db.conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM movies")
    count = cursor.fetchone()[0]
    cursor.close()
    print(f"✓ Database connected successfully")
    print(f"  Found {count:,} movies in database")
except Exception as e:
    print(f"✗ Database connection failed: {e}")
    print("  Check your .env file configuration")

print("\n" + "=" * 60)

if failed == 0:
    print("✅ System Ready! You can start the application.")
    print("\nRun: python app_integrated.py")
else:
    print("⚠ Some dependencies are missing. Install them first.")
    print("\nRun: pip install -r requirements.txt")
    sys.exit(1)

print("=" * 60 + "\n")
