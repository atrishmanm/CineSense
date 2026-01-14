"""Test script to start the app with error handling"""
import sys
import traceback

try:
    print("Importing app module...")
    import app
    print("App module imported successfully")
    
    print("Starting application...")
    app.main()
except Exception as e:
    print(f"\n❌ Error occurred: {e}")
    print("\nFull traceback:")
    traceback.print_exc()
    sys.exit(1)
