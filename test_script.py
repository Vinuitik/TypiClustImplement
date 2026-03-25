import sys
from pathlib import Path
import traceback

def diagnostic():
    print("Checking Python environment...")
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    
    print("\nChecking AL Methods imports...")
    try:
        import al_methods
        print("al_methods directory imported successfully.")
        print(f"Available attributes in al_methods: {dir(al_methods)}")
        
        from al_methods import STRATEGY_REGISTRY
        print(f"STRATEGY_REGISTRY imported successfully: {list(STRATEGY_REGISTRY.keys())}")
        
    except Exception as e:
        print("\nERROR testing imports:")
        traceback.print_exc()

if __name__ == "__main__":
    diagnostic()
