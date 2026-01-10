#!/usr/bin/env python3
"""
Quick start script - validates setup and runs a single test example.
Use this to verify everything works before running full batch evaluation.
"""

import os
import sys


def check_environment():
    """Check if environment is properly set up."""
    print("Checking environment...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("✗ Python 3.8+ required")
        return False
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor}")
    
    # Check GPU
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("⚠ No GPU detected - will be slow!")
    except ImportError:
        print("✗ PyTorch not installed")
        return False
    
    # Check dataset
    if not os.path.exists("Dataset"):
        print("✗ Dataset/ directory not found")
        return False
    
    if not os.path.exists("Dataset/Books"):
        print("✗ Dataset/Books/ directory not found")
        return False
    
    csv_files = []
    if os.path.exists("Dataset/train.csv"):
        csv_files.append("train.csv")
    if os.path.exists("Dataset/test.csv"):
        csv_files.append("test.csv")
    
    if not csv_files:
        print("✗ No train.csv or test.csv found in Dataset/")
        return False
    
    print(f"✓ Dataset found: {csv_files}")
    
    return True


def test_imports():
    """Test if all modules can be imported."""
    print("\nTesting imports...")
    
    modules = [
        'config',
        'prompts',
        'models',
        'pathway_store',
        'constraint_engine',
        'pipeline',
        'evaluation_utils',
        'run'
    ]
    
    for module in modules:
        try:
            __import__(module)
            print(f"✓ {module}")
        except ImportError as e:
            print(f"✗ {module}: {e}")
            return False
    
    return True


def run_minimal_test():
    """Run a minimal test without loading full models."""
    print("\nRunning minimal test...")
    
    from evaluation_utils import parse_llm_output, sanity_check
    import config
    
    # Test parsing
    result = parse_llm_output("VIOLATES", "binary")
    assert result == "VIOLATES", "Parsing failed"
    print("✓ LLM output parsing works")
    
    # Test sanity check
    is_valid, _ = sanity_check(
        "Test constraint",
        "Test evidence with sufficient length",
        "VIOLATES"
    )
    assert is_valid, "Sanity check failed"
    print("✓ Sanity checking works")
    
    # Verify config
    assert config.TEMPERATURE == 0.0, "Temperature must be 0.0"
    assert config.QWEN_QUANTIZATION == "4bit", "Must use 4-bit quantization"
    print("✓ Configuration valid")
    
    return True


def main():
    """Main quick start function."""
    print("="*60)
    print("CONSTRAINT CONSISTENCY CHECKER - QUICK START")
    print("="*60)
    
    # Step 1: Check environment
    if not check_environment():
        print("\n✗ Environment check failed!")
        print("\nPlease ensure:")
        print("  1. Python 3.8+ installed")
        print("  2. pip install -r requirements.txt")
        print("  3. Dataset/ with train.csv/test.csv and Books/")
        sys.exit(1)
    
    # Step 2: Test imports
    if not test_imports():
        print("\n✗ Import test failed!")
        print("\nPlease run: pip install -r requirements.txt")
        sys.exit(1)
    
    # Step 3: Run minimal test
    if not run_minimal_test():
        print("\n✗ Minimal test failed!")
        sys.exit(1)
    
    # Success!
    print("\n" + "="*60)
    print("✓ QUICK START SUCCESSFUL!")
    print("="*60)
    print("\nYour system is ready to run!")
    print("\nNext steps:")
    print("  1. Test full system: python test_system.py")
    print("  2. Run full pipeline: python run.py")
    print("\nNote: First run will download models (~10GB)")
    print("Expected runtime: ~1-2 min per example on T4 GPU")


if __name__ == "__main__":
    main()
