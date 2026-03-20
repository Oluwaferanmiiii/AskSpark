"""
Test runner script for AskSpark Agents
Week 2 Lab 1 - Comprehensive test execution
"""

import subprocess
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

def run_tests(test_type="all", verbose=True, coverage=False):
    """
    Run tests for AskSpark Agents
    
    Args:
        test_type: Type of tests to run ("unit", "integration", "performance", "all")
        verbose: Enable verbose output
        coverage: Enable coverage reporting
    """
    test_dir = Path(__file__).parent
    
    # Build pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Add test files based on type
    if test_type == "unit":
        cmd.extend(["test_agents.py", "test_tools.py"])
    elif test_type == "integration":
        cmd.extend(["test_integration.py"])
    elif test_type == "performance":
        cmd.extend(["-m", "performance"])
    elif test_type == "error_handling":
        cmd.extend(["-m", "error_handling"])
    else:  # all
        cmd.append(".")
    
    # Add options
    if verbose:
        cmd.append("-v")
    
    if coverage:
        cmd.extend(["--cov=askspark.agents", "--cov-report=html", "--cov-report=term"])
    
    # Add test directory
    cmd.extend(["--tb=short"])
    
    # Change to test directory
    original_cwd = os.getcwd()
    os.chdir(test_dir)
    
    try:
        print(f"Running {test_type} tests...")
        print(f"Command: {' '.join(cmd)}")
        print("=" * 60)
        
        # Run tests
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Print output
        if result.stdout:
            print(result.stdout)
        
        if result.stderr:
            print("STDERR:", result.stderr)
        
        # Return result
        return result.returncode == 0
        
    finally:
        os.chdir(original_cwd)

def main():
    """Main test runner"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run AskSpark Agent tests")
    parser.add_argument(
        "--type", 
        choices=["unit", "integration", "performance", "error_handling", "all"],
        default="all",
        help="Type of tests to run"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--coverage", "-c",
        action="store_true", 
        help="Enable coverage reporting"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick tests only (skip performance and integration)"
    )
    
    args = parser.parse_args()
    
    # Adjust test type for quick mode
    if args.quick:
        args.type = "unit"
    
    # Run tests
    success = run_tests(
        test_type=args.type,
        verbose=args.verbose,
        coverage=args.coverage
    )
    
    if success:
        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        sys.exit(0)
    else:
        print("\n" + "=" * 60)
        print("✗ Some tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
