#!/usr/bin/env python3
"""
Comprehensive test runner for AskSpark application
"""

import sys
import os
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestRunner:
    """Comprehensive test runner for AskSpark"""
    
    def __init__(self):
        self.test_results = {}
        self.total_start_time = time.time()
    
    def run_unit_tests(self) -> Dict[str, Any]:
        """Run all unit tests"""
        print("Running Unit Tests...")
        print("=" * 50)
        
        start_time = time.time()
        
        try:
            # Run unit tests
            result = subprocess.run([
                sys.executable, "-m", "pytest", 
                "tests/unit/", 
                "-v", 
                "--tb=short",
                "--color=yes"
            ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
            
            end_time = time.time()
            
            success = result.returncode == 0
            
            return {
                "success": success,
                "duration": end_time - start_time,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
            
        except Exception as e:
            return {
                "success": False,
                "duration": 0,
                "stdout": "",
                "stderr": str(e),
                "returncode": -1
            }
    
    def run_integration_tests(self) -> Dict[str, Any]:
        """Run all integration tests"""
        print("\nRunning Integration Tests...")
        print("=" * 50)
        
        start_time = time.time()
        
        try:
            # Run integration tests
            result = subprocess.run([
                sys.executable, "-m", "pytest", 
                "tests/integration/", 
                "-v", 
                "--tb=short",
                "--color=yes"
            ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
            
            end_time = time.time()
            
            success = result.returncode == 0
            
            return {
                "success": success,
                "duration": end_time - start_time,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
            
        except Exception as e:
            return {
                "success": False,
                "duration": 0,
                "stdout": "",
                "stderr": str(e),
                "returncode": -1
            }
    
    def run_performance_tests(self) -> Dict[str, Any]:
        """Run performance and stress tests"""
        print("\nRunning Performance Tests...")
        print("=" * 50)
        
        start_time = time.time()
        
        try:
            # Run performance tests
            result = subprocess.run([
                sys.executable, "-m", "pytest", 
                "tests/performance/", 
                "-v", 
                "--tb=short",
                "--color=yes"
            ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
            
            end_time = time.time()
            
            success = result.returncode == 0
            
            return {
                "success": success,
                "duration": end_time - start_time,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
            
        except Exception as e:
            return {
                "success": False,
                "duration": 0,
                "stdout": "",
                "stderr": str(e),
                "returncode": -1
            }
    
    def run_coverage_analysis(self) -> Dict[str, Any]:
        """Run test coverage analysis"""
        print("\nRunning Coverage Analysis...")
        print("=" * 50)
        
        start_time = time.time()
        
        try:
            # Run coverage analysis
            result = subprocess.run([
                sys.executable, "-m", "pytest", 
                "tests/", 
                "--cov=src/askspark",
                "--cov-report=term-missing",
                "--cov-report=html",
                "--cov-report=xml",
                "-v"
            ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
            
            end_time = time.time()
            
            success = result.returncode == 0
            
            return {
                "success": success,
                "duration": end_time - start_time,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
            
        except Exception as e:
            return {
                "success": False,
                "duration": 0,
                "stdout": "",
                "stderr": str(e),
                "returncode": -1
            }
    
    def run_data_flow_validation(self) -> Dict[str, Any]:
        """Run specific data flow validation tests"""
        print("\nRunning Data Flow Validation...")
        print("=" * 50)
        
        start_time = time.time()
        
        try:
            # Run specific data flow tests
            result = subprocess.run([
                sys.executable, "-m", "pytest", 
                "tests/integration/test_data_flow.py", 
                "-v", 
                "--tb=short",
                "--color=yes"
            ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
            
            end_time = time.time()
            
            success = result.returncode == 0
            
            return {
                "success": success,
                "duration": end_time - start_time,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
            
        except Exception as e:
            return {
                "success": False,
                "duration": 0,
                "stdout": "",
                "stderr": str(e),
                "returncode": -1
            }
    
    def check_dependencies(self) -> Dict[str, Any]:
        """Check if all test dependencies are available"""
        print("\nChecking Test Dependencies...")
        print("=" * 50)
        
        required_packages = [
            "pytest",
            "pytest_cov", 
            "psutil",
            "pandas",
            "plotly"
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
                print(f"PASS {package}")
            except ImportError:
                missing_packages.append(package)
                print(f"FAIL {package}")
        
        success = len(missing_packages) == 0
        
        return {
            "success": success,
            "missing_packages": missing_packages,
            "total_packages": len(required_packages),
            "available_packages": len(required_packages) - len(missing_packages)
        }
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all test suites"""
        print("\nStarting Comprehensive Test Suite")
        print("=" * 60)
        
        # Check dependencies first
        dep_check = self.check_dependencies()
        if not dep_check["success"]:
            print(f"\nMissing dependencies: {', '.join(dep_check['missing_packages'])}")
            print("Install with: pip install pytest pytest-cov psutil pandas plotly")
            return {"overall_success": False, "dependency_check": dep_check}
        
        # Run all test suites
        results = {
            "dependency_check": dep_check,
        }
        
        if dep_check["success"]:
            results.update({
                "unit_tests": self.run_unit_tests(),
                "integration_tests": self.run_integration_tests(),
                "performance_tests": self.run_performance_tests(),
                "data_flow_validation": self.run_data_flow_validation(),
                "coverage_analysis": self.run_coverage_analysis()
            })
        else:
            # Create placeholder results for failed dependency check
            results.update({
                "unit_tests": {"success": False, "duration": 0, "stdout": "", "stderr": "Dependencies missing", "returncode": -1},
                "integration_tests": {"success": False, "duration": 0, "stdout": "", "stderr": "Dependencies missing", "returncode": -1},
                "performance_tests": {"success": False, "duration": 0, "stdout": "", "stderr": "Dependencies missing", "returncode": -1},
                "data_flow_validation": {"success": False, "duration": 0, "stdout": "", "stderr": "Dependencies missing", "returncode": -1},
                "coverage_analysis": {"success": False, "duration": 0, "stdout": "", "stderr": "Dependencies missing", "returncode": -1}
            })
        
        # Calculate overall success
        test_suites = ["unit_tests", "integration_tests", "performance_tests", "data_flow_validation"]
        overall_success = all(results[suite]["success"] for suite in test_suites)
        
        results["overall_success"] = overall_success
        results["total_duration"] = time.time() - self.total_start_time
        
        return results
    
    def print_summary(self, results: Dict[str, Any]):
        """Print test summary"""
        print("\n" + "=" * 60)
        print("\nTEST SUITE SUMMARY")
        print("=" * 60)
        
        # Dependency check
        dep_check = results["dependency_check"]
        print(f"Dependencies: {dep_check['available_packages']}/{dep_check['total_packages']} available")
        
        # Test results
        test_suites = {
            "unit_tests": "Unit Tests",
            "integration_tests": "Integration Tests", 
            "performance_tests": "Performance Tests",
            "data_flow_validation": "Data Flow Validation"
        }
        
        for suite_key, suite_name in test_suites.items():
            suite_result = results[suite_key]
            status = "PASS" if suite_result["success"] else "FAIL"
            duration = suite_result.get("duration", 0)
            print(f"{suite_name:.<30} {status} ({duration:.2f}s)")
        
        # Coverage
        if results["coverage_analysis"]["success"]:
            print(f"Coverage Analysis: COMPLETED")
        else:
            print(f"Coverage Analysis: FAILED")
        
        # Overall
        print("\n" + "-" * 60)
        overall_status = "ALL TESTS PASSED" if results["overall_success"] else "SOME TESTS FAILED"
        print(f"Overall Result: {overall_status}")
        print(f"Total Duration: {results['total_duration']:.2f}s")
        
        # Failed details
        if not results["overall_success"]:
            print("\nFailed Test Suites:")
            for suite_key, suite_name in test_suites.items():
                if not results[suite_key]["success"]:
                    print(f"  - {suite_name}")
                    if results[suite_key]["stderr"]:
                        print(f"    Error: {results[suite_key]['stderr'][:200]}...")
        
        print("\n" + "=" * 60)
    
    def generate_test_report(self, results: Dict[str, Any]) -> str:
        """Generate detailed test report"""
        report = []
        report.append("# AskSpark Test Report")
        report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Summary
        report.append("## Summary")
        overall_status = "PASSED" if results["overall_success"] else "FAILED"
        report.append(f"- Overall Status: {overall_status}")
        report.append(f"- Total Duration: {results['total_duration']:.2f}s")
        report.append("")
        
        # Test Suites
        report.append("## Test Suite Results")
        
        test_suites = {
            "unit_tests": "Unit Tests",
            "integration_tests": "Integration Tests",
            "performance_tests": "Performance Tests", 
            "data_flow_validation": "Data Flow Validation"
        }
        
        for suite_key, suite_name in test_suites.items():
            suite_result = results[suite_key]
            status = "PASS" if suite_result["success"] else "FAIL"
            report.append(f"### {suite_name}")
            report.append(f"- Status: {status}")
            report.append(f"- Duration: {suite_result.get('duration', 0):.2f}s")
            if suite_result["stderr"]:
                report.append(f"- Error: {suite_result['stderr']}")
            report.append("")
        
        # Coverage
        if results["coverage_analysis"]["success"]:
            report.append("## Coverage Analysis")
            report.append(results["coverage_analysis"]["stdout"])
            report.append("")
        
        # Recommendations
        report.append("## Recommendations")
        
        if not results["overall_success"]:
            report.append("- Fix failing test suites before deployment")
            report.append("- Review error messages for troubleshooting")
        else:
            report.append("- All tests passed! Ready for deployment")
            report.append("- Consider adding more edge case tests")
        
        report.append("- Monitor performance metrics in production")
        report.append("- Regularly run data flow validation")
        
        return "\n".join(report)


def main():
    """Main test runner function"""
    runner = TestRunner()
    
    # Run all tests
    results = runner.run_all_tests()
    
    # Print summary
    runner.print_summary(results)
    
    # Generate report
    report = runner.generate_test_report(results)
    
    # Save report
    report_path = Path(__file__).parent.parent / "test_report.md"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Detailed report saved to: {report_path}")
    
    # Exit with appropriate code
    sys.exit(0 if results["overall_success"] else 1)


if __name__ == "__main__":
    main()
