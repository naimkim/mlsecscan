"""
Basic example of using MLSecScan to scan a machine learning model.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import pickle

from mlsecscan import ModelScanner, DependencyScanner, RobustnessScanner


def create_sample_model():
    """Create a simple ML model for demonstration."""
    print("Creating sample model...")
    
    # Generate synthetic data
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    
    # Train a simple model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    # Save model
    with open('sample_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print("✓ Sample model created: sample_model.pkl")
    return model, X, y


def example_1_basic_scan():
    """Example 1: Basic model security scan."""
    print("\n" + "="*60)
    print("Example 1: Basic Model Security Scan")
    print("="*60)
    
    # Create a sample model
    create_sample_model()
    
    # Scan the model
    scanner = ModelScanner()
    result = scanner.scan_model('sample_model.pkl', model_type='pickle')
    
    # Print results
    print(result.summary())


def example_2_dependency_scan():
    """Example 2: Scan dependencies for vulnerabilities."""
    print("\n" + "="*60)
    print("Example 2: Dependency Vulnerability Scan")
    print("="*60)
    
    # Create a sample requirements.txt
    requirements_content = """
numpy==1.23.0
scikit-learn==1.2.0
pandas==1.5.0
tensorflow==2.10.0
"""
    
    with open('sample_requirements.txt', 'w') as f:
        f.write(requirements_content)
    
    print("Created sample_requirements.txt")
    
    # Scan dependencies
    dep_scanner = DependencyScanner()
    
    try:
        vulnerabilities = dep_scanner.scan_requirements('sample_requirements.txt')
        report = dep_scanner.generate_report(vulnerabilities)
        print(report)
    except Exception as e:
        print(f"⚠️  Dependency scan requires pip-audit or safety:")
        print(f"   pip install pip-audit")
        print(f"\nError: {e}")


def example_3_adversarial_test():
    """Example 3: Test model robustness against adversarial attacks."""
    print("\n" + "="*60)
    print("Example 3: Adversarial Robustness Testing")
    print("="*60)
    
    # Create model and test data
    model, X, y = create_sample_model()
    
    # Split data for testing
    X_test = X[:200]
    y_test = y[:200]
    
    # Test robustness
    rob_scanner = RobustnessScanner()
    result = rob_scanner.test_model(
        model, 
        X_test, 
        y_test, 
        epsilon=0.1,
        framework='sklearn'
    )
    
    # Print results
    print(result)
    print(f"Robustness Assessment: {rob_scanner.get_robustness_score(result)}")


def example_4_full_pipeline():
    """Example 4: Complete security assessment pipeline."""
    print("\n" + "="*60)
    print("Example 4: Full Security Assessment Pipeline")
    print("="*60)
    
    print("\n1. Creating model...")
    model, X, y = create_sample_model()
    
    print("\n2. Model Security Scan...")
    model_scanner = ModelScanner()
    scan_result = model_scanner.scan_model('sample_model.pkl')
    
    critical_count = len(scan_result.get_critical_findings())
    high_count = len(scan_result.get_high_findings())
    
    print(f"   Findings: {len(scan_result.findings)} total")
    print(f"   - CRITICAL: {critical_count}")
    print(f"   - HIGH: {high_count}")
    
    print("\n3. Adversarial Robustness Test...")
    rob_scanner = RobustnessScanner()
    rob_result = rob_scanner.test_model(model, X[:200], y[:200], epsilon=0.1)
    print(f"   Attack Success Rate: {rob_result.attack_success_rate:.2%}")
    print(f"   Assessment: {rob_scanner.get_robustness_score(rob_result)}")
    
    print("\n4. Final Security Report...")
    print("="*60)
    
    if critical_count > 0:
        print("⚠️  CRITICAL SECURITY ISSUES FOUND")
        print("   Immediate action required!")
    elif high_count > 0:
        print("⚠️  HIGH SEVERITY ISSUES FOUND")
        print("   Action recommended")
    elif rob_result.attack_success_rate > 0.5:
        print("⚠️  MODEL IS VULNERABLE TO ADVERSARIAL ATTACKS")
        print("   Consider adversarial training")
    else:
        print("✓ Model passed basic security checks")
        print("   Continue with additional testing")
    
    print("="*60)


if __name__ == '__main__':
    print("""
MLSecScan - Example Usage
=========================

This script demonstrates the key features of MLSecScan:
1. Basic model security scanning
2. Dependency vulnerability checking
3. Adversarial robustness testing
4. Full security assessment pipeline
""")
    
    # Run examples
    example_1_basic_scan()
    example_2_dependency_scan()
    example_3_adversarial_test()
    example_4_full_pipeline()
    
    print("\n✓ All examples completed!")
    print("\nNext steps:")
    print("  - Try scanning your own models")
    print("  - Integrate into your CI/CD pipeline")
    print("  - Customize security checks for your use case")
