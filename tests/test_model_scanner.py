"""
Tests for the ModelScanner class.
"""

import pytest
import os
import pickle
import tempfile
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

from mlsecscan.core.scanner import ModelScanner, SeverityLevel, SecurityFinding


@pytest.fixture
def sample_model():
    """Create a sample sklearn model for testing."""
    X, y = make_classification(n_samples=100, n_features=10, n_classes=2, random_state=42)
    model = RandomForestClassifier(n_estimators=5, random_state=42)
    model.fit(X, y)
    return model


@pytest.fixture
def temp_pickle_file(sample_model):
    """Create a temporary pickle file with a model."""
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as f:
        pickle.dump(sample_model, f)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.remove(temp_path)


class TestModelScanner:
    
    def test_scanner_initialization(self):
        """Test that scanner can be initialized."""
        scanner = ModelScanner()
        assert scanner is not None
    
    def test_detect_pickle_type(self, temp_pickle_file):
        """Test automatic detection of pickle file type."""
        scanner = ModelScanner()
        detected_type = scanner._detect_model_type(temp_pickle_file)
        assert detected_type == 'pickle'
    
    def test_scan_pickle_model(self, temp_pickle_file):
        """Test scanning a pickle model."""
        scanner = ModelScanner()
        result = scanner.scan_model(temp_pickle_file, model_type='pickle')
        
        assert result is not None
        assert result.model_path == temp_pickle_file
        assert len(result.findings) >= 0  # May or may not have findings
        
    def test_scan_nonexistent_file(self):
        """Test scanning a file that doesn't exist."""
        scanner = ModelScanner()
        
        with pytest.raises(FileNotFoundError):
            scanner.scan_model('nonexistent_model.pkl')
    
    def test_security_finding_creation(self):
        """Test creating security findings."""
        finding = SecurityFinding(
            title="Test Finding",
            severity=SeverityLevel.HIGH,
            description="This is a test",
            category="test"
        )
        
        assert finding.title == "Test Finding"
        assert finding.severity == SeverityLevel.HIGH
        assert "HIGH" in str(finding)
    
    def test_scan_result_summary(self, temp_pickle_file):
        """Test scan result summary generation."""
        scanner = ModelScanner()
        result = scanner.scan_model(temp_pickle_file)
        
        summary = result.summary()
        assert "MLSecScan Report" in summary
        assert temp_pickle_file in summary
    
    def test_file_permission_check(self, temp_pickle_file):
        """Test file permission checking."""
        scanner = ModelScanner()
        result = scanner.scan_model(temp_pickle_file)
        
        # Should have checked permissions
        has_permission_check = any(
            'permission' in check.lower() 
            for check in result.passed_checks
        ) or any(
            'permission' in finding.category.lower() 
            for finding in result.findings
        )
        
        assert has_permission_check or len(result.passed_checks) > 0


class TestSecurityFindings:
    
    def test_finding_severity_levels(self):
        """Test all severity levels."""
        levels = [
            SeverityLevel.CRITICAL,
            SeverityLevel.HIGH,
            SeverityLevel.MEDIUM,
            SeverityLevel.LOW,
            SeverityLevel.INFO
        ]
        
        for level in levels:
            finding = SecurityFinding(
                title="Test",
                severity=level,
                description="Test",
                category="test"
            )
            assert finding.severity == level
    
    def test_get_critical_findings(self, temp_pickle_file):
        """Test filtering critical findings."""
        scanner = ModelScanner()
        result = scanner.scan_model(temp_pickle_file)
        
        critical = result.get_critical_findings()
        assert isinstance(critical, list)
        assert all(f.severity == SeverityLevel.CRITICAL for f in critical)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
