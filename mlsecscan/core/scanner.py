"""
Core scanner module for ML model security scanning.
"""

import os
import pickle
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SeverityLevel(Enum):
    """Security issue severity levels."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


@dataclass
class SecurityFinding:
    """Represents a security finding from a scan."""
    title: str
    severity: SeverityLevel
    description: str
    category: str
    remediation: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self):
        return f"[{self.severity.value}] {self.title}: {self.description}"


@dataclass
class ScanResult:
    """Results from a security scan."""
    model_path: str
    scan_time: datetime
    findings: List[SecurityFinding] = field(default_factory=list)
    passed_checks: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_finding(self, finding: SecurityFinding):
        """Add a security finding."""
        self.findings.append(finding)
        
    def add_passed_check(self, check_name: str):
        """Add a passed security check."""
        self.passed_checks.append(check_name)
    
    def get_critical_findings(self) -> List[SecurityFinding]:
        """Get all critical severity findings."""
        return [f for f in self.findings if f.severity == SeverityLevel.CRITICAL]
    
    def get_high_findings(self) -> List[SecurityFinding]:
        """Get all high severity findings."""
        return [f for f in self.findings if f.severity == SeverityLevel.HIGH]
    
    def summary(self) -> str:
        """Generate a summary of scan results."""
        critical_count = len(self.get_critical_findings())
        high_count = len(self.get_high_findings())
        medium_count = len([f for f in self.findings if f.severity == SeverityLevel.MEDIUM])
        low_count = len([f for f in self.findings if f.severity == SeverityLevel.LOW])
        
        summary = f"""
MLSecScan Report
================
Model: {self.model_path}
Scan Date: {self.scan_time.strftime('%Y-%m-%d %H:%M:%S')}

Security Findings: {len(self.findings)} total
  - CRITICAL: {critical_count}
  - HIGH: {high_count}
  - MEDIUM: {medium_count}
  - LOW: {low_count}

Passed Checks: {len(self.passed_checks)}
"""
        
        if self.findings:
            summary += "\n⚠️  SECURITY FINDINGS:\n"
            for finding in sorted(self.findings, key=lambda x: x.severity.value):
                summary += f"  {finding}\n"
        
        if self.passed_checks:
            summary += "\n✓ PASSED CHECKS:\n"
            for check in self.passed_checks:
                summary += f"  [✓] {check}\n"
        
        return summary


class ModelScanner:
    """
    Main scanner class for ML model security analysis.
    
    Performs comprehensive security scanning of ML models including:
    - Pickle deserialization safety
    - Malicious code detection
    - Model integrity checks
    """
    
    def __init__(self):
        self.logger = logger
        
    def scan_model(self, model_path: str, model_type: str = 'auto') -> ScanResult:
        """
        Scan a model file for security vulnerabilities.
        
        Args:
            model_path: Path to the model file
            model_type: Type of model (pickle, h5, onnx, pytorch, auto)
            
        Returns:
            ScanResult object with findings
        """
        self.logger.info(f"Starting security scan for: {model_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        result = ScanResult(
            model_path=model_path,
            scan_time=datetime.now()
        )
        
        # Detect model type if auto
        if model_type == 'auto':
            model_type = self._detect_model_type(model_path)
            result.metadata['detected_type'] = model_type
        
        # Run scans based on model type
        if model_type == 'pickle':
            self._scan_pickle_model(model_path, result)
        elif model_type == 'h5':
            self._scan_h5_model(model_path, result)
        elif model_type == 'onnx':
            self._scan_onnx_model(model_path, result)
        elif model_type == 'pytorch':
            self._scan_pytorch_model(model_path, result)
        else:
            result.add_finding(SecurityFinding(
                title="Unknown model type",
                severity=SeverityLevel.MEDIUM,
                description=f"Could not determine model type for {model_path}",
                category="model_format"
            ))
        
        # Run common security checks
        self._check_file_permissions(model_path, result)
        self._check_file_size(model_path, result)
        
        self.logger.info(f"Scan complete. Found {len(result.findings)} issues.")
        return result
    
    def _detect_model_type(self, model_path: str) -> str:
        """Detect the type of ML model from file extension."""
        ext = os.path.splitext(model_path)[1].lower()
        
        type_map = {
            '.pkl': 'pickle',
            '.pickle': 'pickle',
            '.h5': 'h5',
            '.hdf5': 'h5',
            '.onnx': 'onnx',
            '.pt': 'pytorch',
            '.pth': 'pytorch',
        }
        
        return type_map.get(ext, 'unknown')
    
    def _scan_pickle_model(self, model_path: str, result: ScanResult):
        """Scan pickle model for security issues."""
        try:
            # Check for dangerous pickle operations
            with open(model_path, 'rb') as f:
                content = f.read()
                
                # Look for dangerous patterns
                dangerous_patterns = [
                    b'__reduce__',
                    b'__setstate__',
                    b'eval',
                    b'exec',
                    b'compile',
                    b'os.system',
                    b'subprocess',
                ]
                
                for pattern in dangerous_patterns:
                    if pattern in content:
                        result.add_finding(SecurityFinding(
                            title="Dangerous pickle pattern detected",
                            severity=SeverityLevel.HIGH,
                            description=f"Found potentially dangerous pattern: {pattern.decode('utf-8', errors='ignore')}",
                            category="code_injection",
                            remediation="Avoid using pickle for untrusted models. Consider using safer formats like ONNX or SavedModel."
                        ))
            
            # Try to load with restricted unpickler
            try:
                with open(model_path, 'rb') as f:
                    # This is a basic check - in production, use a restricted unpickler
                    pickle.load(f)
                result.add_passed_check("Pickle file loads without errors")
            except Exception as e:
                result.add_finding(SecurityFinding(
                    title="Pickle loading failed",
                    severity=SeverityLevel.MEDIUM,
                    description=f"Failed to load pickle file: {str(e)}",
                    category="model_integrity"
                ))
                
        except Exception as e:
            result.add_finding(SecurityFinding(
                title="Pickle scan error",
                severity=SeverityLevel.LOW,
                description=f"Error during pickle scan: {str(e)}",
                category="scan_error"
            ))
    
    def _scan_h5_model(self, model_path: str, result: ScanResult):
        """Scan HDF5/Keras model."""
        try:
            import h5py
            with h5py.File(model_path, 'r') as f:
                # Check for custom layers or lambdas
                if 'model_config' in f.attrs:
                    config_str = f.attrs['model_config']
                    if b'lambda' in config_str.lower():
                        result.add_finding(SecurityFinding(
                            title="Lambda layer detected",
                            severity=SeverityLevel.MEDIUM,
                            description="Model contains Lambda layers which can execute arbitrary code",
                            category="code_injection",
                            remediation="Review Lambda layer implementations for security risks"
                        ))
                
                result.add_passed_check("H5 file structure is valid")
        except ImportError:
            result.add_finding(SecurityFinding(
                title="H5PY not available",
                severity=SeverityLevel.INFO,
                description="h5py library not installed, skipping H5 checks",
                category="scan_limitation"
            ))
        except Exception as e:
            result.add_finding(SecurityFinding(
                title="H5 scan error",
                severity=SeverityLevel.LOW,
                description=f"Error during H5 scan: {str(e)}",
                category="scan_error"
            ))
    
    def _scan_onnx_model(self, model_path: str, result: ScanResult):
        """Scan ONNX model."""
        try:
            import onnx
            model = onnx.load(model_path)
            onnx.checker.check_model(model)
            result.add_passed_check("ONNX model is valid")
            
            # Check for custom operators
            custom_ops = set()
            for node in model.graph.node:
                if node.domain and node.domain != "":
                    custom_ops.add(f"{node.domain}.{node.op_type}")
            
            if custom_ops:
                result.add_finding(SecurityFinding(
                    title="Custom ONNX operators detected",
                    severity=SeverityLevel.LOW,
                    description=f"Found custom operators: {', '.join(custom_ops)}",
                    category="model_complexity",
                    remediation="Review custom operators for security implications"
                ))
                
        except ImportError:
            result.add_finding(SecurityFinding(
                title="ONNX not available",
                severity=SeverityLevel.INFO,
                description="onnx library not installed, skipping ONNX checks",
                category="scan_limitation"
            ))
        except Exception as e:
            result.add_finding(SecurityFinding(
                title="ONNX scan error",
                severity=SeverityLevel.LOW,
                description=f"Error during ONNX scan: {str(e)}",
                category="scan_error"
            ))
    
    def _scan_pytorch_model(self, model_path: str, result: ScanResult):
        """Scan PyTorch model."""
        try:
            import torch
            # Use weights_only=True for safer loading (PyTorch 1.13+)
            try:
                model = torch.load(model_path, weights_only=True)
                result.add_passed_check("PyTorch model loads safely with weights_only=True")
            except Exception:
                # Try regular load
                model = torch.load(model_path)
                result.add_finding(SecurityFinding(
                    title="PyTorch model requires unsafe loading",
                    severity=SeverityLevel.HIGH,
                    description="Model cannot be loaded with weights_only=True, may contain arbitrary code",
                    category="code_injection",
                    remediation="Re-save model with only weights, or use ONNX format"
                ))
                
        except ImportError:
            result.add_finding(SecurityFinding(
                title="PyTorch not available",
                severity=SeverityLevel.INFO,
                description="torch library not installed, skipping PyTorch checks",
                category="scan_limitation"
            ))
        except Exception as e:
            result.add_finding(SecurityFinding(
                title="PyTorch scan error",
                severity=SeverityLevel.LOW,
                description=f"Error during PyTorch scan: {str(e)}",
                category="scan_error"
            ))
    
    def _check_file_permissions(self, model_path: str, result: ScanResult):
        """Check file permissions for security issues."""
        try:
            stat_info = os.stat(model_path)
            mode = stat_info.st_mode
            
            # Check if file is world-writable (dangerous)
            if mode & 0o002:
                result.add_finding(SecurityFinding(
                    title="World-writable model file",
                    severity=SeverityLevel.MEDIUM,
                    description="Model file has world-writable permissions",
                    category="file_permissions",
                    remediation="Change file permissions to be more restrictive (chmod 644)"
                ))
            else:
                result.add_passed_check("File permissions are secure")
                
        except Exception as e:
            self.logger.debug(f"Could not check file permissions: {e}")
    
    def _check_file_size(self, model_path: str, result: ScanResult):
        """Check if file size is suspicious."""
        try:
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            result.metadata['file_size_mb'] = round(size_mb, 2)
            
            # Warn if file is unusually large (>1GB)
            if size_mb > 1024:
                result.add_finding(SecurityFinding(
                    title="Large model file",
                    severity=SeverityLevel.LOW,
                    description=f"Model file is {size_mb:.2f} MB, which is unusually large",
                    category="model_size",
                    remediation="Verify this is expected. Large files may contain embedded data or malicious content."
                ))
                
        except Exception as e:
            self.logger.debug(f"Could not check file size: {e}")
