"""
Dependency vulnerability scanner for ML projects.
"""

import os
import subprocess
import json
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Vulnerability:
    """Represents a dependency vulnerability."""
    package: str
    installed_version: str
    vulnerability_id: str
    severity: str
    description: str
    fixed_version: Optional[str] = None
    
    def __str__(self):
        fix_info = f" (fix: {self.fixed_version})" if self.fixed_version else ""
        return f"[{self.severity}] {self.package}=={self.installed_version}: {self.vulnerability_id}{fix_info}"


class DependencyScanner:
    """
    Scanner for detecting vulnerabilities in Python dependencies.
    
    Uses pip-audit and safety to check for known CVEs in requirements.txt
    """
    
    def __init__(self):
        self.logger = logger
        
    def scan_requirements(self, requirements_path: str) -> List[Vulnerability]:
        """
        Scan a requirements.txt file for vulnerabilities.
        
        Args:
            requirements_path: Path to requirements.txt file
            
        Returns:
            List of Vulnerability objects
        """
        if not os.path.exists(requirements_path):
            raise FileNotFoundError(f"Requirements file not found: {requirements_path}")
        
        vulnerabilities = []
        
        # Try pip-audit first (more comprehensive)
        try:
            vulns = self._scan_with_pip_audit(requirements_path)
            vulnerabilities.extend(vulns)
            self.logger.info(f"pip-audit found {len(vulns)} vulnerabilities")
        except Exception as e:
            self.logger.warning(f"pip-audit scan failed: {e}")
        
        # Fallback to safety
        if not vulnerabilities:
            try:
                vulns = self._scan_with_safety(requirements_path)
                vulnerabilities.extend(vulns)
                self.logger.info(f"safety found {len(vulns)} vulnerabilities")
            except Exception as e:
                self.logger.warning(f"safety scan failed: {e}")
        
        return vulnerabilities
    
    def _scan_with_pip_audit(self, requirements_path: str) -> List[Vulnerability]:
        """Scan using pip-audit."""
        try:
            result = subprocess.run(
                ['pip-audit', '-r', requirements_path, '--format', 'json'],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                # No vulnerabilities found
                return []
            
            # Parse JSON output
            try:
                data = json.loads(result.stdout)
                vulnerabilities = []
                
                for vuln_data in data.get('dependencies', []):
                    package = vuln_data.get('name', 'unknown')
                    version = vuln_data.get('version', 'unknown')
                    
                    for vuln in vuln_data.get('vulns', []):
                        vulnerabilities.append(Vulnerability(
                            package=package,
                            installed_version=version,
                            vulnerability_id=vuln.get('id', 'N/A'),
                            severity=vuln.get('severity', 'UNKNOWN'),
                            description=vuln.get('description', 'No description'),
                            fixed_version=vuln.get('fix_versions', [None])[0] if vuln.get('fix_versions') else None
                        ))
                
                return vulnerabilities
            except json.JSONDecodeError:
                self.logger.error("Failed to parse pip-audit JSON output")
                return []
                
        except FileNotFoundError:
            raise Exception("pip-audit not installed. Install with: pip install pip-audit")
        except subprocess.TimeoutExpired:
            raise Exception("pip-audit scan timed out")
    
    def _scan_with_safety(self, requirements_path: str) -> List[Vulnerability]:
        """Scan using safety (free tier)."""
        try:
            result = subprocess.run(
                ['safety', 'check', '--file', requirements_path, '--json'],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            # Parse output
            try:
                data = json.loads(result.stdout)
                vulnerabilities = []
                
                for vuln_data in data:
                    vulnerabilities.append(Vulnerability(
                        package=vuln_data[0],
                        installed_version=vuln_data[2],
                        vulnerability_id=vuln_data[3],
                        severity=vuln_data[4] if len(vuln_data) > 4 else 'UNKNOWN',
                        description=vuln_data[1],
                        fixed_version=None
                    ))
                
                return vulnerabilities
            except (json.JSONDecodeError, IndexError):
                self.logger.error("Failed to parse safety output")
                return []
                
        except FileNotFoundError:
            raise Exception("safety not installed. Install with: pip install safety")
        except subprocess.TimeoutExpired:
            raise Exception("safety scan timed out")
    
    def get_ml_specific_recommendations(self, vulnerabilities: List[Vulnerability]) -> Dict[str, str]:
        """
        Get ML-specific security recommendations for vulnerable packages.
        
        Args:
            vulnerabilities: List of vulnerabilities found
            
        Returns:
            Dict mapping package names to recommendations
        """
        recommendations = {}
        
        ml_packages = {
            'tensorflow': 'Critical for ML security. Update immediately as TensorFlow vulns can lead to RCE.',
            'torch': 'PyTorch vulnerabilities can allow arbitrary code execution during model loading.',
            'scikit-learn': 'Vulnerabilities may affect model serialization and deserialization.',
            'numpy': 'Core dependency - vulnerabilities can affect all ML operations.',
            'pillow': 'Image processing vulns can be exploited via malicious training data.',
            'opencv-python': 'Computer vision vulnerabilities often allow RCE via image inputs.',
        }
        
        for vuln in vulnerabilities:
            pkg = vuln.package.lower()
            for ml_pkg, recommendation in ml_packages.items():
                if ml_pkg in pkg:
                    recommendations[vuln.package] = recommendation
        
        return recommendations
    
    def generate_report(self, vulnerabilities: List[Vulnerability]) -> str:
        """Generate a human-readable report of vulnerabilities."""
        if not vulnerabilities:
            return "✓ No known vulnerabilities found in dependencies!"
        
        report = f"""
Dependency Vulnerability Report
================================
Total Vulnerabilities: {len(vulnerabilities)}

"""
        # Group by severity
        by_severity = {}
        for vuln in vulnerabilities:
            severity = vuln.severity.upper()
            if severity not in by_severity:
                by_severity[severity] = []
            by_severity[severity].append(vuln)
        
        # Report by severity
        for severity in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'UNKNOWN']:
            if severity in by_severity:
                report += f"\n{severity} Severity ({len(by_severity[severity])}):\n"
                report += "-" * 50 + "\n"
                for vuln in by_severity[severity]:
                    report += f"  {vuln}\n"
                    report += f"    {vuln.description[:100]}...\n"
        
        # ML-specific recommendations
        recommendations = self.get_ml_specific_recommendations(vulnerabilities)
        if recommendations:
            report += "\n\nML Security Recommendations:\n"
            report += "=" * 50 + "\n"
            for package, rec in recommendations.items():
                report += f"\n{package}:\n  {rec}\n"
        
        return report
