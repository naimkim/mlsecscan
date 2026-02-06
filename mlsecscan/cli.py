"""
Command-line interface for MLSecScan.
"""

import click
import os
from mlsecscan.core.scanner import ModelScanner
from mlsecscan.scanners.dependency_scanner import DependencyScanner


@click.group()
@click.version_option(version='0.1.0')
def main():
    """MLSecScan - Machine Learning Model Security Scanner"""
    pass


@main.command()
@click.option('--model', '-m', required=True, help='Path to model file')
@click.option('--type', '-t', default='auto', help='Model type (auto, pickle, h5, onnx, pytorch)')
@click.option('--full-report', '-f', is_flag=True, help='Generate full detailed report')
def scan(model, type, full_report):
    """Scan a model file for security vulnerabilities."""
    click.echo(f"🔍 Scanning model: {model}")
    
    scanner = ModelScanner()
    try:
        result = scanner.scan_model(model, model_type=type)
        
        if full_report:
            click.echo(result.summary())
        else:
            # Brief summary
            click.echo(f"\n✓ Scan complete!")
            click.echo(f"  Findings: {len(result.findings)}")
            click.echo(f"  Passed checks: {len(result.passed_checks)}")
            
            if result.get_critical_findings():
                click.echo("\n⚠️  CRITICAL issues found!")
                for finding in result.get_critical_findings():
                    click.echo(f"  - {finding.title}")
            elif result.get_high_findings():
                click.echo("\n⚠️  HIGH severity issues found!")
                for finding in result.get_high_findings():
                    click.echo(f"  - {finding.title}")
            else:
                click.echo("\n✓ No critical security issues detected")
                
    except FileNotFoundError as e:
        click.echo(f"❌ Error: {e}", err=True)
        return 1
    except Exception as e:
        click.echo(f"❌ Scan failed: {e}", err=True)
        return 1


@main.command()
@click.option('--requirements', '-r', default='requirements.txt', help='Path to requirements.txt')
@click.option('--output', '-o', help='Output file for report')
def check_deps(requirements, output):
    """Check dependencies for known vulnerabilities."""
    click.echo(f"🔍 Checking dependencies in: {requirements}")
    
    if not os.path.exists(requirements):
        click.echo(f"❌ File not found: {requirements}", err=True)
        return 1
    
    scanner = DependencyScanner()
    try:
        vulnerabilities = scanner.scan_requirements(requirements)
        report = scanner.generate_report(vulnerabilities)
        
        if output:
            with open(output, 'w') as f:
                f.write(report)
            click.echo(f"✓ Report saved to: {output}")
        else:
            click.echo(report)
            
    except Exception as e:
        click.echo(f"❌ Dependency scan failed: {e}", err=True)
        click.echo("💡 Make sure pip-audit or safety is installed:")
        click.echo("   pip install pip-audit safety")
        return 1


@main.command()
def init():
    """Initialize a new ML security project."""
    click.echo("🚀 Initializing MLSecScan project...")
    
    # Create directories
    dirs = [
        '.mlsecscan',
        '.mlsecscan/reports',
        '.mlsecscan/scans',
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        click.echo(f"  ✓ Created {dir_path}/")
    
    # Create config file
    config = """# MLSecScan Configuration
scan_on_commit: true
auto_report: true
severity_threshold: MEDIUM
"""
    
    with open('.mlsecscan/config.yaml', 'w') as f:
        f.write(config)
    
    click.echo("✓ Project initialized!")
    click.echo("\nNext steps:")
    click.echo("  1. Scan a model: mlsecscan scan --model path/to/model.pkl")
    click.echo("  2. Check dependencies: mlsecscan check-deps")


if __name__ == '__main__':
    main()
