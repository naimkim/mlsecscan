# MLSecScan 🔒🤖
MLSecScan - ML 모델 보안 스캐너 프로젝트를 성공적으로 만들었습니다!
📦 완성된 내용
핵심 기능:

✅ ML 모델 보안 스캐닝 (Pickle, H5, ONNX, PyTorch)
✅ Dependency 취약점 검사
✅ FGSM 기반 Adversarial 공격 테스트
✅ CLI 및 Python API
✅ 단위 테스트 및 예제

첫 주 목표

PGD 공격 구현
테스트 커버리지 80%
첫 블로그 포스트 작성


포트폴리오 강화

Hugging Face 모델 스캔 케이스 스터디
CI/CD 파이프라인 통합
Docker 컨테이너화

실전 보안 기술: Threat modeling, vulnerability scanning
ML 전문성: Adversarial ML, model security
소프트웨어 공학: Clean code, testing, documentation
자동화: CLI tools, CI/CD integration
컴플라이언스: NIST AI RMF alignment

## 📊 Dashboard Preview

![MLSecScan Dashboard](docs/images/mlsecscan-dashboard-full.png)

### Key Features

**Security Metrics**
- Model accuracy monitoring
- Risk assessment tracking
- Real-time vehicle status

**Interactive Visualizations**
- Feature importance analysis
- Battery health impact
- Network quality correlation

### Live Demo
[View Figma Prototype](your-figma-share-link-here)

**Machine Learning Model Security Scanner**

A comprehensive security scanning tool for ML models and pipelines that helps identify vulnerabilities, malicious code, and security risks in machine learning artifacts.

## 🎯 Features

- **Model File Security Scanning**
  - Detect malicious code in pickle, h5, ONNX, and PyTorch models
  - Identify unsafe deserialization patterns
  - Check for embedded code execution risks

- **Dependency Vulnerability Analysis**
  - Scan requirements.txt for known CVEs
  - Check for outdated ML libraries
  - Suggest secure version upgrades

- **Data Leakage Detection**
  - Test for training data memorization
  - Membership inference attack simulation
  - Privacy risk assessment

- **Adversarial Robustness Testing**
  - FGSM attack simulation
  - Model robustness scoring
  - Attack success rate reporting

- **Compliance Reporting**
  - NIST AI Risk Management Framework checklist
  - ISO/IEC AI security standards alignment
  - Automated security report generation

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/naimkim/mlsecscan.git
cd mlsecscan

# Install dependencies
pip install -r requirements.txt

# Run basic scan
python -m mlsecscan scan --model path/to/model.pkl

# Generate full security report
python -m mlsecscan scan --model path/to/model.pkl --full-report
```

## 📋 Installation

```bash
pip install -e .
```

Or install from PyPI (when available):
```bash
pip install mlsecscan
```

## 🔧 Usage Examples

### Scan a Pickle Model
```python
from mlsecscan import ModelScanner

scanner = ModelScanner()
results = scanner.scan_model('model.pkl', model_type='pickle')
print(results.summary())
```

### Check Dependencies
```python
from mlsecscan import DependencyScanner

dep_scanner = DependencyScanner()
vulnerabilities = dep_scanner.scan_requirements('requirements.txt')
```

### Test Adversarial Robustness
```python
from mlsecscan import RobustnessScanner

rob_scanner = RobustnessScanner()
robustness_score = rob_scanner.test_model(model, test_data)
```

## 🏗️ Project Structure

```
mlsecscan/
├── mlsecscan/
│   ├── __init__.py
│   ├── core/
│   │   ├── scanner.py          # Main scanning engine
│   │   └── reporter.py         # Report generation
│   ├── scanners/
│   │   ├── model_scanner.py    # Model file scanning
│   │   ├── dependency_scanner.py
│   │   ├── adversarial_scanner.py
│   │   └── privacy_scanner.py
│   ├── detectors/
│   │   ├── pickle_detector.py  # Pickle security checks
│   │   ├── code_injection.py   # Code injection detection
│   │   └── malware_detector.py
│   └── utils/
│       ├── threat_models.py    # Threat modeling utilities
│       └── compliance.py       # Compliance frameworks
├── tests/
│   ├── test_model_scanner.py
│   ├── test_adversarial.py
│   └── fixtures/
├── examples/
│   ├── basic_scan.py
│   └── advanced_threat_modeling.py
├── docs/
│   ├── threat_model.md
│   └── security_guidelines.md
├── requirements.txt
├── setup.py
├── .gitignore
└── README.md
```

## 🛡️ Threat Coverage

- **Model Poisoning**: Detection of backdoors and poisoned models
- **Evasion Attacks**: Adversarial example testing
- **Data Poisoning**: Training data integrity checks
- **Model Inversion**: Privacy leakage detection
- **Supply Chain**: Dependency vulnerability scanning
- **Code Injection**: Malicious code in model files

## 📊 Example Output

```
MLSecScan Report
================
Model: sentiment_model.pkl
Scan Date: 2026-02-06

⚠️  SECURITY FINDINGS:
[HIGH] Unsafe pickle deserialization detected
[MEDIUM] 3 dependencies with known CVEs
[LOW] Model vulnerable to FGSM attacks (success rate: 45%)

✓ PASSED CHECKS:
[✓] No code injection patterns found
[✓] Data leakage test passed
[✓] NIST AI RMF basic compliance

Recommendation: Update scikit-learn to version 1.3.2+
```

## 🤝 Contributing

Contributions welcome! This project is built for learning and portfolio development.

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/amazing-scanner`)
3. Commit changes (`git commit -m 'Add amazing scanner'`)
4. Push to branch (`git push origin feature/amazing-scanner`)
5. Open a Pull Request

## 📚 Learning Resources

- [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework)
- [OWASP Machine Learning Security Top 10](https://owasp.org/www-project-machine-learning-security-top-10/)
- [Adversarial Robustness Toolbox](https://github.com/Trusted-AI/adversarial-robustness-toolbox)

## 📄 License

MIT License - see LICENSE file for details

## 🎓 Portfolio Project

This project demonstrates skills in:
- ML Security & Threat Modeling
- Python Security Tools Development
- Adversarial Machine Learning
- Security Compliance Frameworks
- Software Engineering Best Practices

---

**Built for learning and securing ML systems** 🚀
