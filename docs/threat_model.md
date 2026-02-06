# ML System Threat Model

## Overview
This document outlines the threat model for machine learning systems that MLSecScan is designed to protect against.

## Threat Categories

### 1. Model Poisoning
**Description**: Attackers inject malicious data during training to compromise model behavior.

**Attack Vectors**:
- Data poisoning in training datasets
- Backdoor insertion during training
- Supply chain attacks on training pipelines

**MLSecScan Detection**:
- Model integrity checks
- Anomaly detection in model weights
- Training data validation (future)

**Severity**: CRITICAL

---

### 2. Evasion Attacks
**Description**: Adversaries craft inputs to fool the model at inference time.

**Attack Vectors**:
- Adversarial examples (FGSM, PGD, C&W)
- Input perturbations below human perception threshold
- Gradient-based attacks

**MLSecScan Detection**:
- Adversarial robustness testing
- FGSM attack simulation
- Model confidence analysis

**Severity**: HIGH

---

### 3. Model Extraction
**Description**: Attackers steal model functionality through query access.

**Attack Vectors**:
- Query-based model stealing
- Knowledge distillation attacks
- API abuse

**MLSecScan Detection**:
- API rate limiting recommendations
- Query pattern analysis (future)

**Severity**: MEDIUM

---

### 4. Model Inversion
**Description**: Recovering training data from model outputs.

**Attack Vectors**:
- Membership inference attacks
- Training data reconstruction
- Privacy leakage through predictions

**MLSecScan Detection**:
- Privacy leakage tests
- Membership inference simulation
- Differential privacy checks (future)

**Severity**: HIGH

---

### 5. Supply Chain Attacks
**Description**: Compromised dependencies or model files.

**Attack Vectors**:
- Malicious code in pickle files
- Vulnerable dependencies (CVEs)
- Typosquatting in package names
- Compromised model repositories

**MLSecScan Detection**:
- Pickle deserialization scanning
- Dependency vulnerability checking
- Code injection detection

**Severity**: CRITICAL

---

### 6. Code Injection
**Description**: Arbitrary code execution through model files.

**Attack Vectors**:
- Malicious pickle files
- Custom layers with embedded code
- Lambda functions in models
- Unsafe deserialization

**MLSecScan Detection**:
- Pickle content analysis
- Lambda layer detection
- Unsafe operation scanning

**Severity**: CRITICAL

---

## NIST AI RMF Alignment

MLSecScan aligns with the NIST AI Risk Management Framework:

### Govern
- Security policy enforcement
- Compliance checking
- Audit trail generation

### Map
- Threat identification
- Risk assessment
- Context documentation

### Measure
- Robustness testing
- Vulnerability scoring
- Performance metrics

### Manage
- Remediation guidance
- Continuous monitoring
- Incident response

---

## Attack Surface Analysis

### Model Files
- **Pickle (.pkl)**: HIGH RISK - Can execute arbitrary code
- **HDF5 (.h5)**: MEDIUM RISK - Can contain custom layers
- **ONNX (.onnx)**: LOW RISK - Relatively safe format
- **PyTorch (.pt)**: HIGH RISK - Can execute code during loading

### Dependencies
- **ML Libraries**: TensorFlow, PyTorch, scikit-learn
- **Data Processing**: NumPy, Pandas, Pillow
- **Serving**: Flask, FastAPI, TensorFlow Serving

### Deployment
- **API Endpoints**: Authentication, rate limiting, input validation
- **Model Updates**: Version control, integrity checks
- **Monitoring**: Logging, anomaly detection

---

## Mitigation Strategies

### Immediate Actions
1. **Never load untrusted pickle files**
   - Use ONNX or SavedModel format
   - Implement restricted unpickler

2. **Keep dependencies updated**
   - Regular vulnerability scanning
   - Automated patching

3. **Validate all inputs**
   - Input sanitization
   - Anomaly detection

### Long-term Defenses
1. **Adversarial Training**
   - Include adversarial examples in training
   - Use certified defenses

2. **Privacy-Preserving ML**
   - Differential privacy
   - Federated learning
   - Secure aggregation

3. **Continuous Monitoring**
   - Real-time attack detection
   - Behavioral analytics
   - Automated response

---

## References

1. [OWASP Machine Learning Security Top 10](https://owasp.org/www-project-machine-learning-security-top-10/)
2. [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework)
3. [Adversarial Robustness Toolbox](https://github.com/Trusted-AI/adversarial-robustness-toolbox)
4. [MITRE ATLAS - Adversarial Threat Landscape for AI Systems](https://atlas.mitre.org/)

---

## Threat Matrix

| Threat | Likelihood | Impact | Risk Level | MLSecScan Coverage |
|--------|-----------|--------|------------|-------------------|
| Model Poisoning | Medium | Critical | HIGH | Partial |
| Evasion Attacks | High | High | HIGH | Good |
| Model Extraction | Medium | Medium | MEDIUM | Limited |
| Model Inversion | Low | High | MEDIUM | Partial |
| Supply Chain | High | Critical | CRITICAL | Excellent |
| Code Injection | Medium | Critical | CRITICAL | Excellent |

---

*Last Updated: 2026-02-06*
