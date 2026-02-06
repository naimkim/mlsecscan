# MLSecScan - 빠른 시작 가이드

## 🚀 5분 안에 시작하기

### 1. 프로젝트 클론
```bash
git clone https://github.com/yourusername/mlsecscan.git
cd mlsecscan
```

### 2. 가상환경 설정
```bash
# 가상환경 생성
python -m venv venv

# 활성화 (Linux/Mac)
source venv/bin/activate

# 활성화 (Windows)
venv\Scripts\activate
```

### 3. 의존성 설치
```bash
# 개발 모드로 설치
pip install -e .

# 또는 기본 설치만
pip install -r requirements.txt
```

### 4. 첫 번째 스캔 실행
```bash
# 예제 스크립트 실행
cd examples
python basic_scan.py
```

---

## 📋 주요 사용법

### CLI 사용

#### 모델 스캔
```bash
# 단일 모델 스캔
mlsecscan scan --model path/to/model.pkl

# 상세 리포트
mlsecscan scan --model model.pkl --full-report

# 특정 모델 타입 지정
mlsecscan scan --model model.h5 --type h5
```

#### Dependency 체크
```bash
# requirements.txt 스캔
mlsecscan check-deps

# 다른 파일 지정
mlsecscan check-deps --requirements myreqs.txt

# 리포트 저장
mlsecscan check-deps --output security_report.txt
```

#### 프로젝트 초기화
```bash
# 새 프로젝트에 MLSecScan 설정
mlsecscan init
```

---

### Python API 사용

#### 기본 모델 스캔
```python
from mlsecscan import ModelScanner

scanner = ModelScanner()
result = scanner.scan_model('model.pkl')

# 결과 출력
print(result.summary())

# Critical 이슈만 확인
for finding in result.get_critical_findings():
    print(finding)
```

#### Dependency 스캔
```python
from mlsecscan.scanners import DependencyScanner

scanner = DependencyScanner()
vulns = scanner.scan_requirements('requirements.txt')

# 보고서 생성
report = scanner.generate_report(vulns)
print(report)
```

#### Adversarial 테스트
```python
from mlsecscan.scanners import RobustnessScanner
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

# 모델 준비
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
model = RandomForestClassifier()
model.fit(X[:800], y[:800])

# 로버스트니스 테스트
scanner = RobustnessScanner()
result = scanner.test_model(model, X[800:], y[800:], epsilon=0.1)

print(result)
print(scanner.get_robustness_score(result))
```

---

## 🧪 테스트 실행

```bash
# 모든 테스트 실행
pytest

# 커버리지 포함
pytest --cov=mlsecscan

# 상세 출력
pytest -v

# 특정 테스트만
pytest tests/test_model_scanner.py
```

---

## 🔧 개발 환경 설정

### Pre-commit hooks 설정
```bash
pip install pre-commit
pre-commit install
```

### 코드 포맷팅
```bash
# Black으로 포맷
black mlsecscan/

# Flake8으로 린트
flake8 mlsecscan/

# 타입 체크
mypy mlsecscan/
```

---

## 📂 프로젝트 구조

```
mlsecscan/
├── mlsecscan/           # 메인 패키지
│   ├── core/           # 핵심 스캐너
│   ├── scanners/       # 특화 스캐너들
│   ├── detectors/      # 탐지 모듈
│   └── utils/          # 유틸리티
├── tests/              # 테스트
├── examples/           # 사용 예제
├── docs/              # 문서
└── requirements.txt   # 의존성
```

---

## 💡 일반적인 문제 해결

### 문제: pip-audit가 설치되지 않음
```bash
pip install pip-audit
```

### 문제: TensorFlow 버전 충돌
```bash
# CPU 버전 사용
pip install tensorflow-cpu==2.13.0
```

### 문제: "Module not found" 에러
```bash
# 개발 모드로 재설치
pip install -e .
```

---

## 🎯 다음 단계

1. **예제 실행**: `examples/basic_scan.py` 실행해보기
2. **문서 읽기**: `docs/threat_model.md` 읽기
3. **테스트 작성**: 자신만의 테스트 케이스 추가
4. **기여하기**: 새로운 기능이나 버그 픽스 PR 보내기

---

## 📚 추가 자료

- [전체 문서](docs/)
- [위협 모델](docs/threat_model.md)
- [개발 로드맵](ROADMAP.md)
- [이슈 트래커](https://github.com/yourusername/mlsecscan/issues)

---

## 🤝 도움이 필요하신가요?

- 🐛 버그 리포트: [GitHub Issues]
- 💬 질문: [GitHub Discussions]
- 📧 이메일: your.email@example.com

---

**Happy Scanning!** 🔒🤖
