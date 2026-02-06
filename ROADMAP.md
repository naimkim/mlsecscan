# MLSecScan - 프로젝트 로드맵 및 학습 가이드

## 🎯 프로젝트 목표
취업 포트폴리오용 ML Security 프로젝트로, 실제 산업에서 요구하는 ML 보안 기술을 시연합니다.

---

## 📚 Phase 1: 기초 구축 (완료 ✓)

### 완료된 작업
- ✅ 프로젝트 구조 설정
- ✅ Git 저장소 초기화
- ✅ 기본 모델 스캐너 (pickle, h5, ONNX, PyTorch)
- ✅ Dependency 취약점 스캐너
- ✅ FGSM 기반 Adversarial 테스트
- ✅ CLI 인터페이스
- ✅ 단위 테스트
- ✅ 위협 모델링 문서
- ✅ 사용 예제

### 다음 학습 주제
1. **Pickle Security 심화**
   - Restricted unpickler 구현
   - Safe pickle alternatives 연구
   - 📖 읽을 자료: [Dangerous Pickle](https://intoli.com/blog/dangerous-pickles/)

2. **Adversarial ML 기초**
   - FGSM 원리 이해
   - 📖 읽을 논문: "Explaining and Harnessing Adversarial Examples" (Goodfellow et al.)

---

## 🚀 Phase 2: 핵심 기능 확장 (2-3주)

### 구현할 기능

#### 1. 고급 Adversarial 공격
```python
# 목표: PGD, C&W, DeepFool 구현
mlsecscan/scanners/adversarial_scanner.py 확장

추가할 공격:
- PGD (Projected Gradient Descent)
- C&W (Carlini & Wagner)
- DeepFool
```

**학습 자료:**
- [Adversarial Robustness Toolbox 문서](https://adversarial-robustness-toolbox.readthedocs.io/)
- 논문: "Towards Evaluating the Robustness of Neural Networks"

#### 2. 모델 백도어 탐지
```python
# 목표: 트로이 목마 탐지
mlsecscan/detectors/backdoor_detector.py 생성

기능:
- Activation clustering
- Neural cleanse
- STRIP (STRong Intentional Perturbation)
```

**학습 자료:**
- 논문: "Neural Cleanse: Identifying and Mitigating Backdoor Attacks"
- [TrojanZoo](https://github.com/ain-soph/trojanzoo)

#### 3. Privacy Leakage 테스트
```python
# 목표: Membership inference 공격
mlsecscan/scanners/privacy_scanner.py 생성

기능:
- Membership inference attack
- Model inversion attack
- 차분 프라이버시 검증
```

**학습 자료:**
- 논문: "Membership Inference Attacks Against Machine Learning Models"
- [Privacy Meter](https://github.com/privacytrustlab/ml_privacy_meter)

---

## 🔧 Phase 3: 실전 통합 (3-4주)

### 1. CI/CD 통합
```yaml
# .github/workflows/security-scan.yml
name: ML Security Scan

on: [push, pull_request]

jobs:
  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Scan ML models
        run: mlsecscan scan --model models/*.pkl
```

### 2. 웹 대시보드
```python
# Flask/FastAPI 기반 대시보드
mlsecscan/web/app.py

기능:
- 실시간 스캔 결과 시각화
- 취약점 트렌드 차트
- 자동 리포트 생성
```

### 3. Docker 컨테이너화
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY . .
RUN pip install -e .
ENTRYPOINT ["mlsecscan"]
```

---

## 📊 Phase 4: 포트폴리오 완성 (4-5주)

### 실전 케이스 스터디

#### Case Study 1: Hugging Face 모델 검증
```python
# examples/huggingface_security.py
from mlsecscan import ModelScanner
from transformers import AutoModel

# Download model
model = AutoModel.from_pretrained("bert-base-uncased")

# Security scan
scanner = ModelScanner()
result = scanner.scan_transformer_model(model)
```

#### Case Study 2: 산업별 보안 체크리스트
- **Healthcare ML**: HIPAA 준수, 환자 데이터 프라이버시
- **Financial ML**: 모델 fairness, adversarial robustness
- **Autonomous Vehicles**: Safety-critical 모델 검증

### 블로그 포스트 작성
1. "ML 모델의 숨겨진 위협: Pickle 파일이 위험한 이유"
2. "Adversarial Attack으로부터 모델 보호하기"
3. "MLSecScan으로 ML 파이프라인 보안 자동화하기"

---

## 🎓 학습 리소스

### 필수 읽기 (우선순위 순)
1. **OWASP ML Security Top 10**
   - https://owasp.org/www-project-machine-learning-security-top-10/

2. **NIST AI RMF**
   - https://www.nist.gov/itl/ai-risk-management-framework

3. **Adversarial Robustness Toolbox**
   - https://github.com/Trusted-AI/adversarial-robustness-toolbox

### 추천 논문
1. "Explaining and Harnessing Adversarial Examples" (Goodfellow, 2015)
2. "Towards Evaluating the Robustness of Neural Networks" (Carlini & Wagner, 2017)
3. "Membership Inference Attacks Against Machine Learning Models" (Shokri et al., 2017)

### 온라인 코스
- Coursera: "AI Security and Privacy"
- YouTube: "Adversarial Machine Learning" by Ian Goodfellow

---

## 💼 취업 준비 체크리스트

### 기술 역량
- [ ] Adversarial ML 공격/방어 이해
- [ ] Threat modeling 능력
- [ ] Security tool 개발 경험
- [ ] CI/CD integration
- [ ] 보안 컴플라이언스 (NIST, ISO)

### 포트폴리오
- [ ] GitHub 프로젝트 완성도 90%+
- [ ] README.md 전문성
- [ ] 3개 이상 케이스 스터디
- [ ] 블로그 포스트 2개 이상
- [ ] 데모 비디오

### 네트워킹
- [ ] MLSecOps 커뮤니티 참여
- [ ] Kaggle/GitHub discussions
- [ ] LinkedIn 기술 포스트
- [ ] Conference/Meetup 참석

---

## 🔄 다음 단계 (우선순위)

### 이번 주 (Week 1-2)
1. **PGD 공격 구현** - adversarial_scanner.py에 추가
2. **테스트 커버리지 80%** - pytest-cov 사용
3. **첫 블로그 포스트** - "Pickle 보안"

### 다음 주 (Week 3-4)
1. **Backdoor detection** - Neural Cleanse 구현
2. **CI/CD 통합** - GitHub Actions
3. **Docker 컨테이너** - 배포 자동화

### 한 달 후 (Week 5-8)
1. **Hugging Face 케이스 스터디**
2. **웹 대시보드** - Flask/React
3. **케이스 스터디 2개 완성**

---

## 📝 코드 품질 기준

### 코딩 스타일
- Black formatter 사용
- Type hints 추가
- Docstrings (Google style)

### 테스트
- 단위 테스트 80%+ 커버리지
- Integration tests
- Security regression tests

### 문서화
- API documentation (Sphinx)
- 사용자 가이드
- 개발자 가이드

---

## 🎯 최종 목표

**3개월 후:**
- ⭐ 500+ stars on GitHub
- 📦 PyPI 패키지 출시
- 📝 3개 블로그 포스트
- 🎤 1개 기술 발표
- 💼 ML Security 엔지니어 포지션 획득

---

## 💡 팁

1. **작게 시작**: 한 번에 하나씩 구현
2. **테스트 먼저**: TDD 접근
3. **문서화 습관**: 코드 작성하면서 동시에
4. **커뮤니티 활용**: 질문하고 피드백 받기
5. **꾸준함**: 매일 1시간씩

---

## 📞 리소스

- **공식 문서**: [프로젝트 Wiki]
- **이슈 트래커**: [GitHub Issues]
- **토론**: [GitHub Discussions]
- **슬랙**: [ML Security Community]

---

**시작은 작지만, 꾸준히 하면 큰 프로젝트가 됩니다!** 🚀

*마지막 업데이트: 2026-02-06*
