"""
OTA Update Prediction Model - Demo for MLSecScan
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pickle

# 가상의 차량 OTA 업데이트 데이터
np.random.seed(42)
n_samples = 1000

# Features: 차량 나이, 주행거리, 이전 업데이트 횟수, 배터리 상태
vehicle_age = np.random.randint(0, 60, n_samples)  # months
mileage = np.random.randint(0, 100000, n_samples)  # km
prev_updates = np.random.randint(0, 20, n_samples)
battery_health = np.random.uniform(70, 100, n_samples)  # %

# Target: 업데이트 성공 여부
# 간단한 규칙: 새 차량, 배터리 좋으면 성공률 높음
success_prob = (100 - vehicle_age) / 100 * battery_health / 100
update_success = (np.random.random(n_samples) < success_prob).astype(int)

X = np.column_stack([vehicle_age, mileage, prev_updates, battery_health])
y = update_success

# 모델 학습
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# 저장
pickle.dump(model, open('ota_predictor.pkl', 'wb'))
print("✓ OTA update predictor model created!")

# MLSecScan으로 스캔
from mlsecscan import ModelScanner
scanner = ModelScanner()
result = scanner.scan_model('ota_predictor.pkl')
print(result.summary())