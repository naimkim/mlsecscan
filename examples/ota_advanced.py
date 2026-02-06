"""
Advanced OTA Update Risk Prediction Model
==========================================

This model predicts the risk level of OTA updates for embedded automotive systems.
Designed for demonstration of ML security scanning in automotive context.

Use Case: 
- Predict optimal timing for ECU firmware updates
- Assess update risk based on vehicle conditions
- Prevent update failures in critical situations

Author: MLSecScan Project
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


class OTAUpdatePredictor:
    """
    OTA Update Risk Assessment Model for Automotive ECUs
    
    Features:
    - Vehicle operational parameters
    - Environmental conditions
    - Update history
    - Network quality metrics
    """
    
    def __init__(self, model_type='random_forest'):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = [
            'vehicle_age_months',
            'total_mileage_km',
            'battery_voltage',
            'battery_soc_percent',
            'ecu_temperature_celsius',
            'network_signal_strength',
            'previous_update_count',
            'days_since_last_update',
            'engine_running_hours',
            'diagnostic_trouble_codes',
            'avg_daily_mileage',
            'update_file_size_mb'
        ]
        
    def generate_synthetic_data(self, n_samples=5000):
        """
        Generate realistic synthetic OTA update data
        
        This simulates real-world vehicle conditions and update outcomes
        """
        np.random.seed(42)
        
        # Vehicle characteristics
        vehicle_age_months = np.random.randint(0, 72, n_samples)  # 0-6 years
        total_mileage_km = vehicle_age_months * np.random.uniform(800, 2500, n_samples)
        
        # Electrical system
        battery_voltage = np.random.normal(12.6, 0.5, n_samples)  # Volts
        battery_voltage = np.clip(battery_voltage, 11.0, 14.5)
        battery_soc_percent = np.random.uniform(20, 100, n_samples)  # State of Charge
        
        # ECU status
        ecu_temperature_celsius = np.random.normal(45, 15, n_samples)
        ecu_temperature_celsius = np.clip(ecu_temperature_celsius, -10, 85)
        
        # Network quality (LTE/5G signal strength in dBm)
        network_signal_strength = np.random.normal(-75, 15, n_samples)
        network_signal_strength = np.clip(network_signal_strength, -110, -40)
        
        # Update history
        previous_update_count = np.random.poisson(3, n_samples)
        days_since_last_update = np.random.exponential(90, n_samples)
        
        # Vehicle usage patterns
        engine_running_hours = total_mileage_km / np.random.uniform(30, 80, n_samples)
        diagnostic_trouble_codes = np.random.poisson(0.5, n_samples)
        avg_daily_mileage = total_mileage_km / (vehicle_age_months * 30 + 1)
        
        # Update characteristics
        update_file_size_mb = np.random.uniform(10, 500, n_samples)
        
        # Create DataFrame
        data = pd.DataFrame({
            'vehicle_age_months': vehicle_age_months,
            'total_mileage_km': total_mileage_km,
            'battery_voltage': battery_voltage,
            'battery_soc_percent': battery_soc_percent,
            'ecu_temperature_celsius': ecu_temperature_celsius,
            'network_signal_strength': network_signal_strength,
            'previous_update_count': previous_update_count,
            'days_since_last_update': days_since_last_update,
            'engine_running_hours': engine_running_hours,
            'diagnostic_trouble_codes': diagnostic_trouble_codes,
            'avg_daily_mileage': avg_daily_mileage,
            'update_file_size_mb': update_file_size_mb
        })
        
        # Generate target: Update Success (0: FAIL, 1: SUCCESS)
        # Complex rules based on automotive engineering principles
        success_score = 0.0
        
        # Battery health is critical
        success_score += (battery_voltage > 12.0) * 0.25
        success_score += (battery_soc_percent > 50) * 0.20
        
        # Temperature within safe range
        temp_safe = (ecu_temperature_celsius > 0) & (ecu_temperature_celsius < 70)
        success_score += temp_safe * 0.15
        
        # Good network quality
        success_score += (network_signal_strength > -85) * 0.15
        
        # No critical errors
        success_score += (diagnostic_trouble_codes == 0) * 0.10
        
        # Vehicle not too old
        success_score += (vehicle_age_months < 48) * 0.10
        
        # Reasonable file size for network
        success_score += (update_file_size_mb < 200) * 0.05
        
        # Add some realistic noise
        noise = np.random.normal(0, 0.1, n_samples)
        success_prob = np.clip(success_score + noise, 0, 1)
        
        # Generate binary outcome
        update_success = (np.random.random(n_samples) < success_prob).astype(int)
        
        # Add risk category (for multi-class classification example)
        risk_level = np.zeros(n_samples, dtype=int)
        risk_level[success_prob < 0.3] = 2  # HIGH RISK
        risk_level[(success_prob >= 0.3) & (success_prob < 0.7)] = 1  # MEDIUM RISK
        risk_level[success_prob >= 0.7] = 0  # LOW RISK
        
        data['update_success'] = update_success
        data['risk_level'] = risk_level
        
        return data
    
    def train(self, data, target='update_success'):
        """Train the OTA update prediction model"""
        
        X = data[self.feature_names]
        y = data[target]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=4,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=150,
                max_depth=8,
                learning_rate=0.1,
                random_state=42
            )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
        
        print(f"\n{'='*60}")
        print(f"OTA Update Predictor - Training Results")
        print(f"{'='*60}")
        print(f"Model Type: {self.model_type}")
        print(f"Training Samples: {len(X_train)}")
        print(f"Test Samples: {len(X_test)}")
        print(f"\nPerformance Metrics:")
        print(f"  Train Accuracy: {train_score:.4f}")
        print(f"  Test Accuracy:  {test_score:.4f}")
        print(f"  CV Accuracy:    {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # Predictions
        y_pred = self.model.predict(X_test_scaled)
        
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                   target_names=['FAIL', 'SUCCESS']))
        
        print(f"\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\nTop 5 Important Features:")
            for idx, row in importance_df.head().iterrows():
                print(f"  {row['feature']:30s}: {row['importance']:.4f}")
        
        print(f"{'='*60}\n")
        
        return test_score
    
    def save_model(self, filepath='ota_predictor_advanced.pkl'):
        """Save the trained model"""
        model_package = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_type': self.model_type,
            'timestamp': datetime.now().isoformat(),
            'version': '1.0.0'
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_package, f)
        
        print(f"✓ Model saved to: {filepath}")
        return filepath
    
    def predict_update_risk(self, vehicle_data):
        """
        Predict update risk for a specific vehicle
        
        Args:
            vehicle_data: dict with vehicle parameters
            
        Returns:
            prediction and confidence
        """
        X = pd.DataFrame([vehicle_data])[self.feature_names]
        X_scaled = self.scaler.transform(X)
        
        prediction = self.model.predict(X_scaled)[0]
        probability = self.model.predict_proba(X_scaled)[0]
        
        return {
            'prediction': 'SUCCESS' if prediction == 1 else 'FAIL',
            'confidence': max(probability),
            'success_probability': probability[1] if len(probability) > 1 else probability[0]
        }


def main():
    """Main execution for OTA prediction model"""
    
    print("""
╔══════════════════════════════════════════════════════════╗
║  Advanced OTA Update Risk Prediction System             ║
║  Automotive Embedded Systems - MLSecScan Demo           ║
╚══════════════════════════════════════════════════════════╝
""")
    
    # Initialize predictor
    predictor = OTAUpdatePredictor(model_type='random_forest')
    
    # Generate realistic automotive data
    print("Generating synthetic vehicle OTA data...")
    data = predictor.generate_synthetic_data(n_samples=5000)
    
    print(f"✓ Generated {len(data)} vehicle update scenarios")
    print(f"\nData Statistics:")
    print(f"  Success Rate: {data['update_success'].mean():.2%}")
    print(f"  Risk Distribution:")
    print(f"    LOW:    {(data['risk_level']==0).sum()} ({(data['risk_level']==0).mean():.1%})")
    print(f"    MEDIUM: {(data['risk_level']==1).sum()} ({(data['risk_level']==1).mean():.1%})")
    print(f"    HIGH:   {(data['risk_level']==2).sum()} ({(data['risk_level']==2).mean():.1%})")
    
    # Train model
    print("\nTraining OTA update prediction model...")
    accuracy = predictor.train(data, target='update_success')
    
    # Save model
    model_path = predictor.save_model('ota_predictor_advanced.pkl')
    
    # Example prediction
    print("\n" + "="*60)
    print("Example Prediction - Test Vehicle Scenarios")
    print("="*60)
    
    test_scenarios = [
        {
            'name': 'Ideal Conditions',
            'vehicle_age_months': 12,
            'total_mileage_km': 15000,
            'battery_voltage': 12.8,
            'battery_soc_percent': 85,
            'ecu_temperature_celsius': 35,
            'network_signal_strength': -65,
            'previous_update_count': 2,
            'days_since_last_update': 60,
            'engine_running_hours': 250,
            'diagnostic_trouble_codes': 0,
            'avg_daily_mileage': 42,
            'update_file_size_mb': 150
        },
        {
            'name': 'Risky Conditions',
            'vehicle_age_months': 60,
            'total_mileage_km': 120000,
            'battery_voltage': 11.8,
            'battery_soc_percent': 35,
            'ecu_temperature_celsius': 75,
            'network_signal_strength': -95,
            'previous_update_count': 8,
            'days_since_last_update': 5,
            'engine_running_hours': 2500,
            'diagnostic_trouble_codes': 3,
            'avg_daily_mileage': 65,
            'update_file_size_mb': 450
        }
    ]
    
    for scenario in test_scenarios:
        name = scenario.pop('name')
        result = predictor.predict_update_risk(scenario)
        print(f"\n{name}:")
        print(f"  Prediction: {result['prediction']}")
        print(f"  Confidence: {result['confidence']:.2%}")
        print(f"  Success Probability: {result['success_probability']:.2%}")
    
    # Security scan with MLSecScan
    print("\n" + "="*60)
    print("MLSecScan Security Analysis")
    print("="*60)
    
    try:
        from mlsecscan import ModelScanner, RobustnessScanner
        
        # Model security scan
        print("\n1. Model File Security Scan...")
        scanner = ModelScanner()
        scan_result = scanner.scan_model(model_path)
        print(scan_result.summary())
        
        # Adversarial robustness test
        print("\n2. Adversarial Robustness Test...")
        X = data[predictor.feature_names].values[:500]
        y = data['update_success'].values[:500]
        
        rob_scanner = RobustnessScanner()
        rob_result = rob_scanner.test_model(
            predictor.model, 
            predictor.scaler.transform(X), 
            y, 
            epsilon=0.1,
            framework='sklearn'
        )
        
        print(rob_result)
        print(f"\nRobustness Assessment: {rob_scanner.get_robustness_score(rob_result)}")
        
    except ImportError:
        print("⚠️  MLSecScan not installed. Skipping security analysis.")
    
    print("\n" + "="*60)
    print("✓ OTA Update Prediction Model Complete!")
    print("="*60)
    print(f"\nModel saved to: {model_path}")
    print("Ready for deployment in automotive OTA systems")


if __name__ == '__main__':
    main()