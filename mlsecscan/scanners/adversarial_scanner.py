"""
Adversarial robustness scanner for ML models.
"""

import numpy as np
import logging
from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RobustnessResult:
    """Results from adversarial robustness testing."""
    attack_name: str
    original_accuracy: float
    adversarial_accuracy: float
    attack_success_rate: float
    epsilon: float
    samples_tested: int
    metadata: Dict[str, Any] = None
    
    def __str__(self):
        return f"""
{self.attack_name} Attack Results:
  Original Accuracy: {self.original_accuracy:.2%}
  Adversarial Accuracy: {self.adversarial_accuracy:.2%}
  Attack Success Rate: {self.attack_success_rate:.2%}
  Epsilon: {self.epsilon}
  Samples Tested: {self.samples_tested}
"""


class RobustnessScanner:
    """
    Scanner for testing model robustness against adversarial attacks.
    
    Implements basic adversarial attacks to test model security:
    - FGSM (Fast Gradient Sign Method)
    - PGD (Projected Gradient Descent) - future
    """
    
    def __init__(self):
        self.logger = logger
    
    def test_model(
        self, 
        model: Any, 
        X_test: np.ndarray, 
        y_test: np.ndarray,
        epsilon: float = 0.1,
        framework: str = 'auto'
    ) -> RobustnessResult:
        """
        Test model robustness against adversarial attacks.
        
        Args:
            model: The ML model to test
            X_test: Test data
            y_test: Test labels
            epsilon: Perturbation magnitude
            framework: ML framework ('tensorflow', 'pytorch', 'sklearn', 'auto')
            
        Returns:
            RobustnessResult with test results
        """
        if framework == 'auto':
            framework = self._detect_framework(model)
        
        self.logger.info(f"Testing model robustness with {framework} framework")
        
        # Get original accuracy
        original_accuracy = self._evaluate_model(model, X_test, y_test, framework)
        
        # Generate adversarial examples with FGSM
        X_adv = self._fgsm_attack(model, X_test, y_test, epsilon, framework)
        
        # Test on adversarial examples
        adv_accuracy = self._evaluate_model(model, X_adv, y_test, framework)
        
        # Calculate attack success rate
        attack_success_rate = 1.0 - (adv_accuracy / max(original_accuracy, 0.001))
        
        result = RobustnessResult(
            attack_name="FGSM",
            original_accuracy=original_accuracy,
            adversarial_accuracy=adv_accuracy,
            attack_success_rate=attack_success_rate,
            epsilon=epsilon,
            samples_tested=len(X_test),
            metadata={'framework': framework}
        )
        
        self.logger.info(f"Robustness test complete: {attack_success_rate:.2%} attack success rate")
        return result
    
    def _detect_framework(self, model: Any) -> str:
        """Auto-detect the ML framework being used."""
        model_type = type(model).__name__
        module = type(model).__module__
        
        if 'tensorflow' in module or 'keras' in module:
            return 'tensorflow'
        elif 'torch' in module:
            return 'pytorch'
        elif 'sklearn' in module:
            return 'sklearn'
        else:
            self.logger.warning(f"Unknown framework for model type: {model_type}")
            return 'unknown'
    
    def _evaluate_model(
        self, 
        model: Any, 
        X: np.ndarray, 
        y: np.ndarray,
        framework: str
    ) -> float:
        """Evaluate model accuracy."""
        try:
            if framework == 'sklearn':
                return model.score(X, y)
            elif framework == 'tensorflow':
                predictions = model.predict(X)
                if len(predictions.shape) > 1 and predictions.shape[1] > 1:
                    predictions = np.argmax(predictions, axis=1)
                else:
                    predictions = (predictions > 0.5).astype(int).flatten()
                
                if len(y.shape) > 1:
                    y = np.argmax(y, axis=1)
                
                return np.mean(predictions == y)
            elif framework == 'pytorch':
                import torch
                model.eval()
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X)
                    outputs = model(X_tensor)
                    predictions = torch.argmax(outputs, dim=1).numpy()
                    return np.mean(predictions == y)
            else:
                self.logger.error(f"Unsupported framework: {framework}")
                return 0.0
        except Exception as e:
            self.logger.error(f"Error evaluating model: {e}")
            return 0.0
    
    def _fgsm_attack(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        epsilon: float,
        framework: str
    ) -> np.ndarray:
        """
        Fast Gradient Sign Method attack.
        
        Generates adversarial examples by adding small perturbations
        in the direction of the gradient.
        """
        if framework == 'sklearn':
            # For sklearn, use a simple approximation
            return self._fgsm_sklearn(model, X, y, epsilon)
        elif framework == 'tensorflow':
            return self._fgsm_tensorflow(model, X, y, epsilon)
        elif framework == 'pytorch':
            return self._fgsm_pytorch(model, X, y, epsilon)
        else:
            self.logger.warning(f"FGSM not implemented for {framework}, returning original data")
            return X
    
    def _fgsm_sklearn(
        self, 
        model: Any, 
        X: np.ndarray, 
        y: np.ndarray, 
        epsilon: float
    ) -> np.ndarray:
        """FGSM for sklearn models (simplified version)."""
        # For sklearn, we'll use a simple numerical gradient approximation
        X_adv = X.copy()
        delta = 1e-4
        
        for i in range(len(X)):
            # Get original prediction
            orig_pred = model.predict_proba([X[i]])[0] if hasattr(model, 'predict_proba') else model.predict([X[i]])
            
            # Compute approximate gradient
            gradient = np.zeros_like(X[i])
            for j in range(len(X[i])):
                X_perturbed = X[i].copy()
                X_perturbed[j] += delta
                
                new_pred = model.predict_proba([X_perturbed])[0] if hasattr(model, 'predict_proba') else model.predict([X_perturbed])
                
                # Simple gradient approximation
                gradient[j] = (new_pred[y[i]] - orig_pred[y[i]]) / delta if hasattr(model, 'predict_proba') else 0
            
            # Apply FGSM perturbation
            X_adv[i] = X[i] - epsilon * np.sign(gradient)
        
        return X_adv
    
    def _fgsm_tensorflow(
        self, 
        model: Any, 
        X: np.ndarray, 
        y: np.ndarray, 
        epsilon: float
    ) -> np.ndarray:
        """FGSM for TensorFlow/Keras models."""
        try:
            import tensorflow as tf
            
            X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
            y_tensor = tf.convert_to_tensor(y)
            
            # Ensure y is in the right format
            if len(y_tensor.shape) == 1:
                y_tensor = tf.one_hot(y_tensor, depth=model.output_shape[-1])
            
            with tf.GradientTape() as tape:
                tape.watch(X_tensor)
                predictions = model(X_tensor)
                loss = tf.keras.losses.categorical_crossentropy(y_tensor, predictions)
            
            # Get gradient
            gradient = tape.gradient(loss, X_tensor)
            
            # Create adversarial examples
            X_adv = X_tensor + epsilon * tf.sign(gradient)
            
            return X_adv.numpy()
            
        except Exception as e:
            self.logger.error(f"TensorFlow FGSM failed: {e}")
            return X
    
    def _fgsm_pytorch(
        self, 
        model: Any, 
        X: np.ndarray, 
        y: np.ndarray, 
        epsilon: float
    ) -> np.ndarray:
        """FGSM for PyTorch models."""
        try:
            import torch
            import torch.nn as nn
            
            X_tensor = torch.FloatTensor(X)
            y_tensor = torch.LongTensor(y)
            
            X_tensor.requires_grad = True
            
            # Forward pass
            outputs = model(X_tensor)
            loss = nn.CrossEntropyLoss()(outputs, y_tensor)
            
            # Backward pass
            model.zero_grad()
            loss.backward()
            
            # Generate adversarial examples
            X_adv = X_tensor + epsilon * X_tensor.grad.sign()
            
            return X_adv.detach().numpy()
            
        except Exception as e:
            self.logger.error(f"PyTorch FGSM failed: {e}")
            return X
    
    def get_robustness_score(self, result: RobustnessResult) -> str:
        """
        Get a human-readable robustness score.
        
        Args:
            result: RobustnessResult from testing
            
        Returns:
            String describing robustness level
        """
        asr = result.attack_success_rate
        
        if asr < 0.1:
            return "EXCELLENT - Model is highly robust to adversarial attacks"
        elif asr < 0.3:
            return "GOOD - Model shows good robustness"
        elif asr < 0.5:
            return "MODERATE - Model has some vulnerabilities"
        elif asr < 0.7:
            return "POOR - Model is vulnerable to attacks"
        else:
            return "CRITICAL - Model is highly vulnerable to adversarial attacks"
