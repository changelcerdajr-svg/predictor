# pipeline/calibration/temperature_scaling.py
"""
Temperature scaling: the simplest, most theoretically sound calibrator.
A single scalar T divides logits: p_cal = sigmoid(logit(p_raw) / T)

Why temperature scaling over isotonic/Platt on sports data:
- Isotonic regression can overfit on ~2000 games per season
- Temperature scaling has 1 parameter; cannot overfit
- Preserves ranking (monotonic transformation)
"""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from sklearn.metrics import brier_score_loss

class TemperatureScaler:
    def __init__(self):
        self.T = 1.0
    
    def fit(self, probs: np.ndarray, labels: np.ndarray) -> "TemperatureScaler":
        """
        CRITICAL: fit ONLY on the same distribution seen in production.
        If you filter bets by confidence in production, fit only on 
        games that would have passed that filter.
        """
        def nll(T):
            calibrated = expit(logit(np.clip(probs, 1e-6, 1-1e-6)) / T)
            return brier_score_loss(labels, calibrated)
        
        result = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
        self.T = result.x
        return self
    
    def transform(self, probs: np.ndarray) -> np.ndarray:
        return expit(logit(np.clip(probs, 1e-6, 1-1e-6)) / self.T)
    
    def fit_transform(self, probs, labels):
        return self.fit(probs, labels).transform(probs)