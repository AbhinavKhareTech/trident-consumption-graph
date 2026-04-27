"""Probability calibration for the ensemble meta-learner."""
from __future__ import annotations
import numpy as np
from sklearn.calibration import calibration_curve


def evaluate_calibration(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> dict:
    """Evaluate probability calibration of ensemble predictions."""
    fraction_pos, mean_predicted = calibration_curve(y_true, y_prob, n_bins=n_bins)
    ece = np.mean(np.abs(fraction_pos - mean_predicted))
    return {"expected_calibration_error": float(ece), "fraction_positives": fraction_pos.tolist(),
            "mean_predicted": mean_predicted.tolist()}
