"""Ensemble meta-learner: stacked generalization of three prongs.

Combines prediction scores from PyG (structural), DGL (temporal),
and XGBoost (tabular) into calibrated final probabilities.

The ensemble consistently outperforms any single prong because each
captures orthogonal signals:
- PyG: strong structure, misses behavioral drift
- DGL: catches drift, misses tabular thresholds
- XGBoost: handles thresholds, misses graph topology

The biryani-every-Thursday-but-stopped example:
- PyG: 0.91 (strong structural affinity, many edges)
- DGL: 0.43 (decaying temporal weight, drift detected)
- XGBoost: 0.35 (recency_days = 21, frequency dropping)
- Ensemble: 0.38 (correctly resolves the conflict)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score


@dataclass
class ProngScores:
    """Prediction scores from each prong for a set of (user, entity) pairs."""

    pyg_scores: np.ndarray  # [N, 1] structural link probabilities
    dgl_scores: np.ndarray  # [N, 1] temporal link probabilities
    xgb_scores: np.ndarray  # [N, 1] tabular probabilities

    def to_matrix(self) -> np.ndarray:
        """Stack into [N, 3] feature matrix for meta-learner."""
        return np.hstack([self.pyg_scores, self.dgl_scores, self.xgb_scores])


class TridentEnsemble:
    """Stacked generalization meta-learner.

    Level 0: PyG, DGL, XGBoost produce independent scores.
    Level 1: Logistic regression (or lightweight MLP) combines them.
    Calibration: Platt scaling ensures output probabilities are well-calibrated.
    """

    def __init__(self, calibrate: bool = True) -> None:
        self.meta_learner = LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            C=1.0,
        )
        self.calibrate = calibrate
        self.calibrator: CalibratedClassifierCV | None = None
        self._is_fitted = False

    def fit(
        self,
        prong_scores: ProngScores,
        labels: np.ndarray,
    ) -> dict[str, float]:
        """Fit the meta-learner on prong outputs.

        Args:
            prong_scores: Scores from all three prongs.
            labels: Binary labels (1 = interacted in next 7 days).

        Returns:
            Metrics including ensemble AUC and per-prong AUC for comparison.
        """
        X = prong_scores.to_matrix()
        y = labels

        # Fit meta-learner
        self.meta_learner.fit(X, y)

        if self.calibrate:
            self.calibrator = CalibratedClassifierCV(
                self.meta_learner, cv=3, method="sigmoid"  # Platt scaling
            )
            self.calibrator.fit(X, y)

        self._is_fitted = True

        # Compute metrics
        metrics = self._compute_metrics(prong_scores, labels)
        return metrics

    def predict_proba(self, prong_scores: ProngScores) -> np.ndarray:
        """Get final calibrated ensemble probabilities.

        Args:
            prong_scores: Scores from all three prongs.

        Returns:
            [N,] array of calibrated probabilities.
        """
        if not self._is_fitted:
            raise RuntimeError("Ensemble not fitted. Call fit() first.")

        X = prong_scores.to_matrix()

        if self.calibrator is not None:
            return self.calibrator.predict_proba(X)[:, 1]
        return self.meta_learner.predict_proba(X)[:, 1]

    def predict_with_breakdown(
        self,
        prong_scores: ProngScores,
    ) -> list[dict[str, float]]:
        """Predict with per-prong score breakdown for interpretability.

        Returns list of dicts with ensemble + individual prong scores.
        """
        ensemble_probs = self.predict_proba(prong_scores)
        results = []
        for i in range(len(ensemble_probs)):
            results.append({
                "ensemble_score": float(ensemble_probs[i]),
                "pyg_structural": float(prong_scores.pyg_scores[i, 0]),
                "dgl_temporal": float(prong_scores.dgl_scores[i, 0]),
                "xgb_tabular": float(prong_scores.xgb_scores[i, 0]),
            })
        return results

    def ablation_study(
        self,
        prong_scores: ProngScores,
        labels: np.ndarray,
    ) -> dict[str, float]:
        """Compare ensemble vs each single prong.

        This is the key validation: does the ensemble actually help?
        """
        results = {}

        # Full ensemble
        X_full = prong_scores.to_matrix()
        cv_full = cross_val_score(
            LogisticRegression(max_iter=1000), X_full, labels, cv=5, scoring="roc_auc"
        )
        results["ensemble_auc"] = float(cv_full.mean())
        results["ensemble_std"] = float(cv_full.std())

        # PyG only
        cv_pyg = cross_val_score(
            LogisticRegression(max_iter=1000),
            prong_scores.pyg_scores, labels, cv=5, scoring="roc_auc",
        )
        results["pyg_only_auc"] = float(cv_pyg.mean())

        # DGL only
        cv_dgl = cross_val_score(
            LogisticRegression(max_iter=1000),
            prong_scores.dgl_scores, labels, cv=5, scoring="roc_auc",
        )
        results["dgl_only_auc"] = float(cv_dgl.mean())

        # XGBoost only
        cv_xgb = cross_val_score(
            LogisticRegression(max_iter=1000),
            prong_scores.xgb_scores, labels, cv=5, scoring="roc_auc",
        )
        results["xgb_only_auc"] = float(cv_xgb.mean())

        # PyG + DGL (no tabular)
        X_gnn = np.hstack([prong_scores.pyg_scores, prong_scores.dgl_scores])
        cv_gnn = cross_val_score(
            LogisticRegression(max_iter=1000), X_gnn, labels, cv=5, scoring="roc_auc",
        )
        results["pyg_dgl_auc"] = float(cv_gnn.mean())

        # Lift calculations
        best_single = max(results["pyg_only_auc"], results["dgl_only_auc"], results["xgb_only_auc"])
        results["ensemble_lift_over_best_single"] = results["ensemble_auc"] - best_single
        results["ensemble_lift_over_gnn_only"] = results["ensemble_auc"] - results["pyg_dgl_auc"]

        return results

    def get_prong_weights(self) -> dict[str, float]:
        """Return learned weights for each prong from the meta-learner.

        Higher weight = that prong contributes more to final prediction.
        """
        if not self._is_fitted:
            raise RuntimeError("Ensemble not fitted.")

        coefs = self.meta_learner.coef_[0]
        return {
            "pyg_weight": float(coefs[0]),
            "dgl_weight": float(coefs[1]),
            "xgb_weight": float(coefs[2]),
            "intercept": float(self.meta_learner.intercept_[0]),
        }

    def _compute_metrics(
        self,
        prong_scores: ProngScores,
        labels: np.ndarray,
    ) -> dict[str, float]:
        """Compute AUC for ensemble and each prong."""
        ensemble_probs = self.predict_proba(prong_scores)

        metrics = {
            "ensemble_auc": float(roc_auc_score(labels, ensemble_probs)),
            "pyg_auc": float(roc_auc_score(labels, prong_scores.pyg_scores.ravel())),
            "dgl_auc": float(roc_auc_score(labels, prong_scores.dgl_scores.ravel())),
            "xgb_auc": float(roc_auc_score(labels, prong_scores.xgb_scores.ravel())),
        }

        weights = self.get_prong_weights()
        metrics.update(weights)

        return metrics
