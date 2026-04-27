"""Tests for ensemble meta-learner."""
import numpy as np

from bgi_trident.graph.ensemble.stacker import ProngScores, TridentEnsemble


def test_ensemble_fit_and_predict():
    n = 200
    np.random.seed(42)
    scores = ProngScores(
        pyg_scores=np.random.rand(n, 1),
        dgl_scores=np.random.rand(n, 1),
        xgb_scores=np.random.rand(n, 1),
    )
    labels = (scores.pyg_scores.ravel() + scores.dgl_scores.ravel() > 1.0).astype(int)
    ensemble = TridentEnsemble(calibrate=False)
    metrics = ensemble.fit(scores, labels)
    assert "ensemble_auc" in metrics
    preds = ensemble.predict_proba(scores)
    assert len(preds) == n
    assert all(0 <= p <= 1 for p in preds)


def test_ablation_study():
    n = 200
    np.random.seed(42)
    scores = ProngScores(
        pyg_scores=np.random.rand(n, 1),
        dgl_scores=np.random.rand(n, 1),
        xgb_scores=np.random.rand(n, 1),
    )
    labels = (scores.pyg_scores.ravel() + scores.xgb_scores.ravel() > 1.0).astype(int)
    ensemble = TridentEnsemble(calibrate=False)
    results = ensemble.ablation_study(scores, labels)
    assert "ensemble_auc" in results
    assert "pyg_only_auc" in results
    assert "ensemble_lift_over_best_single" in results
