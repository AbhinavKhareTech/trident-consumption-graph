"""Ablation study: ensemble vs single-prong comparison.

Re-exports TridentEnsemble.ablation_study and adds reporting utilities.
"""
from __future__ import annotations
import numpy as np
from bgi_trident.graph.ensemble.stacker import ProngScores, TridentEnsemble


def run_ablation(prong_scores: ProngScores, labels: np.ndarray) -> dict[str, float]:
    """Run full ablation study and return results."""
    ensemble = TridentEnsemble(calibrate=False)
    return ensemble.ablation_study(prong_scores, labels)


def format_ablation_report(results: dict[str, float]) -> str:
    """Format ablation results as readable report."""
    lines = ["=== BGI Trident Ablation Study ===", ""]
    lines.append(f"Ensemble AUC:     {results.get('ensemble_auc', 0):.4f} (+/- {results.get('ensemble_std', 0):.4f})")
    lines.append(f"PyG only AUC:     {results.get('pyg_only_auc', 0):.4f}")
    lines.append(f"DGL only AUC:     {results.get('dgl_only_auc', 0):.4f}")
    lines.append(f"XGBoost only AUC: {results.get('xgb_only_auc', 0):.4f}")
    lines.append(f"PyG+DGL AUC:      {results.get('pyg_dgl_auc', 0):.4f}")
    lines.append("")
    lines.append(f"Ensemble lift over best single: {results.get('ensemble_lift_over_best_single', 0):+.4f}")
    lines.append(f"Ensemble lift over GNN-only:    {results.get('ensemble_lift_over_gnn_only', 0):+.4f}")
    return "\n".join(lines)
