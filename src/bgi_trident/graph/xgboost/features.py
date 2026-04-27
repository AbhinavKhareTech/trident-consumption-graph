"""Feature engineering utilities for Prong 3.

Re-exports XGBoostFeatureExtractor from model.py and adds
convenience functions for feature pipeline construction.
"""
from __future__ import annotations

import pandas as pd

from bgi_trident.graph.xgboost.model import XGBoostFeatureExtractor


def build_feature_pipeline(food_orders: pd.DataFrame, instamart_orders: pd.DataFrame,
                           dineout_bookings: pd.DataFrame) -> pd.DataFrame:
    """Build complete tabular feature set for all user-entity pairs."""
    extractor = XGBoostFeatureExtractor(food_orders, instamart_orders, dineout_bookings)
    food_features = extractor.extract_food_features()
    instamart_features = extractor.extract_instamart_features()
    return pd.concat([food_features, instamart_features], ignore_index=True)
