"""Tests for XGBoost Prong 3 feature extraction."""
from bgi_trident.graph.xgboost.model import XGBoostFeatureExtractor


def test_food_feature_extraction(sample_food_orders, sample_instamart_orders, sample_dineout_bookings):
    extractor = XGBoostFeatureExtractor(sample_food_orders, sample_instamart_orders, sample_dineout_bookings)
    features = extractor.extract_food_features()
    assert len(features) > 0
    assert "order_count" in features.columns
    assert "recency_days" in features.columns
    assert "cross_domain_frequency" in features.columns
