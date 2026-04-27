"""Prong 3: XGBoost on hand-engineered consumption features.

Gradient-boosted trees on tabular features that GNNs are not great at:
- Sharp thresholds (user never orders after 10 PM)
- Multiplicative interactions (high AOV + weekend + group > 3 = dineout signal)
- Tabular patterns that message-passing architectures smooth over

This prong handles the signals that fall through the cracks of both
PyG (structural) and DGL (temporal) graph models.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    import xgboost as xgb
except ImportError:
    xgb = None  # type: ignore[assignment]

from sklearn.model_selection import train_test_split


@dataclass
class ConsumptionFeatures:
    """Hand-engineered features for a (user, entity) pair."""

    # Frequency signals
    order_count: float
    frequency_per_week: float
    span_days: float

    # Recency signals
    recency_days: float
    days_since_first: float

    # Monetary signals
    avg_spend: float
    total_spend: float
    max_single_order: float

    # Temporal patterns
    pct_weekday: float
    pct_weekend: float
    pct_lunch: float  # 11:00-14:00
    pct_dinner: float  # 19:00-22:00
    most_common_hour: int
    most_common_dow: int

    # Basket patterns
    avg_basket_size: float
    basket_diversity: float  # Unique items / total items

    # Reorder signals (Instamart specific)
    avg_reorder_interval: float
    reorder_regularity: float  # Std dev of intervals (lower = more regular)

    # Cross-domain signals
    has_paired_instamart: bool
    has_followed_dineout: bool
    cross_domain_frequency: float


class XGBoostFeatureExtractor:
    """Extract tabular consumption features from raw interaction data."""

    def __init__(self, food_orders: pd.DataFrame, instamart_orders: pd.DataFrame,
                 dineout_bookings: pd.DataFrame) -> None:
        self.food = food_orders.copy()
        self.instamart = instamart_orders.copy()
        self.dineout = dineout_bookings.copy()

        # Parse timestamps
        for df in [self.food, self.instamart, self.dineout]:
            if "timestamp" in df.columns:
                df["ts"] = pd.to_datetime(df["timestamp"])
                df["hour"] = df["ts"].dt.hour
                df["dow"] = df["ts"].dt.dayofweek

    def extract_food_features(self) -> pd.DataFrame:
        """Extract features for (user, restaurant) pairs."""
        now = pd.Timestamp.now()
        features = []

        for (uid, rid), group in self.food.groupby(["user_id", "restaurant_id"]):
            n = len(group)
            span = max((group["ts"].max() - group["ts"].min()).days, 1)
            recency = (now - group["ts"].max()).days

            # Temporal distribution
            n_weekday = (group["dow"] < 5).sum()
            n_lunch = group["hour"].between(11, 14).sum()
            n_dinner = group["hour"].between(19, 22).sum()

            # Cross-domain: check if user has instamart orders on same day
            food_dates = set(group["ts"].dt.date)
            insta_user = self.instamart[self.instamart["user_id"] == uid]
            insta_dates = set(insta_user["ts"].dt.date) if not insta_user.empty else set()
            paired_days = food_dates & insta_dates

            # Cross-domain: check if dineout booking within 48h
            dine_user = self.dineout[self.dineout["user_id"] == uid]
            has_dineout_follow = False
            if not dine_user.empty:
                for _, frow in group.iterrows():
                    gaps = (dine_user["ts"] - frow["ts"]).dt.total_seconds() / 3600
                    if ((gaps > 0) & (gaps <= 48)).any():
                        has_dineout_follow = True
                        break

            features.append({
                "user_id": uid,
                "entity_id": rid,
                "entity_type": "restaurant",
                "order_count": n,
                "frequency_per_week": n / max(span / 7, 1),
                "span_days": span,
                "recency_days": recency,
                "avg_spend": group["amount"].mean(),
                "total_spend": group["amount"].sum(),
                "max_single_order": group["amount"].max(),
                "pct_weekday": n_weekday / n,
                "pct_weekend": 1 - (n_weekday / n),
                "pct_lunch": n_lunch / n,
                "pct_dinner": n_dinner / n,
                "most_common_hour": group["hour"].mode().iloc[0] if n > 0 else 0,
                "most_common_dow": group["dow"].mode().iloc[0] if n > 0 else 0,
                "has_paired_instamart": len(paired_days) > 0,
                "has_followed_dineout": has_dineout_follow,
                "cross_domain_frequency": len(paired_days) / max(n, 1),
            })

        return pd.DataFrame(features)

    def extract_instamart_features(self) -> pd.DataFrame:
        """Extract features for (user, product) pairs with reorder signals."""
        now = pd.Timestamp.now()
        features = []

        for (uid, pid), group in self.instamart.groupby(["user_id", "product_id"]):
            n = len(group)
            span = max((group["ts"].max() - group["ts"].min()).days, 1)
            recency = (now - group["ts"].max()).days

            # Reorder interval analysis
            if n >= 2:
                intervals = group["ts"].sort_values().diff().dt.days.dropna()
                avg_interval = intervals.mean()
                interval_std = intervals.std()
            else:
                avg_interval = 0.0
                interval_std = 0.0

            features.append({
                "user_id": uid,
                "entity_id": pid,
                "entity_type": "product",
                "order_count": n,
                "frequency_per_week": n / max(span / 7, 1),
                "span_days": span,
                "recency_days": recency,
                "avg_spend": group["amount"].mean(),
                "total_spend": group["amount"].sum(),
                "avg_reorder_interval": avg_interval,
                "reorder_regularity": 1.0 / max(interval_std, 0.1),  # Higher = more regular
            })

        return pd.DataFrame(features)


class XGBoostProng:
    """Complete Prong 3: Feature extraction + XGBoost model.

    Trains a gradient-boosted classifier to predict whether a user
    will interact with an entity in the next 7 days.
    """

    FEATURE_COLS = [
        "order_count", "frequency_per_week", "span_days", "recency_days",
        "avg_spend", "total_spend", "pct_weekday", "pct_weekend",
        "pct_lunch", "pct_dinner", "most_common_hour", "most_common_dow",
        "has_paired_instamart", "has_followed_dineout", "cross_domain_frequency",
    ]

    def __init__(self, params: dict | None = None) -> None:
        if xgb is None:
            raise ImportError("xgboost is required for Prong 3")

        self.params = params or {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 200,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "seed": 42,
        }
        self.model: xgb.XGBClassifier | None = None

    def train(
        self,
        features_df: pd.DataFrame,
        labels: np.ndarray,
        val_size: float = 0.2,
    ) -> dict[str, float]:
        """Train XGBoost classifier.

        Args:
            features_df: DataFrame with FEATURE_COLS columns.
            labels: Binary labels (1 = interacted in next 7 days).
            val_size: Validation split ratio.

        Returns:
            Dictionary with train and validation AUC scores.
        """
        available_cols = [c for c in self.FEATURE_COLS if c in features_df.columns]
        X = features_df[available_cols].fillna(0).values
        y = labels

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=val_size, random_state=42, stratify=y
        )

        self.model = xgb.XGBClassifier(**self.params)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        train_score = self.model.score(X_train, y_train)
        val_score = self.model.score(X_val, y_val)

        return {"train_accuracy": train_score, "val_accuracy": val_score}

    def predict_proba(self, features_df: pd.DataFrame) -> np.ndarray:
        """Get Prong 3 probability scores.

        Returns:
            Array of shape [N, 1] with predicted probabilities.
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        available_cols = [c for c in self.FEATURE_COLS if c in features_df.columns]
        X = features_df[available_cols].fillna(0).values
        return self.model.predict_proba(X)[:, 1].reshape(-1, 1)

    def feature_importance(self) -> dict[str, float]:
        """Return feature importance ranking."""
        if self.model is None:
            raise RuntimeError("Model not trained.")
        importance = self.model.feature_importances_
        available_cols = [c for c in self.FEATURE_COLS if c in (self.model.feature_names_in_ or [])]
        if not available_cols:
            available_cols = self.FEATURE_COLS[:len(importance)]
        return dict(sorted(
            zip(available_cols, importance), key=lambda x: x[1], reverse=True
        ))
