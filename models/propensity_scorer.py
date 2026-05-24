"""
Per-segment propensity scorer.
Trains a separate gradient boosted model for each customer segment,
allowing segment-specific feature importance and calibration.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
import joblib


PROPENSITY_FEATURES = [
    "security_score", "has_mfa_enabled", "phishing_incidents_12m",
    "unmanaged_devices_pct", "email_open_rate", "webinar_attended",
    "days_since_last_activity", "feature_usage_pct", "monthly_charges",
    "employee_count", "support_tickets_6m",
]


class SegmentPropensityScorer:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models: dict[str, CalibratedClassifierCV] = {}

    def fit(self, df: pd.DataFrame, segment_col: str = "segment_name", target: str = "converted"):
        feature_cols = [c for c in PROPENSITY_FEATURES if c in df.columns]
        for segment, group in df.groupby(segment_col):
            if group[target].nunique() < 2 or len(group) < 50:
                print(f"  Skipping {segment} (insufficient data)")
                continue
            X = group[feature_cols].fillna(group[feature_cols].median())
            y = group[target]
            base = GradientBoostingClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, random_state=self.random_state)
            calibrated = CalibratedClassifierCV(base, method="isotonic", cv=3)
            calibrated.fit(X, y)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
            scores = cross_val_score(base, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
            print(f"  {segment}: AUC={scores.mean():.3f}±{scores.std():.3f} (n={len(group):,})")
            self.models[segment] = (calibrated, feature_cols)
        return self

    def predict(self, df: pd.DataFrame, segment_col: str = "segment_name") -> pd.Series:
        probas = pd.Series(np.nan, index=df.index)
        for segment, (model, feature_cols) in self.models.items():
            mask = df[segment_col] == segment
            if mask.sum() == 0:
                continue
            X = df.loc[mask, feature_cols].fillna(df.loc[mask, feature_cols].median())
            probas.loc[mask] = model.predict_proba(X)[:, 1]
        return probas