"""
K-Means clustering for SMB account segmentation.
Includes optimal K selection via silhouette + elbow, and cluster profiling.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


SEGMENT_FEATURES = [
    "security_score", "has_mfa_enabled", "phishing_incidents_12m",
    "unmanaged_devices_pct", "email_open_rate", "webinar_attended",
    "days_since_last_activity", "feature_usage_pct", "monthly_charges",
    "employee_count",
]

SEGMENT_NAMES = {
    0: "Security-Aware Growers",
    1: "High-Exposure Laggards",
    2: "Budget-Constrained",
    3: "Low-Fit Accounts",
}


class SMBSegmenter:
    def __init__(self, k: int = None, k_range: tuple = (2, 8), random_state: int = 42):
        self.k = k
        self.k_range = k_range
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = None
        self.feature_cols = None

    def select_k(self, X_scaled: np.ndarray) -> int:
        scores = {}
        for k in range(self.k_range[0], self.k_range[1] + 1):
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = km.fit_predict(X_scaled)
            scores[k] = silhouette_score(X_scaled, labels)
        best_k = max(scores, key=scores.get)
        print(f"Silhouette scores: { {k: f'{v:.3f}' for k, v in scores.items()} }")
        print(f"Selected K={best_k}")
        return best_k

    def fit(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in SEGMENT_FEATURES if c in df.columns]
        self.feature_cols = cols
        X = df[cols].fillna(df[cols].median())
        X_scaled = self.scaler.fit_transform(X)

        if self.k is None:
            self.k = self.select_k(X_scaled)

        self.model = KMeans(n_clusters=self.k, random_state=self.random_state, n_init=20)
        df = df.copy()
        df["segment"] = self.model.fit_predict(X_scaled)
        df["segment_name"] = df["segment"].map(SEGMENT_NAMES).fillna("Segment_" + df["segment"].astype(str))
        return df

    def profile(self, df: pd.DataFrame) -> pd.DataFrame:
        """Statistical profile of each cluster."""
        if "segment_name" not in df.columns:
            raise ValueError("Run fit() first.")
        profile = df.groupby("segment_name")[self.feature_cols + ["converted"]].agg(["mean", "count"])
        return profile