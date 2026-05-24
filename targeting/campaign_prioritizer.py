"""
Campaign prioritization: tiers accounts by expected conversion value.
Outputs a ranked target list with segment, propensity score, and tier.
"""
import pandas as pd
import numpy as np


def prioritize(
    df: pd.DataFrame,
    propensity_col: str = "propensity_score",
    segment_col: str = "segment_name",
    deal_size_col: str = "monthly_charges",
    top_n: int = 5000,
) -> pd.DataFrame:
    """
    Expected value = propensity_score × estimated deal size.
    Tier 1: top quintile EV → high-touch outreach
    Tier 2: 60–80th pct → automated nurture
    Tier 3: below 60th pct → email-only
    """
    df = df.copy()
    df["expected_value"] = df[propensity_col] * df[deal_size_col]
    df["priority_tier"] = pd.qcut(
        df["expected_value"],
        q=[0, 0.6, 0.8, 1.0],
        labels=["Tier 3 – Email Only", "Tier 2 – Nurture", "Tier 1 – High Touch"]
    )
    ranked = df.sort_values("expected_value", ascending=False).head(top_n)
    tier_summary = ranked.groupby("priority_tier").agg(
        n=("expected_value", "count"),
        avg_propensity=(propensity_col, "mean"),
        avg_expected_value=("expected_value", "mean"),
    )
    print("\nCampaign Tier Summary:")
    print(tier_summary.to_string(float_format="{:.3f}".format))
    return ranked