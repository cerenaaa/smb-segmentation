"""Synthetic SMB account dataset for Defender campaign targeting."""
import numpy as np
import pandas as pd

INDUSTRIES = ["Retail", "Healthcare", "Legal", "Real Estate", "Construction",
              "Food & Beverage", "Professional Services", "Education"]
REGIONS = ["NA", "EMEA", "APAC", "LATAM"]


def generate_smb_accounts(n: int = 15_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame()
    df["account_id"] = [f"SMB_{i:06d}" for i in range(n)]
    df["industry"] = rng.choice(INDUSTRIES, n)
    df["region"] = rng.choice(REGIONS, n, p=[0.50, 0.25, 0.15, 0.10])
    df["employee_count"] = rng.integers(5, 300, n)
    df["monthly_charges"] = rng.uniform(50, 500, n).round(2)
    df["m365_license_tier"] = rng.choice(["Basic", "Business", "Business Premium", "E3"], n,
                                          p=[0.30, 0.35, 0.25, 0.10])

    # Security posture
    df["has_mfa_enabled"] = rng.binomial(1, 0.55, n)
    df["security_score"] = rng.beta(2, 3, n) * 100  # 0–100
    df["phishing_incidents_12m"] = rng.poisson(0.8, n)
    df["unmanaged_devices_pct"] = rng.beta(2, 4, n).round(3)

    # Engagement
    df["email_open_rate"] = rng.beta(2, 5, n).round(3)
    df["webinar_attended"] = rng.binomial(1, 0.12, n)
    df["days_since_last_activity"] = rng.integers(0, 120, n)
    df["support_tickets_6m"] = rng.poisson(1.2, n)
    df["feature_usage_pct"] = rng.beta(3, 3, n).round(3)
    df["renewal_days_out"] = rng.integers(-60, 365, n)

    # Ground truth: campaign conversion
    logit = (
        -2.8
        + 1.5 * (1 - df["has_mfa_enabled"])          # no MFA = higher security pain
        + 0.03 * df["phishing_incidents_12m"]
        + 0.4 * df["unmanaged_devices_pct"] * 3
        - 0.01 * df["security_score"] / 10
        + 0.8 * df["webinar_attended"]
        + 2.0 * df["email_open_rate"]
        + 0.3 * df["feature_usage_pct"]
        - 0.005 * df["days_since_last_activity"]
        + (df["renewal_days_out"].between(0, 90)).astype(float) * 0.5
        + rng.normal(0, 0.5, n)
    )
    prob = 1 / (1 + np.exp(-logit))
    df["converted"] = (rng.uniform(size=n) < prob).astype(int)
    print(f"Generated {n:,} SMB accounts | Conversion rate: {df['converted'].mean():.1%}")
    return df


if __name__ == "__main__":
    df = generate_smb_accounts()
    df.to_csv("data/smb_accounts.csv", index=False)