"""
Train segmentation + propensity models.
Usage: python train.py
"""
from data.synthetic_smb import generate_smb_accounts
from models.segmentation import SMBSegmenter
from models.propensity_scorer import SegmentPropensityScorer
from targeting.campaign_prioritizer import prioritize

def main():
    print("Generating SMB data...")
    df = generate_smb_accounts(n=15_000)

    print("\nFitting segmentation model...")
    segmenter = SMBSegmenter()
    df = segmenter.fit(df)

    print("\nSegment profile:")
    profile = segmenter.profile(df)
    print(profile["converted"]["mean"].to_string())

    print("\nFitting per-segment propensity models...")
    scorer = SegmentPropensityScorer()
    scorer.fit(df)
    df["propensity_score"] = scorer.predict(df)

    print("\nPrioritizing campaign targets...")
    targets = prioritize(df, top_n=5_000)
    targets.to_csv("results/campaign_targets.csv", index=False)
    print(f"\n✓ {len(targets):,} accounts prioritized → results/campaign_targets.csv")

if __name__ == "__main__":
    import pathlib; pathlib.Path("results").mkdir(exist_ok=True)
    main()