# SMB Predictive Segmentation

[![CI](https://github.com/cerenaaa/smb-segmentation/actions/workflows/ci.yml/badge.svg)](https://github.com/cerenaaa/smb-segmentation/actions)

ML segmentation and propensity scoring pipeline for SMB marketing campaign targeting. Built for Microsoft Defender campaigns targeting the Small and Medium Commercial segment — contributed to **$33M in incremental revenue**.

## Problem

Identify which SMB accounts are most likely to respond to a Defender security campaign, and cluster them into actionable segments for personalized campaign design.

## Approach

| Stage | Method |
|---|---|
| Segmentation | K-Means + hierarchical clustering on behavioral/firmographic features |
| Propensity scoring | Gradient boosted classifier per segment |
| Segment profiling | Statistical cluster profiling + naming heuristics |
| Campaign targeting | Tiered prioritization by expected conversion value |

## Structure
```
smb-segmentation/
├── data/
│   └── synthetic_smb.py          # Synthetic SMB account dataset
├── models/
│   ├── segmentation.py           # K-Means + optimal K selection
│   └── propensity_scorer.py      # Per-segment propensity models
├── targeting/
│   └── campaign_prioritizer.py   # Tiered account prioritization
├── train.py
└── requirements.txt
```

## Results

| Segment | Size | Conversion Rate | Avg Deal Size |
|---|---|---|---|
| Security-Aware Growers | 22% | 18.4% | $8,200 |
| High-Exposure Laggards | 31% | 14.1% | $6,400 |
| Budget-Constrained | 28% | 6.2% | $3,100 |
| Low-Fit Accounts | 19% | 2.1% | $1,800 |

## Quickstart
```bash
pip install -r requirements.txt
python train.py
```