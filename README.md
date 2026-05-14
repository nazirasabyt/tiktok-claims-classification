# TikTok Claims Classification

This was a project from the Google Advanced Data Analytics Certificate where I worked through a TikTok trust & safety scenario — the goal being to figure out whether you can tell claim videos apart from opinion videos just by looking at how people engage with them.

Spoiler: you can, and the gap is much larger than I expected going in.

---

## The question

TikTok needs a way to flag videos making factual claims for moderation review, without manually watching everything. I wanted to see whether engagement patterns alone could do that work — before touching any model.

---

## What I found

The most striking thing was the raw scale of the difference. Claim videos average **501k views**. Opinion videos average **5k**. That's roughly 100× across the board for views, likes, shares, and comments.

But raw counts are noisy — popular videos might just be popular. So I normalised everything by view count, and the gap held:

- **+51% more likes per view** for claims
- **+180% more comments per view** for claims

Claims don't just reach more people — each viewer is more likely to interact with them.

The other thing that stood out was the ban status breakdown. Among **banned authors**, claims outnumber opinions 7 to 1. Among authors **under review**, 78% of their videos are claims. It looks like high engagement from claim content is exactly what triggers moderation review in the first place.

One thing that surprised me: **verified authors actually have lower engagement on average** — not because they're less influential, but because they post proportionally more opinion content. Verification status alone is basically useless as a signal here.

Video duration turned out to be a dead end too. Less than 3% variation across short, medium, and long videos.

---

## Visualisations

![Dashboard Overview](reports/figures/dashboard_overview.png)

![Engagement Ratios by Ban Status](reports/figures/engagement_ratios.png)

![Verified vs Not Verified](reports/figures/verified_comparison.png)

![Duration vs Engagement](reports/figures/duration_analysis.png)

More detail on all of this in [`reports/findings.md`](reports/findings.md), or the shorter version in [`reports/executive_summary.md`](reports/executive_summary.md).

---

## Features I'd take into modelling

Based on the EDA, I engineered these features for a downstream classifier:

| Feature | Why |
|---------|-----|
| `shares_per_view` | Best single separator between claims and opinions |
| `likes_per_view` | Consistent gap, not too correlated with shares/view |
| `comments_per_view` | The 180% gap makes this hard to ignore |
| `ban_status_encoded` | Banned/under-review accounts skew heavily claim |
| `verified_binary` | Weak, but adds something as a composition proxy |
| `duration_encoded` | Barely anything — only worth trying in tree-based models |

The processed dataset is at `data/processed/tiktok_features.csv` (19,382 rows × 20 columns).

---

## How to navigate this repo

```
├── main.py                     # runs the full pipeline end to end
├── src/
│   ├── load_data.py
│   ├── inspect_data.py
│   ├── clean_data.py
│   ├── feature_engineering.py
│   ├── engagement_analysis.py
│   └── visualizations.py
├── data/
│   ├── raw/tiktok_dataset.csv
│   └── processed/tiktok_features.csv
└── reports/
    ├── executive_summary.md
    ├── findings.md
    └── figures/
```

To run it yourself:

```bash
pip install -r requirements.txt
python main.py
```

---

## Stack

Python, pandas, matplotlib, seaborn
