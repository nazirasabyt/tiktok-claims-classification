# Executive Summary — TikTok Claims Classification

## Project Goal

Identify behavioural and engagement patterns that distinguish **claim videos** (content making factual assertions) from **opinion videos** on TikTok, using a dataset of ~19,000 videos. Findings inform feature selection for a downstream classification model.

---

## Dataset

- **19,382 videos** · 12 raw features · near-balanced target (50.4% claim / 49.6% opinion)
- 298 rows (1.5%) had missing engagement values and were excluded from aggregations
- Categorical variables — `claim_status`, `verified_status`, `author_ban_status` — were clean with no encoding issues

---

## Key Findings

![Engagement Dashboard](figures/dashboard_overview.png)

### 1. Claims generate ~100× more raw engagement than opinions

Claim videos average **501k views**, 171k likes, and 33k shares. Opinion videos average **5k views**, 1k likes, and 218 shares. The median share gap (18k vs 121) confirms this is not driven by a handful of outliers — it is a dataset-wide pattern.

### 2. Claims are proportionally more engaging, even per view

After normalising by view count, claim videos still attract **51% more likes per view** and **51% more shares per view** than opinion videos. Viewers who encounter claim content are measurably more likely to interact with it.

### 3. Banned and under-review authors overwhelmingly post claims

Among **banned** authors, claims outnumber opinions **7:1**. Among authors **under review**, claims represent 78% of videos vs 34% for active accounts. High engagement appears to be a direct precursor to moderation action.

### 4. Verified status is not a reliable signal

Verified authors actually produce **lower engagement** on average — not because they are less influential, but because they post proportionally more opinion content. Verification alone cannot separate claims from opinions.

### 5. Video duration is not predictive

Engagement differences across short (≤20s), medium (21–40s), and long (>40s) videos are under 3% — duration adds minimal discriminating power on its own.

---

## Visualisations

![Engagement Ratios by Ban Status](figures/engagement_ratios.png)

*Likes, comments, and shares per view broken down by claim status and author ban category. The claim advantage holds across all moderation tiers.*

![Verified vs Not Verified Authors](figures/verified_comparison.png)

*Average engagement and engagement ratios by verification status. Not-verified authors dominate on every metric due to their disproportionate share of claim content.*

![Duration vs Engagement](figures/duration_analysis.png)

*Avg views, likes, and shares by video duration bucket. Minimal variation across categories confirms duration is a weak predictor.*

---

## Recommended Features for Modelling

| Feature | Signal Strength | Rationale |
|---------|----------------|-----------|
| `shares_per_view` | Strong | Best single-ratio separator between claims and opinions |
| `likes_per_view` | Strong | Consistent 51% gap, low collinearity with shares/view |
| `comments_per_view` | Strong | 180% gap; discussion intensity is a strong claim marker |
| `ban_status_encoded` | Moderate | Banned/under-review accounts skew heavily toward claims |
| `verified_binary` | Weak | Adds marginal signal as a composition proxy |
| `duration_encoded` | Weak | <3% variation; include only in tree-based models |

---

## Conclusion

The strongest predictor of claim status is **per-view engagement** — claims attract disproportionately more likes, shares, and comments for every viewer they reach. Raw view counts alone are noisy (claims are popular, but not all popular videos are claims). Author ban status serves as a useful secondary feature. These findings support building a classifier on normalised engagement ratios combined with author moderation signals.
