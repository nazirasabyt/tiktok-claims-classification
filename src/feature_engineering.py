import pandas as pd


def engineer_features(df, save_path="data/processed/tiktok_features.csv"):
    data = df.copy()

    # ── Engagement ratio features ─────────────────────────────────────────────
    # Raw counts are scale-dependent. Ratios normalise for video size,
    # giving the model a signal for quality rather than popularity.
    data['likes_per_view']    = data['video_like_count']    / data['video_view_count']
    data['comments_per_view'] = data['video_comment_count'] / data['video_view_count']
    data['shares_per_view']   = data['video_share_count']   / data['video_view_count']
    data['downloads_per_view']= data['video_download_count']/ data['video_view_count']

    # ── Categorical encoding ──────────────────────────────────────────────────
    data['claim_status_binary'] = (data['claim_status'] == 'claim').astype(int)
    data['verified_binary']     = (data['verified_status'] == 'verified').astype(int)

    ban_order = {'active': 0, 'under review': 1, 'banned': 2}
    data['ban_status_encoded'] = data['author_ban_status'].map(ban_order)

    # ── Duration category ─────────────────────────────────────────────────────
    data['duration_category'] = pd.cut(
        data['video_duration_sec'],
        bins=[0, 20, 40, float('inf')],
        labels=['short', 'medium', 'long']
    )
    data['duration_encoded'] = data['duration_category'].cat.codes

    feature_cols = [
        'likes_per_view', 'comments_per_view', 'shares_per_view', 'downloads_per_view',
        'claim_status_binary', 'verified_binary', 'ban_status_encoded', 'duration_encoded'
    ]

    print("Feature summary (grouped by claim status):")
    print(data.groupby('claim_status')[feature_cols].agg(['mean', 'median']).round(4))

    print("\nNull counts in engineered features:")
    print(data[feature_cols].isnull().sum())

    # ── Save ML-ready dataset ─────────────────────────────────────────────────
    data.to_csv(save_path, index=False)
    print(f"\n  ML-ready dataset saved → {save_path}  ({data.shape[0]:,} rows × {data.shape[1]} cols)")

    return data


if __name__ == "__main__":
    df = pd.read_csv("data/raw/tiktok_dataset.csv")
    engineer_features(df)
