import pandas as pd


def analyze_engagement(df):
    data = df.copy()

    # ── Part 1: Claims vs Opinions ────────────────────────────────────────────
    print("=" * 60)
    print("PART 1: Claims vs Opinions — Average Engagement")
    print("=" * 60)

    engagement_by_claim = data.groupby('claim_status').agg(
        avg_views=   ('video_view_count',    'mean'),
        avg_likes=   ('video_like_count',    'mean'),
        avg_comments=('video_comment_count', 'mean'),
        avg_shares=  ('video_share_count',   'mean'),
    ).round(2)
    print(engagement_by_claim)

    # ── Part 2: GroupBy Analysis ──────────────────────────────────────────────

    print("\n" + "=" * 60)
    print("PART 2a: Engagement by Verification Status")
    print("=" * 60)

    by_verified = data.groupby('verified_status').agg(
        avg_views=   ('video_view_count',    'mean'),
        avg_likes=   ('video_like_count',    'mean'),
        avg_comments=('video_comment_count', 'mean'),
        avg_shares=  ('video_share_count',   'mean'),
    ).round(2)
    print(by_verified)

    print("\n" + "=" * 60)
    print("PART 2b: Engagement by Author Ban Status")
    print("=" * 60)

    by_ban = data.groupby('author_ban_status').agg(
        count=        ('video_view_count', 'count'),
        avg_views=    ('video_view_count', 'mean'),
        median_views= ('video_view_count', 'median'),
        avg_likes=    ('video_like_count', 'mean'),
        median_likes= ('video_like_count', 'median'),
        avg_shares=   ('video_share_count','mean'),
        median_shares=('video_share_count','median'),
    ).round(2)
    print(by_ban)

    print("\n" + "=" * 60)
    print("PART 2c: Engagement by Claim Status × Author Ban Status")
    print("=" * 60)

    by_claim_ban = data.groupby(['claim_status', 'author_ban_status']).agg(
        count=    ('video_view_count', 'count'),
        avg_views=('video_view_count', 'mean'),
        avg_likes=('video_like_count', 'mean'),
        avg_shares=('video_share_count','mean'),
    ).round(2)
    print(by_claim_ban)

    print("\n" + "=" * 60)
    print("PART 2d: Engagement by Video Duration Category")
    print("=" * 60)

    data['duration_category'] = pd.cut(
        data['video_duration_sec'],
        bins=[0, 20, 40, float('inf')],
        labels=['short', 'medium', 'long']
    )

    by_duration = data.groupby('duration_category', observed=True).agg(
        count=    ('video_view_count',  'count'),
        avg_views=('video_view_count',  'mean'),
        avg_shares=('video_share_count','mean'),
        avg_likes=('video_like_count',  'mean'),
    ).round(2)
    print(by_duration)

    # ── Part 3: Engagement Trends ─────────────────────────────────────────────

    print("\n" + "=" * 60)
    print("PART 3: Engagement Trends — Correlations")
    print("=" * 60)

    engagement_cols = [
        'video_view_count', 'video_like_count',
        'video_share_count', 'video_comment_count'
    ]
    correlations = data[engagement_cols].corr().round(3)
    print(correlations)

    print("\nKey insight — median share count by claim status:")
    print(data.groupby('claim_status')['video_share_count'].median().round(2))

    print("\nKey insight — median comment count by claim status:")
    print(data.groupby('claim_status')['video_comment_count'].median().round(2))


if __name__ == "__main__":
    df = pd.read_csv("data/raw/tiktok_dataset.csv")
    analyze_engagement(df)
