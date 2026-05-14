import pandas as pd


def load_data(path="data/raw/tiktok_dataset.csv"):
    df = pd.read_csv(path)
    print(f"  Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


if __name__ == "__main__":
    df = load_data()
    print(df.head())
    print(df.shape)
    print(df.columns.tolist())


# python3 src/load_data.py

   # claim_status    video_id  video_duration_sec                           video_transcription_text  ... video_view_count video_like_count  video_share_count  video_download_count  video_comment_count
# 0  1        claim  7017666017                  59  someone shared with me that drone deliveries a...  ...         343296.0          19425.0              241.0                   1.0                  0.0
# 1  2        claim  4014381136                  32  someone shared with me that there are more mic...  ...         140877.0          77355.0            19034.0                1161.0                684.0
# 2  3        claim  9859838091                  31  someone shared with me that american industria...  ...         902185.0          97690.0             2858.0                 833.0                329.0
# 3  4        claim  1866847991                  25  someone shared with me that the metro of st. p...  ...         437506.0         239954.0            34812.0                1234.0                584.0
# 4  5        claim  7105231098                  19  someone shared with me that the number of busi...  ...          56167.0          34987.0             4110.0                 547.0                152.0

# [5 rows x 12 columns]
# (19382, 12)
# ['#', 
# 'claim_status',
#  'video_id', 
#  'video_duration_sec', 
# 'video_transcription_text', 
# 'verified_status', 
# 'author_ban_status', 
# 'video_view_count',
#  'video_like_count', 
# 'video_share_count', 
# 'video_download_count', 
# 'video_comment_count']