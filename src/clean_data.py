# clean_data.py

import pandas as pd


def clean_data(df):
    df = df.copy()
    df["verified_status"] = df["verified_status"].map({
        "verified": 1,
        "not verified": 0
    })
    print(f"  verified_status encoded: {df['verified_status'].value_counts().to_dict()}")
    return df


if __name__ == "__main__":
    df = pd.read_csv("data/raw/tiktok_dataset.csv")
    df = clean_data(df)
    print(df[["verified_status"]].head())
