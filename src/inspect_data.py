# inspect_data.py

import pandas as pd


def inspect_data(df):
    # -----------------------
    # BASIC OVERVIEW
    # -----------------------

    print("\n=== Shape ===")
    print(df.shape)
    print("\n=== COLUMNS ===")
    print(df.columns)

    print("\n=== DTYPES ===")
    print(df.dtypes)

    print("\n=== SAMPLE ROWS ===")
    print(df.head())

    # -----------------------
    # NULL VALUES
    # -----------------------

    print("\n=== NULL VALUES ===")
    print(df.isnull().sum())

    print("\n=== NULL VALUES PERCENTAGE ===")
    print((df.isnull().mean() * 100).round(2))

    # -----------------------
    # DUPLICATES
    # -----------------------
    print("\n=== DUPLICATES ===")
    print(df.duplicated().sum())

    # -----------------------
    # NUMERIC SUMMARY
    # -----------------------

    print("\n=== NUMERIC SUMMARY ===")
    print(df.describe())

    # -----------------------
    # CATEGORICAL SUMMARY
    # -----------------------

    categorical_cols = df.select_dtypes(include="object").columns

    for col in categorical_cols:
        print(f"\n=== VALUE COUNTS: {col} ===")
        print(df[col].value_counts().head(10))

    # -----------------------
    # OUTLIER CHECK (IQR METHOD)
    # -----------------------

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

    print("\n=== OUTLIER CHECK ===")

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outliers = df[(df[col] < lower) | (df[col] > upper)]

        print(f"{col}: {len(outliers)} outliers")


# Important DA Insight
# These are not necessarily "bad data".
# In social media datasets, high engagement outliers are often REAL.
# Removing them blindly can destroy important patterns.
# At this stage: no catastrophic issues, categorical columns clean,
# target variable balanced, engagement metrics skewed as expected.


if __name__ == "__main__":
    df = pd.read_csv("data/raw/tiktok_dataset.csv")
    inspect_data(df)
