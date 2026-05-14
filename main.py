"""
Run the full TikTok Claims Classification pipeline:

  Raw Data
     ↓
  load  →  inspect  →  clean  →  features  →  analysis  →  visualizations
     ↓
  outputs saved automatically to data/processed/ and reports/figures/

Usage (from project root):
  python main.py
"""

import os
import time

from src.load_data          import load_data
from src.inspect_data       import inspect_data
from src.clean_data         import clean_data
from src.feature_engineering import engineer_features
from src.engagement_analysis import analyze_engagement
from src.visualizations     import generate_visualizations


def step(label):
    print(f"\n{'─' * 60}")
    print(f"  {label}")
    print(f"{'─' * 60}")


def main():
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("reports/figures", exist_ok=True)

    start = time.time()

    step("STEP 1 — Load data")
    df = load_data("data/raw/tiktok_dataset.csv")

    step("STEP 2 — Inspect data")
    inspect_data(df)

    step("STEP 3 — Clean data")
    df = clean_data(df)

    step("STEP 4 — Feature engineering")
    df = engineer_features(df, save_path="data/processed/tiktok_features.csv")

    step("STEP 5 — Engagement analysis")
    analyze_engagement(df)

    step("STEP 6 — Visualizations")
    generate_visualizations(df, out="reports/figures")

    elapsed = time.time() - start
    print(f"\n{'=' * 60}")
    print(f"  Pipeline complete in {elapsed:.1f}s")
    print(f"  Processed data → data/processed/tiktok_features.csv")
    print(f"  Figures        → reports/figures/")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()


# python3 main.py