# build_features.py

"""
CommissionLens
------------------------------------------------------------

STEP 3:
Build ML-ready feature dataset.

This script:
1. Loads master NAV dataset
2. Loads benchmark data
3. Merges benchmark returns
4. Computes rolling financial features
5. Computes alpha, beta, Sharpe, IR
6. Creates quarterly snapshots
7. Creates future prediction target
8. Saves final ML dataset

Author: Prateek Dixit
"""

import pandas as pd
import numpy as np
from pathlib import Path


# =========================================================
# PATHS
# =========================================================

DATA_DIR = Path("data")

PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_DIR = DATA_DIR / "external"

NAV_PATH = (
    PROCESSED_DIR /
    "master_nav_dataset.csv"
)

BENCHMARK_PATH = (
    EXTERNAL_DIR /
    "nifty50.csv"
)

FINAL_DATASET_PATH = (
    PROCESSED_DIR /
    "final_model_dataset.csv"
)


# =========================================================
# LOAD NAV DATA
# =========================================================

def load_nav_data():

    df = pd.read_csv(NAV_PATH)

    df["date"] = pd.to_datetime(df["date"])

    print("NAV dataset loaded")
    print(df.shape)

    return df


# =========================================================
# LOAD BENCHMARK DATA
# =========================================================

def load_benchmark_data():

    benchmark_df = pd.read_csv(
        BENCHMARK_PATH
    )

    print("\nBenchmark columns:")
    print(benchmark_df.columns.tolist())

    # -----------------------------------------------------
    # Rename columns depending on source
    # -----------------------------------------------------

    rename_map = {}

    for col in benchmark_df.columns:

        col_lower = col.lower()

        if "date" in col_lower:
            rename_map[col] = "date"

        elif (
            "close" in col_lower
            or "price" in col_lower
        ):
            rename_map[col] = "close"

    benchmark_df.rename(
        columns=rename_map,
        inplace=True
    )

    # -----------------------------------------------------
    # Keep required columns
    # -----------------------------------------------------

    benchmark_df = benchmark_df[
        ["date", "close"]
    ]

    # -----------------------------------------------------
    # Convert types
    # -----------------------------------------------------

    benchmark_df["date"] = pd.to_datetime(
        benchmark_df["date"]
    )

    benchmark_df["close"] = (
        benchmark_df["close"]
        .astype(str)
        .str.replace(",", "")
        .astype(float)
    )

    benchmark_df.sort_values(
        "date",
        inplace=True
    )

    benchmark_df.reset_index(
        drop=True,
        inplace=True
    )

    # -----------------------------------------------------
    # Benchmark returns
    # -----------------------------------------------------

    benchmark_df["benchmark_return"] = (
        benchmark_df["close"]
        .pct_change()
    )

    print("\nBenchmark dataset loaded")
    print(benchmark_df.shape)

    return benchmark_df


# =========================================================
# MERGE BENCHMARK
# =========================================================

def merge_benchmark(nav_df, benchmark_df):

    benchmark_df = benchmark_df[
        ["date", "benchmark_return"]
    ]

    merged_df = pd.merge(
        nav_df,
        benchmark_df,
        on="date",
        how="inner"
    )

    print("\nMerged dataset shape:")
    print(merged_df.shape)

    return merged_df


# =========================================================
# ROLLING RETURNS
# =========================================================

def compute_rolling_return(df):

    df["rolling_252d_return"] = (
        df.groupby("schemecode")[
            "daily_return"
        ]
        .transform(
            lambda x:
            x.rolling(252)
             .mean()
             .shift(1)
        )
    )

    return df


# =========================================================
# VOLATILITY
# =========================================================

def compute_volatility(df):

    df["rolling_volatility"] = (
        df.groupby("schemecode")[
            "daily_return"
        ]
        .transform(
            lambda x:
            x.rolling(252)
             .std()
             .shift(1)
        )
    )

    return df


# =========================================================
# SHARPE RATIO
# =========================================================

def compute_sharpe_ratio(df):

    risk_free_rate = 0.06

    daily_rf = risk_free_rate / 252

    df["rolling_sharpe"] = (
        (
            df["rolling_252d_return"]
            - daily_rf
        )
        /
        df["rolling_volatility"]
    )

    return df


# =========================================================
# BETA
# =========================================================

def compute_beta(df):

    def rolling_beta(group):

        covariance = (
            group["daily_return"]
            .rolling(252)
            .cov(group["benchmark_return"])
        )

        benchmark_variance = (
            group["benchmark_return"]
            .rolling(252)
            .var()
        )

        beta = covariance / benchmark_variance

        return beta.shift(1)

    df["beta"] = (
        df.groupby("schemecode")
        .apply(rolling_beta)
        .reset_index(level=0, drop=True)
    )

    return df


# =========================================================
# ALPHA
# =========================================================

def compute_alpha(df):

    df["alpha"] = (
        df["daily_return"]
        -
        (
            df["beta"]
            * df["benchmark_return"]
        )
    )

    return df


# =========================================================
# INFORMATION RATIO
# =========================================================

def compute_information_ratio(df):

    excess_return = (
        df["daily_return"]
        -
        df["benchmark_return"]
    )

    tracking_error = (
        excess_return
        .rolling(252)
        .std()
    )

    df["information_ratio"] = (
        excess_return
        .rolling(252)
        .mean()
        /
        tracking_error
    )

    return df


# =========================================================
# EXPENSE RATIO GAP
# =========================================================

def add_expense_gap(df):

    """
    Placeholder expense gap.

    Later:
    Replace with actual TER data.
    """

    df["expense_gap"] = 0.01

    return df


# =========================================================
# NET ALPHA
# =========================================================

def compute_net_alpha(df):

    daily_expense = (
        df["expense_gap"] / 252
    )

    df["net_alpha"] = (
        df["alpha"]
        - daily_expense
    )

    return df


# =========================================================
# MOMENTUM
# =========================================================

def compute_momentum(df):

    df["momentum_90d"] = (
        df.groupby("schemecode")[
            "nav"
        ]
        .pct_change(90)
        .shift(1)
    )

    return df


# =========================================================
# QUARTERLY SNAPSHOTS
# =========================================================

def create_quarterly_snapshots(df):

    df["quarter"] = (
        df["date"]
        .dt.to_period("Q")
    )

    quarterly_df = (
        df.groupby(
            ["schemecode", "quarter"]
        )
        .tail(1)
        .copy()
    )

    quarterly_df.reset_index(
        drop=True,
        inplace=True
    )

    print("\nQuarterly snapshot dataset:")
    print(quarterly_df.shape)

    return quarterly_df


# =========================================================
# FUTURE TARGET
# =========================================================

def create_target_variable(df):

    df = df.sort_values(
        ["schemecode", "date"]
    )

    # ---------------------------------------------
    # Future 1-quarter net alpha
    # ---------------------------------------------

    df["future_net_alpha"] = (
        df.groupby("schemecode")[
            "net_alpha"
        ]
        .shift(-1)
    )

    # ---------------------------------------------
    # Classification target
    # ---------------------------------------------

    df["commission_justified"] = np.where(
        df["future_net_alpha"] > 0,
        1,
        0
    )

    return df


# =========================================================
# CLEAN DATASET
# =========================================================

def clean_dataset(df):

    df.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )

    required_cols = [
        "rolling_252d_return",
        "rolling_volatility",
        "rolling_sharpe",
        "beta",
        "alpha",
        "information_ratio",
        "net_alpha",
        "momentum_90d",
        "commission_justified"
    ]

    df.dropna(
        subset=required_cols,
        inplace=True
    )

    df.reset_index(
        drop=True,
        inplace=True
    )

    print("\nFinal cleaned dataset:")
    print(df.shape)

    return df


# =========================================================
# SAVE DATASET
# =========================================================

def save_dataset(df):

    df.to_csv(
        FINAL_DATASET_PATH,
        index=False
    )

    print("\nFinal dataset saved:")
    print(FINAL_DATASET_PATH)


# =========================================================
# MAIN PIPELINE
# =========================================================

def main():

    # -----------------------------------------------------
    # Load datasets
    # -----------------------------------------------------

    nav_df = load_nav_data()

    benchmark_df = load_benchmark_data()

    # -----------------------------------------------------
    # Merge benchmark
    # -----------------------------------------------------

    merged_df = merge_benchmark(
        nav_df,
        benchmark_df
    )

    # -----------------------------------------------------
    # Feature engineering
    # -----------------------------------------------------

    merged_df = compute_rolling_return(
        merged_df
    )

    merged_df = compute_volatility(
        merged_df
    )

    merged_df = compute_sharpe_ratio(
        merged_df
    )

    merged_df = compute_beta(
        merged_df
    )

    merged_df = compute_alpha(
        merged_df
    )

    merged_df = compute_information_ratio(
        merged_df
    )

    merged_df = add_expense_gap(
        merged_df
    )

    merged_df = compute_net_alpha(
        merged_df
    )

    merged_df = compute_momentum(
        merged_df
    )

    # -----------------------------------------------------
    # Quarterly snapshots
    # -----------------------------------------------------

    quarterly_df = create_quarterly_snapshots(
        merged_df
    )

    # -----------------------------------------------------
    # Create target
    # -----------------------------------------------------

    quarterly_df = create_target_variable(
        quarterly_df
    )

    # -----------------------------------------------------
    # Clean dataset
    # -----------------------------------------------------

    quarterly_df = clean_dataset(
        quarterly_df
    )

    # -----------------------------------------------------
    # Save dataset
    # -----------------------------------------------------

    save_dataset(
        quarterly_df
    )

    # -----------------------------------------------------
    # Preview
    # -----------------------------------------------------

    print("\nHEAD:")
    print(quarterly_df.head())

    print("\nCOLUMNS:")
    print(quarterly_df.columns.tolist())

    print("\nTARGET DISTRIBUTION:")
    print(
        quarterly_df["commission_justified"]
        .value_counts(normalize=True)
    )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()