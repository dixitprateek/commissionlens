# fetch_mutual_fund_data.py

"""
CommissionLens
-------------------------------------------------------

STEP 2:
Fetch historical NAV data for all filtered
equity mutual funds.

This script:
1. Loads clean equity fund universe
2. Fetches NAV history for each scheme
3. Cleans and standardizes NAV data
4. Saves individual fund CSVs
5. Builds master NAV dataset

Author: Prateek Dixit
"""

import requests
import pandas as pd
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed


# =====================================================
# PATHS
# =====================================================

DATA_DIR = Path("data")

RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

NAV_DIR = RAW_DIR / "nav_data"

NAV_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================
# FILES
# =====================================================

UNIVERSE_PATH = (
    PROCESSED_DIR /
    "equity_fund_universe.csv"
)

MASTER_NAV_PATH = (
    PROCESSED_DIR /
    "master_nav_dataset.csv"
)


# =====================================================
# API URL
# =====================================================

BASE_URL = "https://api.mfapi.in/mf"


# =====================================================
# LOAD FUND UNIVERSE
# =====================================================

def load_fund_universe():

    df = pd.read_csv(
        UNIVERSE_PATH
    )

    print("Fund universe loaded")
    print(df.shape)

    return df


# =====================================================
# FETCH SINGLE FUND NAV
# =====================================================

def fetch_single_fund_nav(row):

    scheme_code = row["schemecode"]
    scheme_name = row["schemename"]

    url = f"{BASE_URL}/{scheme_code}"

    try:

        response = requests.get(
            url,
            timeout=20
        )

        if response.status_code != 200:
            print(f"Failed: {scheme_code}")
            return None

        data = response.json()

        if "data" not in data:
            return None

        nav_df = pd.DataFrame(
            data["data"]
        )

        if nav_df.empty:
            return None

        # --------------------------------------------
        # Standardize columns
        # --------------------------------------------

        nav_df.rename(
            columns={
                "date": "date",
                "nav": "nav"
            },
            inplace=True
        )

        # --------------------------------------------
        # Convert types
        # --------------------------------------------

        nav_df["date"] = pd.to_datetime(
            nav_df["date"],
            format="%d-%m-%Y",
            errors="coerce"
        )

        nav_df["nav"] = pd.to_numeric(
            nav_df["nav"],
            errors="coerce"
        )

        # --------------------------------------------
        # Add metadata
        # --------------------------------------------

        nav_df["schemecode"] = scheme_code
        nav_df["schemename"] = scheme_name

        # --------------------------------------------
        # Drop invalid rows
        # --------------------------------------------

        nav_df.dropna(
            subset=["date", "nav"],
            inplace=True
        )

        # --------------------------------------------
        # Sort
        # --------------------------------------------

        nav_df.sort_values(
            "date",
            inplace=True
        )

        nav_df.reset_index(
            drop=True,
            inplace=True
        )

        # --------------------------------------------
        # Save individual fund CSV
        # --------------------------------------------

        safe_name = (
            str(scheme_name)
            .replace("/", "_")
            .replace("\\", "_")
            .replace(":", "_")
            .replace("*", "_")
            .replace("?", "_")
            .replace('"', "_")
            .replace("<", "_")
            .replace(">", "_")
            .replace("|", "_")
        )

        save_path = (
            NAV_DIR /
            f"{scheme_code}_{safe_name}.csv"
        )

        nav_df.to_csv(
            save_path,
            index=False
        )

        print(
            f"Fetched: {scheme_name}"
        )

        return nav_df

    except Exception as e:

        print(
            f"Error fetching {scheme_code}"
        )

        print(e)

        return None


# =====================================================
# MULTI-THREAD FETCHING
# =====================================================

def fetch_all_nav_data(df):

    print("\nFetching NAV data...")

    all_nav_dfs = []

    rows = [
        row
        for _, row in df.iterrows()
    ]

    with ThreadPoolExecutor(
        max_workers=10
    ) as executor:

        futures = [
            executor.submit(
                fetch_single_fund_nav,
                row
            )
            for row in rows
        ]

        for future in as_completed(futures):

            result = future.result()

            if result is not None:
                all_nav_dfs.append(result)

    return all_nav_dfs


# =====================================================
# COMBINE DATASETS
# =====================================================

def combine_nav_data(all_nav_dfs):

    print("\nCombining NAV datasets...")

    master_df = pd.concat(
        all_nav_dfs,
        axis=0,
        ignore_index=True
    )

    master_df.sort_values(
        ["schemecode", "date"],
        inplace=True
    )

    master_df.reset_index(
        drop=True,
        inplace=True
    )

    print(
        f"Master dataset shape: {master_df.shape}"
    )

    return master_df


# =====================================================
# COMPUTE RETURNS
# =====================================================

def compute_returns(df):

    print("\nComputing daily returns...")

    df["daily_return"] = (
        df.groupby("schemecode")["nav"]
        .pct_change()
    )

    return df


# =====================================================
# SAVE MASTER DATASET
# =====================================================

def save_master_dataset(df):

    df.to_csv(
        MASTER_NAV_PATH,
        index=False
    )

    print("\nMaster dataset saved:")
    print(MASTER_NAV_PATH)


# =====================================================
# MAIN PIPELINE
# =====================================================

def main():

    # --------------------------------------------
    # Load fund universe
    # --------------------------------------------

    universe_df = load_fund_universe()

    # --------------------------------------------
    # OPTIONAL:
    # Limit during testing
    # --------------------------------------------

    # universe_df = universe_df.head(20)

    # --------------------------------------------
    # Fetch NAV data
    # --------------------------------------------

    all_nav_dfs = fetch_all_nav_data(
        universe_df
    )

    # --------------------------------------------
    # Combine
    # --------------------------------------------

    master_df = combine_nav_data(
        all_nav_dfs
    )

    # --------------------------------------------
    # Compute returns
    # --------------------------------------------

    master_df = compute_returns(
        master_df
    )

    # --------------------------------------------
    # Save
    # --------------------------------------------

    save_master_dataset(
        master_df
    )

    # --------------------------------------------
    # Preview
    # --------------------------------------------

    print("\nHEAD:")
    print(master_df.head())

    print("\nCOLUMNS:")
    print(master_df.columns.tolist())

    print("\nUNIQUE FUNDS:")
    print(
        master_df["schemecode"]
        .nunique()
    )


# =====================================================
# ENTRY POINT
# =====================================================

if __name__ == "__main__":
    main()