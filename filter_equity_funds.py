# filter_equity_funds.py

"""
CommissionLens
--------------------------------------------------

STEP 1:
Create clean equity mutual fund universe.

This script:
1. Fetches all mutual fund schemes
2. Filters actively managed equity funds
3. Removes ETFs / Index / Debt / Hybrid
4. Keeps only Growth plans
5. Removes IDCW / Dividend plans
6. Creates normalized fund names
7. Identifies Direct-Regular pairs
8. Saves final clean universe

Author: Prateek Dixit
"""

import requests
import pandas as pd
import re
from pathlib import Path


# =====================================================
# PATHS
# =====================================================

DATA_DIR = Path("data")

RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================
# API URL
# =====================================================

AMFI_URL = "https://api.mfapi.in/mf"


# =====================================================
# FETCH ALL MUTUAL FUND SCHEMES
# =====================================================

def fetch_all_schemes():

    print("Fetching all mutual fund schemes...")

    response = requests.get(AMFI_URL)

    if response.status_code != 200:
        raise Exception(
            f"API request failed: {response.status_code}"
        )

    data = response.json()

    df = pd.DataFrame(data)

    print(f"Total schemes fetched: {len(df)}")

    return df


# =====================================================
# STANDARDIZE COLUMN NAMES
# =====================================================

def standardize_columns(df):

    df.columns = [
        col.lower().strip()
        for col in df.columns
    ]

    print("\nColumns:")
    print(df.columns.tolist())

    return df


# =====================================================
# FILTER EQUITY FUNDS
# =====================================================

def filter_equity_funds(df):

    print("\nFiltering equity mutual funds...")

    scheme_col = "schemename"

    # Lowercase version
    df["scheme_lower"] = (
        df[scheme_col]
        .str.lower()
    )

    # -------------------------------------------------
    # KEEP EQUITY FUNDS
    # -------------------------------------------------

    equity_keywords = [
        "equity",
        "flexi cap",
        "mid cap",
        "small cap",
        "large cap",
        "multi cap",
        "contra",
        "focused",
        "value"
    ]

    equity_mask = df["scheme_lower"].apply(
        lambda x: any(
            keyword in x
            for keyword in equity_keywords
        )
    )

    df = df[equity_mask]

    print(f"After equity filter: {len(df)}")

    # -------------------------------------------------
    # REMOVE ETF / INDEX / DEBT / HYBRID
    # -------------------------------------------------

    remove_keywords = [
        "etf",
        "index",
        "gold",
        "liquid",
        "debt",
        "fof",
        "fund of fund",
        "hybrid",
        "balanced",
        "income",
        "sectoral",
        "thematic"
    ]

    remove_mask = df["scheme_lower"].apply(
        lambda x: any(
            keyword in x
            for keyword in remove_keywords
        )
    )

    df = df[~remove_mask]

    print(f"After ETF/index removal: {len(df)}")

    # -------------------------------------------------
    # KEEP ONLY GROWTH
    # -------------------------------------------------

    growth_mask = (
        df["scheme_lower"]
        .str.contains("growth")
    )

    df = df[growth_mask]

    print(f"After growth filter: {len(df)}")

    # -------------------------------------------------
    # REMOVE IDCW / DIVIDEND
    # -------------------------------------------------

    dividend_keywords = [
        "idcw",
        "dividend",
        "payout",
        "bonus"
    ]

    dividend_mask = df["scheme_lower"].apply(
        lambda x: any(
            keyword in x
            for keyword in dividend_keywords
        )
    )

    df = df[~dividend_mask]

    print(f"After dividend removal: {len(df)}")

    # -------------------------------------------------
    # REMOVE DUPLICATES
    # -------------------------------------------------

    df.drop_duplicates(
        subset=["schemecode"],
        inplace=True
    )

    print(f"Final universe size: {len(df)}")

    return df


# =====================================================
# NORMALIZE FUND NAMES
# =====================================================

def normalize_fund_names(df):

    print("\nNormalizing fund names...")

    def clean_name(name):

        name = name.lower()

        patterns = [
            r"direct",
            r"regular",
            r"growth",
            r"plan",
            r"option",
            r"idcw",
            r"dividend",
            r"-",
            r"\(",
            r"\)",
        ]

        for pattern in patterns:
            name = re.sub(
                pattern,
                "",
                name
            )

        # Remove extra spaces
        name = re.sub(
            r"\s+",
            " ",
            name
        ).strip()

        return name

    df["base_fund_name"] = (
        df["schemename"]
        .apply(clean_name)
    )

    return df


# =====================================================
# IDENTIFY DIRECT / REGULAR PAIRS
# =====================================================

def identify_pairs(df):

    print("\nIdentifying direct-regular pairs...")

    grouped = (
        df.groupby("base_fund_name")
        .size()
        .reset_index(name="count")
    )

    paired_names = grouped[
        grouped["count"] >= 2
    ]["base_fund_name"]

    df = df[
        df["base_fund_name"]
        .isin(paired_names)
    ]

    print(f"Paired fund universe: {len(df)}")

    return df


# =====================================================
# SAVE DATASET
# =====================================================

def save_dataset(df):

    raw_path = (
        RAW_DIR /
        "all_mutual_funds.csv"
    )

    processed_path = (
        PROCESSED_DIR /
        "equity_fund_universe.csv"
    )

    df.to_csv(
        raw_path,
        index=False
    )

    df.to_csv(
        processed_path,
        index=False
    )

    print("\nDatasets saved:")
    print(raw_path)
    print(processed_path)


# =====================================================
# MAIN PIPELINE
# =====================================================

def main():

    # Fetch all schemes
    df = fetch_all_schemes()

    # Standardize columns
    df = standardize_columns(df)

    # Filter equity funds
    df = filter_equity_funds(df)

    # Normalize names
    df = normalize_fund_names(df)

    # Identify pairs
    df = identify_pairs(df)

    # Save dataset
    save_dataset(df)

    # -------------------------------------------------
    # PREVIEW
    # -------------------------------------------------

    print("\nFINAL DATASET PREVIEW:")
    print(df.head())

    print("\nFINAL SHAPE:")
    print(df.shape)

    print("\nUNIQUE FUND HOUSES:")

    print(
        df["schemename"]
        .str.split()
        .str[0]
        .nunique()
    )


# =====================================================
# ENTRY POINT
# =====================================================

if __name__ == "__main__":
    main()