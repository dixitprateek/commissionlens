# sip_backtest.py

"""
CommissionLens
------------------------------------------------------------

STEP 6:
SIP Backtesting

This script:
1. Loads trained XGBoost model
2. Predicts commission justification
3. Selects top predicted funds
4. Simulates monthly SIP investing
5. Computes XIRR
6. Compares:
    - Baseline investing
    - ML-driven investing

Author: Prateek Dixit
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib


# =========================================================
# PATHS
# =========================================================

DATA_DIR = Path("data")

PROCESSED_DIR = DATA_DIR / "processed"

MODELS_DIR = Path("models")

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

DATASET_PATH = (
    PROCESSED_DIR /
    "final_model_dataset.csv"
)

MODEL_PATH = (
    MODELS_DIR /
    "xgboost.pkl"
)


# =========================================================
# LOAD DATASET
# =========================================================

def load_dataset():

    df = pd.read_csv(DATASET_PATH)

    df["date"] = pd.to_datetime(
        df["date"]
    )

    print("Dataset loaded")
    print(df.shape)

    return df


# =========================================================
# LOAD MODEL
# =========================================================

def load_model():

    model = joblib.load(MODEL_PATH)

    print("\nModel loaded")

    return model


# =========================================================
# PREPARE FEATURES
# =========================================================

def prepare_features(df):

    features = [
        "rolling_252d_return",
        "rolling_volatility",
        "rolling_sharpe",
        "beta",
        "alpha",
        "information_ratio",
        "expense_gap",
        "momentum_90d"
    ]

    X = df[features]

    return X


# =========================================================
# PREDICT PROBABILITIES
# =========================================================

def predict_probabilities(
    model,
    X,
    df
):

    probs = model.predict_proba(X)[:, 1]

    df["prediction_prob"] = probs

    return df


# =========================================================
# SELECT TOP FUNDS
# =========================================================

def select_top_funds(df):

    print("\nSelecting top predicted funds...")

    selected_df = (
        df.groupby("quarter")
        .apply(
            lambda x:
            x.nlargest(
                max(1, int(len(x) * 0.10)),
                "prediction_prob"
            )
        )
        .reset_index(drop=True)
    )

    print(
        f"Selected observations: "
        f"{selected_df.shape}"
    )

    return selected_df


# =========================================================
# STABLE XIRR
# =========================================================

def xirr(cashflows):

    dates = [cf[0] for cf in cashflows]
    amounts = [cf[1] for cf in cashflows]

    years = np.array([
        (d - dates[0]).days / 365
        for d in dates
    ])

    amounts = np.array(amounts)

    # -----------------------------------------------------
    # Stable grid-search approach
    # -----------------------------------------------------

    rates = np.linspace(
        -0.99,
        5,
        10000
    )

    npvs = []

    for r in rates:

        try:

            npv = np.sum(
                amounts /
                ((1 + r) ** years)
            )

            npvs.append(npv)

        except:
            npvs.append(np.nan)

    npvs = np.array(npvs)

    # -----------------------------------------------------
    # Safety check
    # -----------------------------------------------------

    if np.all(np.isnan(npvs)):

        return np.nan

    idx = np.nanargmin(
        np.abs(npvs)
    )

    return rates[idx]


# =========================================================
# SIP SIMULATION
# =========================================================

def simulate_sip(df):

    print("\nRunning SIP simulation...")

    monthly_investment = 5000

    # -----------------------------------------------------
    # Use rolling annual return
    # instead of future_net_alpha
    # -----------------------------------------------------

    quarterly_returns = (
        df.groupby("quarter")[
            "rolling_252d_return"
        ]
        .mean()
    )

    # Annualized -> quarterly approximation
    quarterly_returns = (
        quarterly_returns / 4
    )

    # -----------------------------------------------------
    # Clean invalid values
    # -----------------------------------------------------

    quarterly_returns = quarterly_returns.replace(
        [np.inf, -np.inf],
        np.nan
    )

    quarterly_returns = (
        quarterly_returns.dropna()
    )

    # -----------------------------------------------------
    # SIP simulation
    # -----------------------------------------------------

    cashflows = []

    portfolio_value = 0

    quarters = (
        quarterly_returns.index.tolist()
    )

    dates = []

    for i, quarter in enumerate(quarters):

        # Convert string quarter
        # to timestamp
        date = pd.Period(
            quarter,
            freq="Q"
        ).start_time

        dates.append(date)

        # Quarterly SIP
        sip_amount = (
            monthly_investment * 3
        )

        cashflows.append(
            (date, -sip_amount)
        )

        # Quarterly growth
        quarterly_return = (
            quarterly_returns.iloc[i]
        )

        portfolio_value = (
            portfolio_value
            *
            (1 + quarterly_return)
        )

        portfolio_value += sip_amount

    # -----------------------------------------------------
    # Final redemption
    # -----------------------------------------------------

    cashflows.append(
        (
            dates[-1],
            portfolio_value
        )
    )

    final_xirr = xirr(cashflows)

    return portfolio_value, final_xirr


# =========================================================
# BASELINE STRATEGY
# =========================================================

def run_baseline_strategy(df):

    print("\nBaseline strategy...")

    portfolio_value, strategy_xirr = (
        simulate_sip(df)
    )

    return portfolio_value, strategy_xirr


# =========================================================
# ML STRATEGY
# =========================================================

def run_ml_strategy(df):

    print("\nML strategy...")

    selected_df = select_top_funds(df)

    portfolio_value, strategy_xirr = (
        simulate_sip(selected_df)
    )

    return portfolio_value, strategy_xirr


# =========================================================
# SAVE RESULTS
# =========================================================

def save_results(results_df):

    save_path = (
        REPORTS_DIR /
        "sip_backtest_results.csv"
    )

    results_df.to_csv(
        save_path,
        index=False
    )

    print(f"\nSaved: {save_path}")


# =========================================================
# MAIN PIPELINE
# =========================================================

def main():

    # -----------------------------------------------------
    # Load data
    # -----------------------------------------------------

    df = load_dataset()

    # -----------------------------------------------------
    # Load model
    # -----------------------------------------------------

    model = load_model()

    # -----------------------------------------------------
    # Prepare features
    # -----------------------------------------------------

    X = prepare_features(df)

    # -----------------------------------------------------
    # Predictions
    # -----------------------------------------------------

    df = predict_probabilities(
        model,
        X,
        df
    )

    # -----------------------------------------------------
    # Baseline strategy
    # -----------------------------------------------------

    baseline_value, baseline_xirr = (
        run_baseline_strategy(df)
    )

    # -----------------------------------------------------
    # ML strategy
    # -----------------------------------------------------

    ml_value, ml_xirr = (
        run_ml_strategy(df)
    )

    # -----------------------------------------------------
    # Results dataframe
    # -----------------------------------------------------

    results_df = pd.DataFrame({

        "strategy": [
            "Baseline",
            "ML Strategy"
        ],

        "final_portfolio_value": [
            round(baseline_value, 2),
            round(ml_value, 2)
        ],

        "xirr": [
            round(baseline_xirr, 4),
            round(ml_xirr, 4)
        ]
    })

    # -----------------------------------------------------
    # Print results
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("SIP BACKTEST RESULTS")
    print("=" * 60)

    print(results_df)

    # -----------------------------------------------------
    # Save results
    # -----------------------------------------------------

    save_results(results_df)

    print("\nSIP backtest complete.")


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()