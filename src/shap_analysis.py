# shap_analysis.py

"""
CommissionLens
------------------------------------------------------------

STEP 5:
SHAP Explainability Analysis

This script:
1. Loads trained XGBoost model
2. Loads final ML dataset
3. Computes SHAP values
4. Generates feature importance plots
5. Identifies top predictive factors

Author: Prateek Dixit
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import shap
import matplotlib.pyplot as plt


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

    print("Dataset loaded")
    print(df.shape)

    return df


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

    return X, features


# =========================================================
# LOAD MODEL
# =========================================================

def load_model():

    model = joblib.load(MODEL_PATH)

    print("\nModel loaded")

    return model


# =========================================================
# COMPUTE SHAP VALUES
# =========================================================

def compute_shap_values(model, X):

    print("\nComputing SHAP values...")

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X)

    return shap_values


# =========================================================
# SHAP SUMMARY PLOT
# =========================================================

def shap_summary_plot(
    shap_values,
    X
):

    print("\nGenerating SHAP summary plot...")

    plt.figure(figsize=(10, 6))

    shap.summary_plot(
        shap_values,
        X,
        show=False
    )

    save_path = (
        REPORTS_DIR /
        "shap_summary_plot.png"
    )

    plt.savefig(
        save_path,
        bbox_inches="tight",
        dpi=300
    )

    print(f"Saved: {save_path}")

    plt.close()


# =========================================================
# BAR IMPORTANCE PLOT
# =========================================================

def shap_bar_plot(
    shap_values,
    X
):

    print("\nGenerating SHAP bar plot...")

    plt.figure(figsize=(10, 6))

    shap.summary_plot(
        shap_values,
        X,
        plot_type="bar",
        show=False
    )

    save_path = (
        REPORTS_DIR /
        "shap_bar_plot.png"
    )

    plt.savefig(
        save_path,
        bbox_inches="tight",
        dpi=300
    )

    print(f"Saved: {save_path}")

    plt.close()


# =========================================================
# FEATURE IMPORTANCE TABLE
# =========================================================

def generate_feature_importance(
    shap_values,
    X
):

    importance_df = pd.DataFrame({
        "feature": X.columns,
        "mean_abs_shap": np.abs(
            shap_values
        ).mean(axis=0)
    })

    importance_df.sort_values(
        "mean_abs_shap",
        ascending=False,
        inplace=True
    )

    save_path = (
        REPORTS_DIR /
        "feature_importance.csv"
    )

    importance_df.to_csv(
        save_path,
        index=False
    )

    print("\nTop Features:")
    print(importance_df)

    print(f"\nSaved: {save_path}")

    return importance_df


# =========================================================
# MAIN PIPELINE
# =========================================================

def main():

    # -----------------------------------------------------
    # Load dataset
    # -----------------------------------------------------

    df = load_dataset()

    # -----------------------------------------------------
    # Prepare features
    # -----------------------------------------------------

    X, features = prepare_features(df)

    # -----------------------------------------------------
    # Load model
    # -----------------------------------------------------

    model = load_model()

    # -----------------------------------------------------
    # Compute SHAP
    # -----------------------------------------------------

    shap_values = compute_shap_values(
        model,
        X
    )

    # -----------------------------------------------------
    # Summary plot
    # -----------------------------------------------------

    shap_summary_plot(
        shap_values,
        X
    )

    # -----------------------------------------------------
    # Bar plot
    # -----------------------------------------------------

    shap_bar_plot(
        shap_values,
        X
    )

    # -----------------------------------------------------
    # Feature importance
    # -----------------------------------------------------

    generate_feature_importance(
        shap_values,
        X
    )

    print("\nSHAP analysis complete.")


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()