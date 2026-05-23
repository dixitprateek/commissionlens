# train_baseline.py

"""
CommissionLens
------------------------------------------------------------

STEP 4:
Train baseline ML models.

This script:
1. Loads final ML dataset
2. Performs temporal train-test split
3. Trains:
    - Logistic Regression
    - Random Forest
    - XGBoost
4. Evaluates models
5. Computes Precision@TopDecile
6. Saves trained models

Author: Prateek Dixit
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib

from sklearn.model_selection import TimeSeriesSplit

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)


# =========================================================
# PATHS
# =========================================================

DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

DATASET_PATH = (
    PROCESSED_DIR /
    "final_model_dataset.csv"
)


# =========================================================
# LOAD DATASET
# =========================================================

def load_dataset():

    df = pd.read_csv(DATASET_PATH)

    df["date"] = pd.to_datetime(df["date"])

    print("Dataset loaded")
    print(df.shape)

    return df


# =========================================================
# FEATURE SELECTION
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

    target = "commission_justified"

    X = df[features]

    y = df[target]

    return X, y, features


# =========================================================
# TEMPORAL SPLIT
# =========================================================

def temporal_train_test_split(df, X, y):

    df = df.sort_values("date")

    split_index = int(len(df) * 0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    print("\nTrain size:", X_train.shape)
    print("Test size:", X_test.shape)

    return X_train, X_test, y_train, y_test


# =========================================================
# PRECISION @ TOP DECILE
# =========================================================

def precision_at_top_decile(y_true, probs):

    results_df = pd.DataFrame({
        "y_true": y_true,
        "prob": probs
    })

    results_df = results_df.sort_values(
        "prob",
        ascending=False
    )

    top_n = int(len(results_df) * 0.10)

    top_df = results_df.head(top_n)

    precision = top_df["y_true"].mean()

    return precision


# =========================================================
# EVALUATION FUNCTION
# =========================================================

def evaluate_model(
    model,
    X_test,
    y_test,
    model_name
):

    preds = model.predict(X_test)

    probs = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(
        y_test,
        preds
    )

    precision = precision_score(
        y_test,
        preds
    )

    recall = recall_score(
        y_test,
        preds
    )

    f1 = f1_score(
        y_test,
        preds
    )

    roc_auc = roc_auc_score(
        y_test,
        probs
    )

    precision_top_decile = (
        precision_at_top_decile(
            y_test,
            probs
        )
    )

    print("\n" + "=" * 60)
    print(model_name)
    print("=" * 60)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")

    print(
        f"Precision@TopDecile: "
        f"{precision_top_decile:.4f}"
    )

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            preds
        )
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "precision_top_decile":
            precision_top_decile
    }


# =========================================================
# LOGISTIC REGRESSION
# =========================================================

def train_logistic_regression(
    X_train,
    y_train
):

    model = LogisticRegression(
        max_iter=1000
    )

    model.fit(
        X_train,
        y_train
    )

    return model


# =========================================================
# RANDOM FOREST
# =========================================================

def train_random_forest(
    X_train,
    y_train
):

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    return model


# =========================================================
# XGBOOST
# =========================================================

def train_xgboost(
    X_train,
    y_train
):

    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss"
    )

    model.fit(
        X_train,
        y_train
    )

    return model


# =========================================================
# SAVE MODEL
# =========================================================

def save_model(model, file_name):

    save_path = (
        MODELS_DIR /
        file_name
    )

    joblib.dump(
        model,
        save_path
    )

    print(f"\nModel saved: {save_path}")


# =========================================================
# MAIN PIPELINE
# =========================================================

def main():

    # -----------------------------------------------------
    # Load dataset
    # -----------------------------------------------------

    df = load_dataset()

    # -----------------------------------------------------
    # Features
    # -----------------------------------------------------

    X, y, features = prepare_features(df)

    print("\nFeatures:")
    print(features)

    # -----------------------------------------------------
    # Temporal split
    # -----------------------------------------------------

    X_train, X_test, y_train, y_test = (
        temporal_train_test_split(
            df,
            X,
            y
        )
    )

    # -----------------------------------------------------
    # Logistic Regression
    # -----------------------------------------------------

    log_model = train_logistic_regression(
        X_train,
        y_train
    )

    evaluate_model(
        log_model,
        X_test,
        y_test,
        "Logistic Regression"
    )

    save_model(
        log_model,
        "logistic_regression.pkl"
    )

    # -----------------------------------------------------
    # Random Forest
    # -----------------------------------------------------

    rf_model = train_random_forest(
        X_train,
        y_train
    )

    evaluate_model(
        rf_model,
        X_test,
        y_test,
        "Random Forest"
    )

    save_model(
        rf_model,
        "random_forest.pkl"
    )

    # -----------------------------------------------------
    # XGBoost
    # -----------------------------------------------------

    xgb_model = train_xgboost(
        X_train,
        y_train
    )

    evaluate_model(
        xgb_model,
        X_test,
        y_test,
        "XGBoost"
    )

    save_model(
        xgb_model,
        "xgboost.pkl"
    )

    print("\nTraining pipeline complete.")


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()