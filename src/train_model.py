"""
Trains fraud detection models on a highly imbalanced dataset.

Handles class imbalance via SMOTE oversampling on the training set only
(never on test data, to avoid leakage/inflated metrics).

Usage:
    python src/train_model.py --data data/creditcard.csv --model random_forest
"""

import argparse
import os
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from data_preprocessing import prepare_train_test
from evaluate import evaluate_model


MODEL_REGISTRY = {
    "logistic_regression": lambda: LogisticRegression(max_iter=1000, class_weight="balanced"),
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=200, max_depth=12, random_state=42, n_jobs=-1, class_weight="balanced"
    ),
}

if XGBOOST_AVAILABLE:
    MODEL_REGISTRY["xgboost"] = lambda: XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )


def train(data_path: str, model_name: str, use_smote: bool, model_out: str):
    model_dir = os.path.dirname(model_out) or "."
    os.makedirs(model_dir, exist_ok=True)
    scaler_out = os.path.join(model_dir, "scaler.joblib")

    print("⏳ Loading and preprocessing data...")
    X_train, X_test, y_train, y_test, _ = prepare_train_test(data_path, scaler_path=scaler_out)

    if use_smote:
        try:
            from imblearn.over_sampling import SMOTE
        except ImportError:
            raise ImportError(
                "imbalanced-learn is not installed. Run `pip install imbalanced-learn`, "
                "or pass --no_smote to train without oversampling."
            )
        print("⚖️  Applying SMOTE to balance the training set...")
        print(f"   Before SMOTE: {np.bincount(y_train)}")
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        print(f"   After SMOTE:  {np.bincount(y_train)}")

    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Options: {list(MODEL_REGISTRY.keys())}")

    print(f"🚀 Training {model_name}...")
    model = MODEL_REGISTRY[model_name]()
    model.fit(X_train, y_train)

    print("📊 Evaluating on held-out test set...")
    evaluate_model(model, X_test, y_test)

    joblib.dump(model, model_out)
    print(f"💾 Model saved to {model_out}")


def main():
    parser = argparse.ArgumentParser(description="Train a credit card fraud detection model.")
    parser.add_argument("--data", type=str, default="data/creditcard.csv")
    parser.add_argument("--model", type=str, default="random_forest",
                        choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--no_smote", action="store_true", help="Disable SMOTE oversampling")
    parser.add_argument("--model_out", type=str, default="models/fraud_model.joblib")
    args = parser.parse_args()

    train(args.data, args.model, use_smote=not args.no_smote, model_out=args.model_out)


if __name__ == "__main__":
    main()
