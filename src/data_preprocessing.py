"""
Preprocessing utilities: loading, scaling, and train/test splitting.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required_cols = {"Class", "Amount", "Time"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    return df


def split_features_target(df: pd.DataFrame):
    X = df.drop(columns=["Class"])
    y = df["Class"]
    return X, y


def scale_features(X_train, X_test, scaler_path: str = "models/scaler.joblib"):
    """Scale 'Amount' and 'Time' (the V1..V28 columns are already PCA-scaled
    in the real dataset, and generated on a similar scale here)."""
    scaler = StandardScaler()
    cols_to_scale = [c for c in ["Time", "Amount"] if c in X_train.columns]

    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

    scaler_dir = os.path.dirname(scaler_path)
    if scaler_dir:
        os.makedirs(scaler_dir, exist_ok=True)
    joblib.dump(scaler, scaler_path)
    return X_train, X_test, scaler


def prepare_train_test(csv_path: str, test_size: float = 0.2, random_state: int = 42,
                        scaler_path: str = "models/scaler.joblib"):
    df = load_data(csv_path)
    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    X_train, X_test, scaler = scale_features(X_train, X_test, scaler_path)
    return X_train, X_test, y_train, y_test, scaler
