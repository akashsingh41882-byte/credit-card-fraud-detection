"""
Loads a trained model + scaler and scores new transactions from a CSV file.

Usage:
    python src/predict.py --input data/new_transactions.csv --output predictions.csv
"""

import argparse
import joblib
import pandas as pd


def predict(input_csv: str, model_path: str, scaler_path: str, output_csv: str, threshold: float):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    df = pd.read_csv(input_csv)
    df_features = df.drop(columns=["Class"]) if "Class" in df.columns else df.copy()

    cols_to_scale = [c for c in ["Time", "Amount"] if c in df_features.columns]
    df_features[cols_to_scale] = scaler.transform(df_features[cols_to_scale])

    proba = model.predict_proba(df_features)[:, 1]
    pred = (proba >= threshold).astype(int)

    result = df.copy()
    result["fraud_probability"] = proba
    result["predicted_fraud"] = pred
    result.to_csv(output_csv, index=False)

    print(f"🔎 Scored {len(result):,} transactions.")
    print(f"🚨 Flagged as fraud: {pred.sum():,} ({pred.mean() * 100:.3f}%)")
    print(f"💾 Results saved to {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="Score transactions for fraud probability.")
    parser.add_argument("--input", type=str, required=True, help="CSV of transactions to score")
    parser.add_argument("--model", type=str, default="models/fraud_model.joblib")
    parser.add_argument("--scaler", type=str, default="models/scaler.joblib")
    parser.add_argument("--output", type=str, default="predictions.csv")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Probability threshold above which a transaction is flagged")
    args = parser.parse_args()

    predict(args.input, args.model, args.scaler, args.output, args.threshold)


if __name__ == "__main__":
    main()
