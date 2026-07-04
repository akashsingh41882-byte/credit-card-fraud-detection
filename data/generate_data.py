"""
Generates a synthetic credit card transaction dataset that mimics the
structure and class imbalance of the well-known Kaggle "Credit Card Fraud
Detection" dataset (28 anonymized PCA features V1-V28, Time, Amount, Class).

Why synthetic data?
--------------------
The real Kaggle dataset (~150MB, 284,807 transactions) can't be bundled in
this repo. This script generates a statistically similar dataset locally so
the whole pipeline runs out of the box. If you want to use the REAL dataset
instead, download it from:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
and drop `creditcard.csv` into this `data/` folder — the rest of the
pipeline works unchanged.

Usage:
    python data/generate_data.py --n_samples 50000 --fraud_ratio 0.0017
"""

import argparse
import numpy as np
import pandas as pd


def generate_synthetic_transactions(n_samples: int, fraud_ratio: float, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    n_fraud = max(1, int(n_samples * fraud_ratio))
    n_legit = n_samples - n_fraud

    n_features = 28  # mimics V1..V28 from the real dataset

    # Legitimate transactions: features drawn from a standard normal distribution
    legit_features = rng.normal(loc=0.0, scale=1.0, size=(n_legit, n_features))
    legit_amount = np.round(rng.gamma(shape=2.0, scale=40.0, size=n_legit), 2)
    legit_time = rng.uniform(0, 172800, size=n_legit)  # 2 days, in seconds
    legit_labels = np.zeros(n_legit, dtype=int)

    # Fraudulent transactions: shifted mean + higher variance on a subset of
    # features, and a different amount distribution -> makes them separable
    # but not trivially so (like real fraud patterns).
    fraud_features = rng.normal(loc=0.0, scale=1.0, size=(n_fraud, n_features))
    shifted_cols = rng.choice(n_features, size=8, replace=False)
    fraud_features[:, shifted_cols] += rng.normal(loc=3.5, scale=1.5, size=(n_fraud, len(shifted_cols)))
    fraud_amount = np.round(rng.gamma(shape=1.2, scale=150.0, size=n_fraud), 2)
    fraud_time = rng.uniform(0, 172800, size=n_fraud)
    fraud_labels = np.ones(n_fraud, dtype=int)

    features = np.vstack([legit_features, fraud_features])
    amount = np.concatenate([legit_amount, fraud_amount])
    time = np.concatenate([legit_time, fraud_time])
    labels = np.concatenate([legit_labels, fraud_labels])

    columns = [f"V{i+1}" for i in range(n_features)]
    df = pd.DataFrame(features, columns=columns)
    df.insert(0, "Time", time)
    df["Amount"] = amount
    df["Class"] = labels

    # Shuffle rows so fraud isn't clustered at the end
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic credit card transaction data.")
    parser.add_argument("--n_samples", type=int, default=50000, help="Total number of transactions")
    parser.add_argument("--fraud_ratio", type=float, default=0.0017, help="Fraction of fraudulent transactions")
    parser.add_argument("--output", type=str, default="data/creditcard.csv", help="Output CSV path")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = generate_synthetic_transactions(args.n_samples, args.fraud_ratio, args.seed)
    df.to_csv(args.output, index=False)

    print(f"🎉 Generated {len(df):,} transactions -> {args.output}")
    print(f"🚨 Fraud cases: {df['Class'].sum():,} ({df['Class'].mean() * 100:.3f}%)")


if __name__ == "__main__":
    main()
