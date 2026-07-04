# 💳 Credit Card Fraud Detection

A machine learning pipeline that predicts whether a credit card transaction is
fraudulent 🕵️, built to handle the extreme class imbalance typical of real-world
fraud data (fraud is usually <1% of transactions).

## ✨ Features

- 📥 **Data pipeline**: loading, scaling (`Time`/`Amount`), stratified train/test split
- ⚖️ **Imbalance handling**: SMOTE oversampling on the training set
- 🤖 **Multiple models**: Logistic Regression, Random Forest, XGBoost
- 📊 **Imbalance-aware evaluation**: precision, recall, F1, ROC-AUC, average
  precision, and confusion matrix (accuracy alone is misleading here!)
- 🔮 **Prediction script** for scoring new/unseen transactions
- 🧪 **Synthetic data generator** so the whole thing runs with zero setup

## 📂 Project Structure

```
credit-card-fraud-detection/
├── data/
│   └── generate_data.py      # creates a synthetic dataset (or drop in the real Kaggle CSV)
├── src/
│   ├── data_preprocessing.py # loading, scaling, splitting
│   ├── train_model.py        # SMOTE + model training
│   ├── evaluate.py           # imbalance-aware metrics
│   └── predict.py            # score new transactions with a saved model
├── notebooks/
│   └── EDA.ipynb             # exploratory data analysis
├── models/                   # trained model + scaler get saved here
├── requirements.txt
└── README.md
```

## 🛠️ Setup

```bash
git clone <your-repo-url>
cd credit-card-fraud-detection
python -m venv venv && source venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

## 🚀 Quickstart

**1️⃣ Get data.** Generate a synthetic dataset:

```bash
python data/generate_data.py --n_samples 50000 --fraud_ratio 0.0017
```

Or use the real dataset 📦: download `creditcard.csv` from
[Kaggle: Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
and place it in `data/`. The real dataset has the same schema (`Time`, `V1`-`V28`,
`Amount`, `Class`), so no code changes are needed.

**2️⃣ Train a model:** 🏋️

```bash
cd src
python train_model.py --data ../data/creditcard.csv --model random_forest
```

Available models: `logistic_regression`, `random_forest`, `xgboost` 🧠.
Add `--no_smote` to train without oversampling.

**3️⃣ Score new transactions:** 🔎

```bash
python predict.py --input ../data/new_transactions.csv --output ../predictions.csv
```

## ⚠️ Why not just use accuracy?

In this dataset roughly 1 in 600 transactions is fraudulent. A model that
predicts "legitimate" for every single transaction scores ~99.8% accuracy
while catching zero fraud — accuracy is meaningless here. This project
reports precision, recall, F1, ROC-AUC, and average precision instead, and
uses SMOTE to prevent the model from just learning to predict the majority
class.

## 💡 Possible extensions

- 🎯 Tune the classification threshold to hit a specific precision/recall target
- 💰 Add a cost-sensitive evaluation (false negatives are usually far more
  expensive than false positives in real fraud systems)
- 🕳️ Try an autoencoder / isolation forest for unsupervised anomaly detection
- 🌐 Wrap `predict.py` in a small Flask/FastAPI service for real-time scoring
- 🔬 Add SHAP explainability to see which features drive each fraud flag

## 📄 License

MIT — feel free to use this as a starting point for your own projects. ⭐
