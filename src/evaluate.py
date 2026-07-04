"""
Evaluation utilities for fraud detection models.

Accuracy is a misleading metric here because ~99.8% of transactions are
legitimate — a model that predicts "not fraud" every time would score
>99% accuracy while catching zero fraud. We report precision, recall,
F1, ROC-AUC, and the confusion matrix instead.
"""

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)


def evaluate_model(model, X_test, y_test, threshold: float = 0.5):
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    print("\n" + "=" * 50)
    print("📊 CLASSIFICATION REPORT")
    print("=" * 50)
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"], digits=4))

    print("🧮 CONFUSION MATRIX")
    cm = confusion_matrix(y_test, y_pred)
    print(f"                Predicted Legit   Predicted Fraud")
    print(f"Actual Legit    {cm[0][0]:<17}{cm[0][1]}")
    print(f"Actual Fraud    {cm[1][0]:<17}{cm[1][1]}")

    roc_auc = roc_auc_score(y_test, y_proba)
    avg_precision = average_precision_score(y_test, y_proba)
    print(f"\n✅ ROC-AUC:              {roc_auc:.4f}")
    print(f"✅ Average Precision:    {avg_precision:.4f}  (better metric for imbalanced data)")
    print("=" * 50)

    return {
        "roc_auc": roc_auc,
        "average_precision": avg_precision,
        "confusion_matrix": cm,
    }
