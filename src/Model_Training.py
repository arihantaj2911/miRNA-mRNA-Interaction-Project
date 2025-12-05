"""
model_training.py

Train ML models for miRNA–mRNA interaction prediction using the combined dataset.
"""

import sys
import os

# Add the src directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


from feature_extraction import extract_features_df
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix
)

from feature_extraction import extract_features_df

# Paths
DATA_PATH = os.path.join("data", "combined_dataset.csv")
FIGURES_DIR = "figures"
MODELS_DIR = "models"

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Change if your label column has a different name
LABEL_COLUMN = "interaction"


def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if LABEL_COLUMN not in df.columns:
        raise ValueError(f"Expected label column '{LABEL_COLUMN}' in dataset.")
    return df


def train_and_evaluate():
    # 1. Load raw data
    df = load_dataset(DATA_PATH)
    y = df[LABEL_COLUMN].values

    # 2. Extract features
    X = extract_features_df(df)

    # 3. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Scale features for Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5. Train Logistic Regression
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train_scaled, y_train)

    # 6. Train Random Forest
    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    # 7. Evaluate models
    def evaluate_model(name, model, X_tr, X_te, y_tr, y_te, is_scaled=False):
        if is_scaled:
            X_tr_use, X_te_use = X_tr, X_te
        else:
            X_tr_use, X_te_use = X_tr, X_te

        y_pred = model.predict(X_te_use)
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_te_use)[:, 1]
        else:
            # fallback for models without predict_proba
            y_proba = y_pred

        metrics = {
            "accuracy": accuracy_score(y_te, y_pred),
            "precision": precision_score(y_te, y_pred, zero_division=0),
            "recall": recall_score(y_te, y_pred, zero_division=0),
            "f1_score": f1_score(y_te, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_te, y_proba),
        }

        print(f"\n=== {name} Metrics ===")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

        return metrics, y_pred, y_proba

    log_metrics, log_y_pred, log_y_proba = evaluate_model(
        "Logistic Regression", log_reg,
        X_train_scaled, X_test_scaled,
        y_train, y_test,
        is_scaled=True
    )

    rf_metrics, rf_y_pred, rf_y_proba = evaluate_model(
        "Random Forest", rf,
        X_train, X_test,
        y_train, y_test,
        is_scaled=False
    )

    # 8. Plot ROC curve (for Random Forest as main model)
    fpr, tpr, _ = roc_curve(y_test, rf_y_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f"Random Forest (AUC = {rf_metrics['roc_auc']:.2f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Random Forest")
    plt.legend(loc="lower right")
    roc_path = os.path.join(FIGURES_DIR, "roc_curve_rf.png")
    plt.savefig(roc_path, dpi=300, bbox_inches="tight")
    plt.close()

    # 9. Confusion matrix (Random Forest)
    cm = confusion_matrix(y_test, rf_y_pred)
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix - Random Forest")
    plt.colorbar()
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks([0, 1], ["0", "1"])
    plt.yticks([0, 1], ["0", "1"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    cm_path = os.path.join(FIGURES_DIR, "confusion_matrix_rf.png")
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()

    # 10. Feature importance (Random Forest)
    importances = rf.feature_importances_
    feature_names = list(X.columns)
    sorted_idx = importances.argsort()[::-1]

    plt.figure()
    plt.bar([feature_names[i] for i in sorted_idx],
            [importances[i] for i in sorted_idx])
    plt.xticks(rotation=30)
    plt.ylabel("Importance")
    plt.title("Feature Importance - Random Forest")
    fi_path = os.path.join(FIGURES_DIR, "feature_importance_rf.png")
    plt.savefig(fi_path, dpi=300, bbox_inches="tight")
    plt.close()

    # 11. Save models and scaler
    joblib.dump(log_reg, os.path.join(MODELS_DIR, "logistic_regression.pkl"))
    joblib.dump(rf, os.path.join(MODELS_DIR, "random_forest.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))

    # 12. Save metrics to CSV
    metrics_df = pd.DataFrame(
        [log_metrics, rf_metrics],
        index=["LogisticRegression", "RandomForest"]
    )
    metrics_df.to_csv(os.path.join(MODELS_DIR, "metrics_summary.csv"))

    print("\nTraining complete.")
    print(f"ROC curve saved to: {roc_path}")
    print(f"Confusion matrix saved to: {cm_path}")
    print(f"Feature importance saved to: {fi_path}")
    print("Models and metrics saved in 'models/' directory.")


if __name__ == "__main__":
    train_and_evaluate()
