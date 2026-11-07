"""Train five classifiers and create a held-out CSV for the Streamlit demo.

Run from this folder with: python train_models.py
The split is stratified and fixed so the saved metrics and test_data.csv agree.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "dry_bean.csv"
MODEL_DIR = ROOT / "model"
RANDOM_STATE = 42
TARGET = "Class"


def build_models(feature_names: list[str]) -> dict[str, Pipeline]:
    """Build preprocessing + classifier pipelines; scaling is fit on train only."""
    numeric_preprocessor = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    def pipeline(estimator):
        numeric_columns = ColumnTransformer(
            [("numeric", numeric_preprocessor, feature_names)],
            remainder="drop",
            verbose_feature_names_out=False,
        )
        return Pipeline([("preprocess", numeric_columns), ("classifier", estimator)])

    return {
        "Logistic Regression": pipeline(
            LogisticRegression(max_iter=2000, class_weight=None, random_state=RANDOM_STATE)
        ),
        "Decision Tree": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("classifier", DecisionTreeClassifier(max_depth=16, min_samples_leaf=3, random_state=RANDOM_STATE)),
            ]
        ),
        "K-Nearest Neighbors": pipeline(KNeighborsClassifier(n_neighbors=7, weights="distance")),
        "Gaussian Naive Bayes": pipeline(GaussianNB()),
        "Support Vector Machine": pipeline(SVC(C=2.0, kernel="rbf", probability=True, random_state=RANDOM_STATE)),
    }


def calculate_metrics(model, X_test: pd.DataFrame, y_test, labels: list[str]) -> dict:
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    try:
        auc = roc_auc_score(y_test, probabilities, labels=list(range(len(labels))), multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")
    return {
        "Accuracy": float(accuracy_score(y_test, predictions)),
        "AUC (macro OVR)": float(auc),
        "Precision (macro)": float(precision_score(y_test, predictions, average="macro", zero_division=0)),
        "Recall (macro)": float(recall_score(y_test, predictions, average="macro", zero_division=0)),
        "F1 (macro)": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        "MCC": float(matthews_corrcoef(y_test, predictions)),
        "Confusion matrix": confusion_matrix(y_test, predictions, labels=list(range(len(labels)))).tolist(),
        "Classification report": classification_report(
            y_test, predictions, labels=list(range(len(labels))), target_names=labels, output_dict=True, zero_division=0
        ),
    }


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset missing: {DATA_PATH}. See README for the UCI download steps.")
    data = pd.read_csv(DATA_PATH)
    if TARGET not in data:
        raise ValueError(f"Expected target column {TARGET!r}; found {data.columns.tolist()}")
    data = data.dropna(subset=[TARGET]).copy()
    X = data.drop(columns=[TARGET])
    X = X.select_dtypes(include="number")
    if X.shape[1] < 12:
        raise ValueError(f"Assignment requires at least 12 input features; found {X.shape[1]}.")

    encoder = LabelEncoder()
    y = encoder.fit_transform(data[TARGET].astype(str))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )
    # Store labels with the test rows so the app can measure performance.
    test_data = X_test.copy()
    test_data[TARGET] = encoder.inverse_transform(y_test)

    MODEL_DIR.mkdir(exist_ok=True)
    test_data.to_csv(ROOT / "test_data.csv", index=False)
    models = build_models(X.columns.tolist())
    results = {}
    file_names = {
        "Logistic Regression": "logistic_regression.joblib",
        "Decision Tree": "decision_tree.joblib",
        "K-Nearest Neighbors": "k_nearest_neighbors.joblib",
        "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
        "Support Vector Machine": "support_vector_machine.joblib",
    }
    for display_name, model in models.items():
        model.fit(X_train, y_train)
        filename = file_names[display_name]
        joblib.dump(model, MODEL_DIR / filename)
        results[display_name] = calculate_metrics(model, X_test, y_test, encoder.classes_.tolist())
        print(f"{display_name}: accuracy={results[display_name]['Accuracy']:.4f}, AUC={results[display_name]['AUC (macro OVR)']:.4f}")

    metadata = {
        "dataset": "UCI Dry Bean",
        "target": TARGET,
        "features": X.columns.tolist(),
        "classes": encoder.classes_.tolist(),
        "n_rows": int(len(data)),
        "n_features": int(X.shape[1]),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "test_split": 0.25,
        "random_state": RANDOM_STATE,
        "metrics": results,
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"\nSaved five models in {MODEL_DIR}")
    print(f"Saved labeled holdout data in {ROOT / 'test_data.csv'}")


if __name__ == "__main__":
    main()
