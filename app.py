"""Streamlit interface for evaluating the saved Dry Bean classifiers."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
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

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
META_PATH = MODEL_DIR / "metadata.json"

st.set_page_config(page_title="Dry Bean Classifier Lab", page_icon="🫘", layout="wide")


@st.cache_data
def load_metadata():
    with META_PATH.open(encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def load_model(filename: str):
    return joblib.load(MODEL_DIR / filename)


def safe_auc(y_true, probabilities, class_ids):
    try:
        return float(roc_auc_score(y_true, probabilities, labels=class_ids, multi_class="ovr", average="macro"))
    except ValueError:
        return float("nan")


def score(model, X, y, classes):
    y_pred = model.predict(X)
    probabilities = model.predict_proba(X)
    class_ids = list(range(len(classes)))
    metrics = {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC (macro OVR)": safe_auc(y, probabilities, class_ids),
        "Precision (macro)": precision_score(y, y_pred, average="macro", zero_division=0),
        "Recall (macro)": recall_score(y, y_pred, average="macro", zero_division=0),
        "F1 (macro)": f1_score(y, y_pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y, y_pred),
    }
    matrix = confusion_matrix(y, y_pred, labels=class_ids)
    report = classification_report(
        y, y_pred, labels=class_ids, target_names=classes, output_dict=True, zero_division=0
    )
    return metrics, matrix, report


st.title("Dry Bean Variety Classifier")
st.write(
    "Compare five trained classifiers on labeled holdout data. Models are loaded from disk; "
    "the app evaluates uploaded rows and never retrains on them."
)

if not META_PATH.exists():
    st.error("Model files are missing. Run `python train_models.py` first.")
    st.stop()

metadata = load_metadata()
classes = metadata["classes"]
feature_names = metadata["features"]
model_names = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "k_nearest_neighbors.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}

with st.sidebar:
    st.header("Evaluation setup")
    st.caption(f"Dataset: {metadata['dataset']}")
    st.caption(f"{metadata['n_rows']:,} rows | {metadata['n_features']} numeric features | {len(classes)} classes")
    selected_model = st.selectbox("Inspect one model", list(model_names))

uploaded = st.file_uploader("Upload labeled test data (CSV)", type=["csv"])
if uploaded is None:
    sample_path = ROOT / "test_data.csv"
    if sample_path.exists():
        st.info("Showing the supplied holdout sample. Upload your own test CSV to evaluate another labeled sample.")
        data = pd.read_csv(sample_path)
    else:
        st.warning("Upload a CSV with the 16 feature columns and a `Class` target column.")
        st.stop()
else:
    try:
        data = pd.read_csv(uploaded, sep=None, engine="python")
    except Exception as exc:
        st.error(f"Could not read the CSV: {exc}")
        st.stop()

data.columns = data.columns.astype(str).str.replace("^\ufeff", "", regex=True).str.strip()
required_columns = feature_names + [metadata["target"]]
missing = [column for column in required_columns if column not in data.columns]
if missing:
    detected = ", ".join(data.columns.astype(str))
    st.error(
        "The CSV must have a header row containing the 16 feature columns and `Class`. "
        "Comma-, semicolon-, and tab-separated files are supported. "
        f"Missing columns: {', '.join(missing)}. Detected columns: {detected}"
    )
    st.stop()
if data.empty:
    st.error("The uploaded CSV has no rows.")
    st.stop()

try:
    X_eval = data[feature_names].apply(pd.to_numeric, errors="raise")
except Exception as exc:
    st.error(f"Every feature column must contain numeric values: {exc}")
    st.stop()
y_labels = data[metadata["target"]].astype(str)
unknown = sorted(set(y_labels) - set(classes))
if unknown:
    st.error("Unknown target labels found: " + ", ".join(unknown))
    st.stop()
class_to_id = {label: idx for idx, label in enumerate(classes)}
y_eval = y_labels.map(class_to_id).to_numpy()

st.subheader("Uploaded data")
st.write(f"Rows: **{len(data):,}** | Features checked: **{len(feature_names)}** | Target: **{metadata['target']}**")
with st.expander("Preview CSV"):
    st.dataframe(data.head(20), use_container_width=True)

all_rows = []
all_scored = {}
try:
    for name, filename in model_names.items():
        model = load_model(filename)
        metrics, matrix, report = score(model, X_eval, y_eval, classes)
        all_rows.append({"Model": name, **metrics})
        all_scored[name] = (metrics, matrix, report, model)
except FileNotFoundError as exc:
    st.error(f"A saved model file was not found: {exc}. Run `python train_models.py`.")
    st.stop()

st.subheader("All-model comparison")
comparison = pd.DataFrame(all_rows).set_index("Model")
st.dataframe(comparison.style.format("{:.4f}"), use_container_width=True)

st.subheader(f"Details: {selected_model}")
metrics, matrix, report, selected = all_scored[selected_model]
cols = st.columns(6)
for col, (label, value) in zip(cols, metrics.items()):
    col.metric(label, f"{value:.4f}")

left, right = st.columns([1, 1])
with left:
    st.markdown("**Confusion matrix** (true rows × predicted columns)")
    st.dataframe(pd.DataFrame(matrix, index=classes, columns=classes), use_container_width=True)
with right:
    st.markdown("**Classification report**")
    st.dataframe(pd.DataFrame(report).T.round(4), use_container_width=True)

with st.expander("How to read these metrics"):
    st.markdown(
        "Accuracy is the overall fraction correct. Precision, recall, and F1 are macro-averaged "
        "across bean classes. AUC is macro one-vs-rest AUC. MCC summarizes the confusion matrix "
        "and ranges from -1 to 1. Use the same labeled test CSV to compare models fairly."
    )
