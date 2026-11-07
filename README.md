# Dry Bean Variety Classifier

## a. Problem statement

Build and compare five supervised classification models that predict one of seven dry bean varieties from measured shape and size characteristics. The interface evaluates labeled test data and shows model metrics without fitting or changing any model.

## b. Dataset description

The project uses the UCI Machine Learning Repository's Dry Bean dataset. It contains 13,611 records, 16 numeric features, no missing values, and seven target classes: Barbunya, Bombay, Cali, Dermosan, Horoz, Seker, and Sira. The features describe bean dimensions, area, perimeter, and shape factors.

Source: [UCI Dry Bean dataset](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset) (CC BY 4.0; cite the UCI dataset if reusing it). The provided `data/dry_bean.csv` is a local copy of the source dataset. `test_data.csv` is the stratified 25% holdout split used for the recorded model evaluation and app demonstration.

## c. GitHub Repository Link

Add the public repository URL here after creating and pushing this project: **https://github.com/sagarsinha-code/ML_Classification_Model**.

## d. Models used and evaluation metrics

`train_models.py` makes one stratified 75/25 split (random state 42), trains these five classifiers, saves each fitted pipeline in `model/`, and writes metrics to `model/metadata.json`:

1. Logistic Regression (standardized inputs)
2. Decision Tree Classifier (maximum depth 16, minimum 3 examples per leaf)
3. K-Nearest Neighbors (7 neighbors, distance weighting, standardized inputs)
4. Gaussian Naive Bayes (standardized inputs)
5. Support Vector Machine (RBF kernel, C=2, probability estimates enabled)

Metrics are Accuracy, macro one-vs-rest AUC, macro Precision, macro Recall, macro F1, and Matthews Correlation Coefficient (MCC). Macro averaging gives each class equal weight. The exact values are generated on your local run and displayed in the app; copy them from the app or `model/metadata.json` into the table below.

| ML Model Name | Accuracy | AUC (macro OVR) | Precision (macro) | Recall (macro) | F1 (macro) | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9236 | 0.9948 | 0.9379 | 0.9350 | 0.9362 | 0.9077 |
| Decision Tree | 0.9016 | 0.9614 | 0.9185 | 0.9163 | 0.9173 | 0.8809 |
| KNN | 0.9201 | 0.9871 | 0.9350 | 0.9309 | 0.9327 | 0.9034 |
| Naive Bayes | 0.8998 | 0.9920 | 0.9126 | 0.9116 | 0.9110 | 0.8796 |
| Support Vector Machine | 0.9268 | 0.9949 | 0.9398 | 0.9382 | 0.9389 | 0.9115 |

### Performance observations (complete using your measured results)

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Strong baseline: 92.36% accuracy and 0.9362 macro F1. Standardization supports its linear decision boundaries. |
| Decision Tree | Lowest accuracy (90.16%) and macro F1 (0.9173) here, but the tree is comparatively easy to interpret. Its depth limit controls complexity. |
| KNN | Competitive at 92.01% accuracy and 0.9327 macro F1. Distance weighting and standardization matter because the feature scales differ. |
| Naive Bayes | AUC is high (0.9920), but macro F1 is the lowest (0.9110). Its independence assumption is a likely limitation for correlated shape measurements. |
| Support Vector Machine | Best overall on this holdout: 92.68% accuracy, 0.9949 AUC, and 0.9389 macro F1. The RBF kernel models nonlinear boundaries, with higher training cost. |
| Overall winner | Support Vector Machine has the strongest scores on this split, narrowly ahead of Logistic Regression. Confirm the choice with class-wise errors and another split before making a broad claim. |

## Run the project

Use Python 3.10 or later. In a terminal opened at the project folder:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python train_models.py
streamlit run app.py
```

The app opens locally in a browser. It loads the saved pipelines from `model/` and defaults to the included `test_data.csv`. To evaluate a different test file, upload a CSV with all 16 feature columns and a `Class` target column. The app checks required columns and known class labels, then reports results for every model. It does not retrain when you upload data.

## Repository files

```text
app.py                 Streamlit evaluation app
train_models.py        Reproducible split, training, metrics, and model saving
requirements.txt       Deployment/runtime dependencies
README.md              Project description and instructions
data/dry_bean.csv      UCI source dataset
test_data.csv          Labeled 25% holdout data for app evaluation
model/*.joblib         Five fitted model pipelines
model/metadata.json    Feature names, labels, and test metrics
```

