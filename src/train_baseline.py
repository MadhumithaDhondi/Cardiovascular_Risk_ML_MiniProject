
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
import joblib

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "raw" / "framingham_heart_disease.csv"
MODEL_DIR = PROJECT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Load the original data
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["TenYearCHD"])
y = df["TenYearCHD"]

# Create a reproducible, stratified train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Keep preprocessing inside the pipeline
model = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    ))
])

# Cross-validation on training data only
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1
)

print("LOGISTIC REGRESSION — BASELINE")
print("=" * 45)
print("5-fold cross-validation ROC-AUC scores:", cv_scores.round(4))
print(f"Mean cross-validation ROC-AUC: {cv_scores.mean():.4f}")
print(f"Standard deviation: {cv_scores.std():.4f}")

# Train on all training data
model.fit(X_train, y_train)

# Evaluate once on the held-out test set
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\nTEST SET RESULTS")
print("=" * 45)
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred, zero_division=0):.4f}")
print(f"F1-score:  {f1_score(y_test, y_pred, zero_division=0):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_prob):.4f}")

print("\nCONFUSION MATRIX")
print("[[True negatives, False positives],")
print(" [False negatives, True positives]]")
print(confusion_matrix(y_test, y_pred))

print("\nCLASSIFICATION REPORT")
print(classification_report(y_test, y_pred, zero_division=0))

# Save the complete pipeline
model_path = MODEL_DIR / "logistic_regression_baseline.joblib"
joblib.dump(model, model_path)
print(f"Saved trained pipeline to: {model_path}")
