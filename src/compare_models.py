
from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "raw" / "framingham_heart_disease.csv"
MODEL_DIR = PROJECT_DIR / "models"
REPORT_DIR = PROJECT_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Load the original dataset and reproduce the same test split
df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["TenYearCHD"])
y = df["TenYearCHD"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Load the trained pipelines
models = {
    "Logistic Regression": joblib.load(
        MODEL_DIR / "logistic_regression_baseline.joblib"
    ),
    "Random Forest": joblib.load(
        MODEL_DIR / "random_forest.joblib"
    )
}

results = []
roc_data = {}

for name, model in models.items():
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1_Score": f1_score(
            y_test, predictions, zero_division=0
        ),
        "ROC_AUC": roc_auc_score(y_test, probabilities)
    })

    fpr, tpr, _ = roc_curve(y_test, probabilities)
    roc_data[name] = (fpr, tpr)

# Save metric comparison
comparison = pd.DataFrame(results)
comparison.to_csv(
    REPORT_DIR / "model_comparison.csv",
    index=False
)

print("\nMODEL COMPARISON")
print("=" * 65)
print(comparison.round(4).to_string(index=False))

# Plot ROC curves
plt.figure(figsize=(8, 6))

for name, (fpr, tpr) in roc_data.items():
    auc = comparison.loc[
        comparison["Model"] == name, "ROC_AUC"
    ].iloc[0]
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")

plt.plot([0, 1], [0, 1], linestyle="--", label="Random ranking")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC Curve Comparison")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(REPORT_DIR / "roc_curve_comparison.png", dpi=300)
plt.close()

# Plot confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

for ax, (name, model) in zip(axes, models.items()):
    predictions = model.predict(X_test)
    matrix = confusion_matrix(y_test, predictions)

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["No event (0)", "Event (1)"]
    )
    display.plot(ax=ax, values_format="d", colorbar=False)
    ax.set_title(name)

plt.tight_layout()
plt.savefig(REPORT_DIR / "confusion_matrices.png", dpi=300)
plt.close()

print("\nFiles saved in the reports folder:")
print("- model_comparison.csv")
print("- roc_curve_comparison.png")
print("- confusion_matrices.png")
