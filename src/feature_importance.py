
from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "models" / "logistic_regression_baseline.joblib"
REPORT_DIR = PROJECT_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Load the trained Logistic Regression pipeline
model = joblib.load(MODEL_PATH)

# Get the feature names from the original dataset
data_path = PROJECT_DIR / "data" / "raw" / "framingham_heart_disease.csv"
df = pd.read_csv(data_path)

feature_names = df.drop(columns=["TenYearCHD"]).columns

# Extract the Logistic Regression coefficients
classifier = model.named_steps["classifier"]
coefficients = classifier.coef_[0]

importance = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefficients
})

importance["Absolute_Coefficient"] = importance["Coefficient"].abs()
importance = importance.sort_values(
    "Absolute_Coefficient",
    ascending=False
)

# Save the results
output_path = REPORT_DIR / "feature_importance.csv"
importance.to_csv(output_path, index=False)

print("\nFEATURE IMPORTANCE — LOGISTIC REGRESSION")
print("=" * 60)
print(
    importance[
        ["Feature", "Coefficient"]
    ].round(4).to_string(index=False)
)

# Plot the features with the largest absolute coefficients
top_features = importance.head(10).sort_values("Coefficient")

plt.figure(figsize=(9, 6))
plt.barh(
    top_features["Feature"],
    top_features["Coefficient"]
)
plt.axvline(0, linestyle="--")
plt.xlabel("Logistic Regression Coefficient")
plt.ylabel("Feature")
plt.title("Top 10 Features by Absolute Coefficient")
plt.tight_layout()

plot_path = REPORT_DIR / "feature_importance.png"
plt.savefig(plot_path, dpi=300)
plt.close()

print("\nFiles saved:")
print("- reports/feature_importance.csv")
print("- reports/feature_importance.png")
