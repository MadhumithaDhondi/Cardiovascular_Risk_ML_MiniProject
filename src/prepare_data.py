
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "raw" / "framingham_heart_disease.csv"
OUTPUT_DIR = PROJECT_DIR / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load the original dataset
df = pd.read_csv(DATA_PATH)

# Separate input features and target
X = df.drop(columns=["TenYearCHD"])
y = df["TenYearCHD"]

# Split before fitting preprocessing steps to avoid data leakage
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Impute missing values and scale numerical features
preprocessor = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Learn preprocessing values from training data only
X_train_prepared = preprocessor.fit_transform(X_train)

# Apply the same transformations to test data
X_test_prepared = preprocessor.transform(X_test)

# Convert prepared arrays back to DataFrames
X_train_prepared = pd.DataFrame(
    X_train_prepared,
    columns=X.columns,
    index=X_train.index
)

X_test_prepared = pd.DataFrame(
    X_test_prepared,
    columns=X.columns,
    index=X_test.index
)

# Save prepared datasets
X_train_prepared.to_csv(OUTPUT_DIR / "X_train.csv", index=False)
X_test_prepared.to_csv(OUTPUT_DIR / "X_test.csv", index=False)
y_train.to_csv(OUTPUT_DIR / "y_train.csv", index=False)
y_test.to_csv(OUTPUT_DIR / "y_test.csv", index=False)

print("DATA PREPARATION COMPLETE")
print("=" * 40)
print(f"Original dataset: {df.shape}")
print(f"Training features: {X_train_prepared.shape}")
print(f"Testing features: {X_test_prepared.shape}")
print(f"Training target: {y_train.shape}")
print(f"Testing target: {y_test.shape}")

print("\nMissing values after imputation:")
print("Training:", int(X_train_prepared.isnull().sum().sum()))
print("Testing:", int(X_test_prepared.isnull().sum().sum()))

print("\nTraining target distribution:")
print(y_train.value_counts().sort_index())

print("\nTesting target distribution:")
print(y_test.value_counts().sort_index())

print(f"\nPrepared files saved in: {OUTPUT_DIR}")
