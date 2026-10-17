
from pathlib import Path
import pandas as pd

# Find the project folder and dataset
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "raw" / "framingham_heart_disease.csv"

# Load the dataset
df = pd.read_csv(DATA_PATH)

print("=" * 50)
print("CARDIOVASCULAR RISK DATASET")
print("=" * 50)

print("\n1. Dataset loaded successfully!")
print(f"Number of rows: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")

print("\n2. Column names:")
print(df.columns.tolist())

print("\n3. First 5 rows:")
print(df.head().to_string())

print("\n4. Data types and non-missing values:")
df.info()

print("\n5. Missing values in each column:")
print(df.isnull().sum())

print("\n6. Target variable distribution:")
if "TenYearCHD" in df.columns:
    print(df["TenYearCHD"].value_counts(dropna=False))
else:
    print("TenYearCHD column not found. Check the column names above.")
