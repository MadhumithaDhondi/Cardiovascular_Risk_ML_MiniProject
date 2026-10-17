
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "raw" / "framingham_heart_disease.csv"

df = pd.read_csv(DATA_PATH)

# Calculate missing values for each column
quality_report = pd.DataFrame({
    "Missing_Count": df.isnull().sum(),
    "Missing_Percentage": (df.isnull().mean() * 100).round(2),
    "Data_Type": df.dtypes.astype(str)
})

print("\nDATA QUALITY REPORT")
print("=" * 60)
print(quality_report.to_string())

# Check for duplicate records
print("\nDuplicate rows:", df.duplicated().sum())

# Check the target distribution
print("\nTARGET DISTRIBUTION")
print(df["TenYearCHD"].value_counts().sort_index())

# Save the report
REPORT_DIR = PROJECT_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_PATH = REPORT_DIR / "data_quality_report.csv"
quality_report.to_csv(REPORT_PATH, index=True, index_label="Column")

print(f"\nReport saved to: {REPORT_PATH}")
