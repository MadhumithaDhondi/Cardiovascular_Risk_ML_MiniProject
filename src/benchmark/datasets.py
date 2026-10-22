"""Dataset registry for the unified benchmark.

Two studies share one pipeline (train.py / visualize.py / make_report.py / make_slides.py):
  cardio      Study A - Kaggle Cardiovascular Disease dataset (70k rows, balanced, label = CVD present now)
  framingham  Study B - Framingham Heart Study (4.2k rows, 15% positive, missing values, label = CHD within 10 years)

Each entry says where the file is, how to clean it, what the features/target are, how to define the gender and age
subgroups, and a few plotting hints. Add a new dataset by adding one more Dataset(...) to REGISTRY.
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]


@dataclass
class Dataset:
    key: str
    title: str
    study: str
    raw_path: Path
    target: str
    features: list
    pos_label: str           # name of the positive class in plots / text
    balance: bool            # True -> class weighting (imbalanced labels)
    age_bins: list           # [(label, lo, hi)] used for the subgroup check
    eda: dict                # plotting hints for fig_eda
    abbr: dict = field(default_factory=dict)  # short feature names for the correlation matrix
    load: callable = None
    clean: callable = None
    quality: callable = None
    sex: callable = None     # df -> Series of "Female"/"Male"
    synthetic: callable = None


# ---------------------------------------------------------------------------------------------------------------
# Study A: Kaggle Cardiovascular Disease dataset
# ---------------------------------------------------------------------------------------------------------------
CARDIO_FEATURES = ["age_years", "gender", "height", "weight", "bmi", "ap_hi", "ap_lo", "pulse_pressure",
                   "cholesterol", "gluc", "smoke", "alco", "active"]


def cardio_load(path):
    df = pd.read_csv(path, sep=";")
    if df.shape[1] == 1:  # fallback if comma separated
        df = pd.read_csv(path)
    return df


def cardio_clean(df):
    """Convert units, remove physiologically implausible rows, add engineered features."""
    df = df.copy()
    if "id" in df.columns:
        df = df.drop(columns=["id"])
    if "age" in df.columns:  # age is stored in days in the Kaggle file
        df["age_years"] = (df["age"] / 365.25).round(1)
        df = df.drop(columns=["age"])
    n0 = len(df)
    df = df[(df.height.between(120, 220)) & (df.weight.between(30, 200))]
    df = df[(df.ap_hi.between(70, 250)) & (df.ap_lo.between(40, 200)) & (df.ap_hi > df.ap_lo)]
    df = df.drop_duplicates()
    df["bmi"] = df.weight / (df.height / 100) ** 2
    df["pulse_pressure"] = df.ap_hi - df.ap_lo
    print(f"[clean] kept {len(df)} of {n0} rows ({n0 - len(df)} removed as implausible/duplicate)")
    return df.reset_index(drop=True)


def cardio_quality(raw):
    """Count rows flagged by each cleaning rule (rules overlap, so counts need not sum to the total removed)."""
    d = raw.copy()
    rules = {
        "Missing values": d.isna().any(axis=1),
        "Duplicate rows": d.drop(columns=["id"], errors="ignore").duplicated(),
        "Height outside 120-220 cm": ~d.height.between(120, 220),
        "Weight outside 30-200 kg": ~d.weight.between(30, 200),
        "Systolic BP outside 70-250": ~d.ap_hi.between(70, 250),
        "Diastolic BP outside 40-200": ~d.ap_lo.between(40, 200),
        "Systolic <= diastolic BP": d.ap_hi <= d.ap_lo,
    }
    flags = pd.DataFrame(rules)
    return {"counts": {k: int(v.sum()) for k, v in rules.items()}, "any_flagged": int(flags.any(axis=1).sum()),
            "n_raw": int(len(raw)), "title": "Rows flagged per cleaning rule", "action": "removed"}


def cardio_synthetic(n=20000, seed=0):
    """Synthetic stand-in with the same schema; ONLY for smoke-testing the pipeline."""
    rng = np.random.default_rng(seed)
    age = rng.uniform(30, 65, n)
    gender = rng.choice([1, 2], n, p=[0.65, 0.35])
    height = rng.normal(165, 8, n) + (gender == 2) * 8
    weight = rng.normal(74, 14, n)
    ap_hi = rng.normal(125, 15, n) + (age - 50) * 0.5
    ap_lo = ap_hi - rng.normal(45, 8, n)
    chol = rng.choice([1, 2, 3], n, p=[0.75, 0.13, 0.12])
    gluc = rng.choice([1, 2, 3], n, p=[0.85, 0.07, 0.08])
    smoke = rng.binomial(1, 0.09, n)
    alco = rng.binomial(1, 0.05, n)
    active = rng.binomial(1, 0.8, n)
    z = (0.06 * (age - 50) + 0.05 * (ap_hi - 125) + 0.5 * (chol - 1) + 0.2 * (gluc - 1)
         + 0.03 * (weight - 74) - 0.3 * active + 0.2 * smoke)
    y = rng.binomial(1, 1 / (1 + np.exp(-z)))
    return pd.DataFrame({"age": age * 365.25, "gender": gender, "height": height.round(),
                         "weight": weight.round(1), "ap_hi": ap_hi.round(), "ap_lo": ap_lo.round(),
                         "cholesterol": chol, "gluc": gluc, "smoke": smoke, "alco": alco,
                         "active": active, "cardio": y})


# ---------------------------------------------------------------------------------------------------------------
# Study B: Framingham Heart Study
# ---------------------------------------------------------------------------------------------------------------
FRAMINGHAM_FEATURES = ["male", "age", "education", "currentSmoker", "cigsPerDay", "BPMeds", "prevalentStroke",
                       "prevalentHyp", "diabetes", "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"]


def fram_load(path):
    return pd.read_csv(path)


def fram_clean(df):
    """No rows are dropped: missing values are imputed inside the model pipeline (train-fold median), never before
    the split, to avoid leakage. This keeps the split identical to the original baseline scripts (seed 42)."""
    df = df.copy()
    n0 = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    df["age_years"] = df["age"]  # common name used by the subgroup code
    print(f"[clean] kept {len(df)} of {n0} rows; {int(df[FRAMINGHAM_FEATURES].isna().any(axis=1).sum())} rows contain "
          f"missing values (imputed in-pipeline)")
    return df


def fram_quality(raw):
    miss = raw.isna().sum()
    counts = {f"{c} (missing)": int(v) for c, v in miss.items() if v > 0}
    counts["Duplicate rows"] = int(raw.duplicated().sum())
    return {"counts": counts, "any_flagged": int(raw.isna().any(axis=1).sum()), "n_raw": int(len(raw)),
            "title": "Rows with missing values, by column", "action": "imputed"}


def fram_synthetic(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    age = rng.integers(32, 71, n)
    male = rng.binomial(1, 0.43, n)
    sysbp = rng.normal(132, 22, n) + (age - 50) * 0.4
    chol = rng.normal(236, 44, n)
    cigs = rng.binomial(1, 0.5, n) * rng.integers(1, 40, n)
    z = -2.9 + 0.06 * (age - 50) + 0.02 * (sysbp - 132) + 0.4 * male + 0.02 * cigs
    y = rng.binomial(1, 1 / (1 + np.exp(-z)))
    df = pd.DataFrame({"male": male, "age": age, "education": rng.integers(1, 5, n).astype(float),
                       "currentSmoker": (cigs > 0).astype(int), "cigsPerDay": cigs.astype(float),
                       "BPMeds": rng.binomial(1, 0.03, n).astype(float), "prevalentStroke": rng.binomial(1, 0.006, n),
                       "prevalentHyp": (sysbp > 140).astype(int), "diabetes": rng.binomial(1, 0.03, n),
                       "totChol": chol, "sysBP": sysbp, "diaBP": sysbp - rng.normal(48, 8, n),
                       "BMI": rng.normal(25.8, 4, n), "heartRate": rng.normal(76, 12, n),
                       "glucose": rng.normal(82, 20, n), "TenYearCHD": y})
    for c in ["glucose", "education", "totChol"]:
        df.loc[rng.random(n) < 0.05, c] = np.nan
    return df


REGISTRY = {
    "cardio": Dataset(
        key="cardio", title="Kaggle Cardiovascular Disease", study="Study A",
        raw_path=REPO / "data" / "raw" / "cardio_train.csv", target="cardio", features=CARDIO_FEATURES,
        pos_label="CVD", balance=False,
        age_bins=[("Age <50", 0, 50), ("Age 50-59", 50, 60), ("Age 60+", 60, 200)],
        eda={"hist1": ("age_years", "Age (years)", None), "hist2": ("ap_hi", "Systolic BP (mmHg)", np.arange(70, 251, 5)),
             "bar": ("cholesterol", {1: "Normal", 2: "Above normal", 3: "Well above"}, "CVD rate by cholesterol level")},
        abbr={"pulse_pressure": "PP", "age_years": "age", "gender": "sex", "cholesterol": "chol", "cardio": "CVD"},
        load=cardio_load, clean=cardio_clean, quality=cardio_quality,
        sex=lambda d: np.where(d["gender"] == 1, "Female", "Male"), synthetic=cardio_synthetic),
    "framingham": Dataset(
        key="framingham", title="Framingham Heart Study", study="Study B",
        raw_path=REPO / "data" / "raw" / "framingham_heart_disease.csv", target="TenYearCHD",
        features=FRAMINGHAM_FEATURES, pos_label="10-yr CHD", balance=True,
        age_bins=[("Age <45", 0, 45), ("Age 45-54", 45, 55), ("Age 55+", 55, 200)],
        eda={"hist1": ("age", "Age (years)", None), "hist2": ("sysBP", "Systolic BP (mmHg)", np.arange(80, 251, 5)),
             "bar": ("prevalentHyp", {0: "No hypertension", 1: "Hypertension"}, "10-yr CHD rate by hypertension")},
        abbr={"currentSmoker": "smoker", "cigsPerDay": "cigs", "prevalentStroke": "stroke", "prevalentHyp": "hyp",
              "education": "educ", "heartRate": "HR", "glucose": "gluc", "totChol": "chol", "TenYearCHD": "CHD"},
        load=fram_load, clean=fram_clean, quality=fram_quality,
        sex=lambda d: np.where(d["male"] == 1, "Male", "Female"), synthetic=fram_synthetic),
}


def get(key):
    if key not in REGISTRY:
        raise SystemExit(f"unknown dataset '{key}' (choose from {list(REGISTRY)})")
    return REGISTRY[key]
