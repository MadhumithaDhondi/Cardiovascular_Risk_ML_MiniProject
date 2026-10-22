"""Unified benchmark: tune and evaluate the same five models on either dataset.

Models: logistic regression (Ridge / Lasso / Elastic Net), Random Forest, XGBoost.
Protocol (identical for both datasets): stratified 80/20 split (seed 42), median imputation + (for logistic regression)
standardisation inside every pipeline, k-fold CV grid search on AUROC using the training part only, one final
evaluation on the held-out test part. Imbalanced data (Framingham) uses class weighting.

Run from the repository root:
    python src/benchmark/train.py --dataset cardio
    python src/benchmark/train.py --dataset framingham
    add --fast for small grids, --synthetic for a pipeline smoke test (never report those numbers)
"""
import argparse
import json
import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, confusion_matrix, f1_score,
                             precision_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    HAVE_XGB = True
except ImportError:  # only when xgboost is missing; install it via requirements.txt
    from sklearn.ensemble import HistGradientBoostingClassifier
    HAVE_XGB = False
    print("[warn] xgboost not installed - falling back to sklearn HistGradientBoosting (install xgboost for real results)")

import datasets

SEED = 42
REPO = datasets.REPO


def pipe(clf, scale=False):
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    return Pipeline(steps + [("clf", clf)])


def build_models(fast, balance, pos_weight):
    """Returns {name: (pipeline, param_grid)}; grid keys are prefixed with the pipeline step name 'clf__'."""
    cw = "balanced" if balance else None
    cs = [0.01, 0.1, 1.0] if fast else [0.001, 0.01, 0.1, 1.0, 10.0]
    lr = lambda pen: pipe(LogisticRegression(penalty=pen, solver="saga", max_iter=500, class_weight=cw), scale=True)
    models = {
        "LogReg (Ridge)": (lr("l2"), {"clf__C": cs}),
        "LogReg (Lasso)": (lr("l1"), {"clf__C": cs}),
        "LogReg (ElasticNet)": (lr("elasticnet"), {"clf__C": cs, "clf__l1_ratio": [0.25, 0.5, 0.75]}),
        "Random Forest": (pipe(RandomForestClassifier(n_estimators=100 if fast else 200, class_weight=cw, n_jobs=-1,
                                                      random_state=SEED)),
                          {"clf__max_depth": [6, 10], "clf__min_samples_leaf": [5, 20]}),
    }
    if HAVE_XGB:
        grid = {"clf__max_depth": [3, 5], "clf__learning_rate": [0.05, 0.1], "clf__n_estimators": [200, 400],
                "clf__reg_lambda": [1, 5], "clf__reg_alpha": [0, 1]}
        if fast:
            grid = {"clf__max_depth": [3, 5], "clf__learning_rate": [0.1], "clf__n_estimators": [100],
                    "clf__reg_lambda": [1], "clf__reg_alpha": [0]}
        est = XGBClassifier(eval_metric="auc", n_jobs=-1, random_state=SEED, tree_method="hist",
                            scale_pos_weight=pos_weight if balance else 1.0)
    else:
        grid = {"clf__max_depth": [3, 5], "clf__learning_rate": [0.05, 0.1], "clf__l2_regularization": [0.0, 1.0]}
        est = HistGradientBoostingClassifier(random_state=SEED, class_weight=cw)
    models["XGBoost (GBM)"] = (pipe(est), grid)
    return models


def group_auc(y, p, mask):
    if mask.sum() < 50 or len(np.unique(y[mask])) < 2:
        return np.nan
    return roc_auc_score(y[mask], p[mask])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(datasets.REGISTRY), required=True)
    ap.add_argument("--data", help="override path to the raw data file")
    ap.add_argument("--synthetic", action="store_true", help="synthetic data (smoke test only)")
    ap.add_argument("--fast", action="store_true", help="smaller grids / 5-fold CV for quick runs")
    ap.add_argument("--out", help="output folder (default results/<dataset>)")
    args = ap.parse_args()
    ds = datasets.get(args.dataset)
    out = Path(args.out) if args.out else REPO / "results" / ds.key
    out.mkdir(parents=True, exist_ok=True)

    raw = ds.synthetic() if args.synthetic else ds.load(args.data or ds.raw_path)
    json.dump(ds.quality(raw), open(out / "data_quality.json", "w"), indent=2)
    df = ds.clean(raw)
    X, y = df[ds.features], df[ds.target].values
    pos_weight = float((y == 0).sum() / max((y == 1).sum(), 1))
    print(f"[data] {ds.study}: {len(df)} rows, {len(ds.features)} features, positive rate {y.mean():.3f}")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    folds = 5 if args.fast else 10
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)

    sex_all = pd.Series(ds.sex(df), index=df.index)
    summary = {"dataset": ds.key, "study": ds.study, "title": ds.title, "pos_label": ds.pos_label,
               "n_raw": int(len(raw)), "n_clean": int(len(df)), "n_features": len(ds.features),
               "n_train": int(len(X_tr)), "n_test": int(len(X_te)), "positive_rate": float(y.mean()),
               "n_missing_rows": int(df[ds.features].isna().any(axis=1).sum()), "class_weighting": bool(ds.balance),
               "synthetic": bool(args.synthetic), "xgboost": HAVE_XGB, "cv_folds": folds,
               "male_rate": float(y[(sex_all == "Male").values].mean()),
               "female_rate": float(y[(sex_all == "Female").values].mean())}
    json.dump(summary, open(out / "data_summary.json", "w"), indent=2)

    rows, probs, best_params, fitted = [], {}, {}, {}
    for name, (est, grid) in build_models(args.fast, ds.balance, pos_weight).items():
        print(f"[train] {name} ...")
        gs = GridSearchCV(est, grid, scoring="roc_auc", cv=cv, n_jobs=-1 if name.startswith("LogReg") else 1, refit=True)
        gs.fit(X_tr, y_tr)
        p = gs.predict_proba(X_te)[:, 1]
        probs[name], fitted[name] = p, gs.best_estimator_
        best_params[name] = {k.replace("clf__", ""): (v if isinstance(v, (int, float, str)) else str(v))
                             for k, v in gs.best_params_.items()}
        pred = p >= 0.5
        tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
        rows.append({"model": name, "cv_auroc": gs.best_score_, "test_auroc": roc_auc_score(y_te, p),
                     "avg_precision": average_precision_score(y_te, p), "accuracy": accuracy_score(y_te, pred),
                     "precision": precision_score(y_te, pred, zero_division=0), "f1": f1_score(y_te, pred, zero_division=0),
                     "sensitivity": tp / (tp + fn), "specificity": tn / (tn + fp)})
        print(f"        CV AUROC {gs.best_score_:.4f} | test AUROC {rows[-1]['test_auroc']:.4f}")

    res = pd.DataFrame(rows)
    res.to_csv(out / "metrics.csv", index=False)
    json.dump(best_params, open(out / "best_params.json", "w"), indent=2)

    # Test-set predictions, so src/benchmark/visualize.py can draw every figure without retraining
    Xt = X_te.copy()
    preds = pd.DataFrame({"y_true": y_te, **probs})
    preds["sex"] = sex_all.loc[Xt.index].values
    preds["age_years"] = df.loc[Xt.index, "age_years"].values
    preds.to_csv(out / "test_predictions.csv", index=False)

    # Feature importance of the main tree model (XGBoost if installed, otherwise the Random Forest)
    imp_name = "XGBoost (GBM)" if HAVE_XGB else "Random Forest"
    clf = fitted[imp_name].named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        pd.Series(clf.feature_importances_, index=ds.features).sort_values().to_csv(out / "feature_importance.csv", header=["importance"])
        (out / "importance_source.txt").write_text(imp_name)

    # Subgroup analysis for the best model
    best = res.sort_values("test_auroc").iloc[-1]["model"]
    pb = probs[best]
    age = preds.age_years
    groups = {"Female": preds.sex == "Female", "Male": preds.sex == "Male"}
    groups.update({lab: (age >= lo) & (age < hi) for lab, lo, hi in ds.age_bins})
    sub = pd.DataFrame([{"model": best, "subgroup": g, "n": int(m.sum()), "auroc": group_auc(y_te, pb, m.values)}
                        for g, m in groups.items()])
    sub.to_csv(out / "subgroups.csv", index=False)

    # Calibration summary of the best model (class weighting inflates predicted probabilities)
    json.dump({"model": best, "mean_predicted": float(pb.mean()), "observed_rate": float(y_te.mean()),
               "brier": float(brier_score_loss(y_te, pb))}, open(out / "calibration.json", "w"), indent=2)

    print(f"\n=== {ds.study}: test results ({ds.title}) ===\n", res.round(4).to_string(index=False))
    print(f"\n=== Subgroup AUROC ({best}) ===\n", sub.round(4).to_string(index=False))
    print(f"\nResults written to {out}")


if __name__ == "__main__":
    main()
