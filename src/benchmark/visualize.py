"""Draw all figures for one dataset from the saved results (no retraining). Run AFTER train.py.

    python src/benchmark/visualize.py --dataset cardio
    python src/benchmark/visualize.py --dataset framingham

Writes PNGs to results/<dataset>/figures/:
  fig_data_quality.png   rows flagged / missing values per column, (cardio: before/after outliers), class balance
  fig_eda.png            age and BP by outcome, outcome rate by a key factor, correlation matrix
  fig_models.png         CV vs test AUROC, threshold metrics, ROC curves, precision-recall curves
  fig_confusion.png      confusion matrix for every model (threshold 0.5)
  fig_diagnostics.png    feature importance, subgroup AUROC, calibration curve of the best model
"""
import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (average_precision_score, confusion_matrix, precision_recall_curve, roc_auc_score,
                             roc_curve)

import datasets

plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.titleweight": "bold", "axes.spines.top": False,
                     "axes.spines.right": False})
COL = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#d62728"]
SHORT = {"LogReg (Ridge)": "LR-Ridge", "LogReg (Lasso)": "LR-Lasso", "LogReg (ElasticNet)": "LR-EN",
         "Random Forest": "RF", "XGBoost (GBM)": "XGB"}


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(datasets.REGISTRY), required=True)
    ap.add_argument("--data", help="override path to the raw data file")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--results", help="results folder (default results/<dataset>)")
    a = ap.parse_args()
    ds = datasets.get(a.dataset)
    R = Path(a.results) if a.results else datasets.REPO / "results" / ds.key
    out = R / "figures"
    out.mkdir(parents=True, exist_ok=True)

    raw = ds.synthetic() if a.synthetic else ds.load(a.data or ds.raw_path)
    df = ds.clean(raw)
    T = ds.target
    dq = json.load(open(R / "data_quality.json"))
    m = pd.read_csv(R / "metrics.csv")
    sub = pd.read_csv(R / "subgroups.csv")
    pr = pd.read_csv(R / "test_predictions.csv")
    models = list(m.model)
    y = pr.y_true.values
    colors = dict(zip(models, COL))
    best = m.sort_values("test_auroc").iloc[-1].model
    posname = ds.pos_label

    # 1. Data quality ---------------------------------------------------------------------------
    if ds.key == "cardio":
        fig, ax = plt.subplots(1, 3, figsize=(11, 2.5), gridspec_kw={"width_ratios": [1.5, 1.2, 0.8]})
    else:
        fig, ax = plt.subplots(1, 2, figsize=(11, 2.5), gridspec_kw={"width_ratios": [1.6, 1.0]})
    c = pd.Series(dq["counts"]).sort_values()
    ax[0].barh(c.index, c.values, color="#c0504d")
    for i, v in enumerate(c.values):
        ax[0].text(v, i, f" {v:,}", va="center", fontsize=7)
    ax[0].set_title(f"{dq['title']} ({dq['any_flagged']:,} of {dq['n_raw']:,} rows {dq['action']})")
    ax[0].set_xlim(0, max(c.max() * 1.25, 5))
    bal = df[T].value_counts().sort_index()
    if ds.key == "cardio":
        ax[1].hist(raw.ap_hi.clip(0, 300), bins=60, alpha=0.6, label="raw", color="#999999")
        ax[1].hist(df.ap_hi, bins=60, alpha=0.7, label="cleaned", color="#1f77b4")
        ax[1].set_yscale("log"); ax[1].set_xlabel("Systolic BP (raw clipped at 300)"); ax[1].legend()
        ax[1].set_title("Outliers before vs after cleaning")
        cb = ax[2]
    else:
        cb = ax[1]
    cb.bar(["No " + posname, posname], bal.values, color=["#2ca02c", "#d62728"])
    for i, v in enumerate(bal.values):
        cb.text(i, v, f"{v / bal.sum() * 100:.1f}%", ha="center", va="bottom")
    cb.set_title("Class balance")
    save(fig, out / "fig_data_quality.png")

    # 2. EDA ------------------------------------------------------------------------------------
    fig, ax = plt.subplots(1, 4, figsize=(12, 2.6))
    (c1, l1, _), (c2, l2, bins2) = ds.eda["hist1"], ds.eda["hist2"]
    for v, lab, col in [(0, "No " + posname, "#2ca02c"), (1, posname, "#d62728")]:
        ax[0].hist(df[df[T] == v][c1], bins=30, alpha=0.55, label=lab, color=col)
        ax[1].hist(df[df[T] == v][c2].dropna(), bins=bins2, alpha=0.55, label=lab, color=col)
    ax[0].set_xlabel(l1); ax[0].set_title(f"{l1.split(' (')[0]} by outcome"); ax[0].legend(fontsize=7)
    ax[1].set_xlabel(l2); ax[1].set_title(f"{l2.split(' (')[0]} by outcome")
    bcol, blabels, btitle = ds.eda["bar"]
    r = df.groupby(bcol)[T].mean() * 100
    ax[2].bar([blabels[k] for k in r.index], r.values, color="#1f77b4")
    for i, v in enumerate(r.values):
        ax[2].text(i, v, f"{v:.0f}%", ha="center", va="bottom")
    ax[2].set_ylabel(f"{posname} rate (%)"); ax[2].set_title(btitle, fontsize=8)
    cols = ds.features + [T]
    cm_ = df[cols].corr()
    im = ax[3].imshow(cm_, cmap="RdBu_r", vmin=-1, vmax=1)
    short = [ds.abbr.get(c_, c_[:6]) for c_ in cm_.columns]
    ax[3].set_xticks(range(len(cm_))); ax[3].set_yticks(range(len(cm_)))
    ax[3].set_xticklabels(short, rotation=90, fontsize=5.5); ax[3].set_yticklabels(short, fontsize=5.5)
    ax[3].set_title("Correlation matrix"); fig.colorbar(im, ax=ax[3], fraction=0.046)
    save(fig, out / "fig_eda.png")

    # 3. Model comparison + ROC + PR ------------------------------------------------------------
    fig, ax = plt.subplots(1, 4, figsize=(13, 2.8))
    x = np.arange(len(m)); w = 0.38
    ax[0].bar(x - w / 2, m.cv_auroc, w, label="CV", color="#9ecae1")
    ax[0].bar(x + w / 2, m.test_auroc, w, label="Test", color="#08519c")
    ax[0].set_xticks(x); ax[0].set_xticklabels([SHORT.get(n, n) for n in m.model], fontsize=7)
    ax[0].set_ylim(max(0.5, min(m.cv_auroc.min(), m.test_auroc.min()) - 0.05), max(m.cv_auroc.max(), m.test_auroc.max()) + 0.03)
    ax[0].legend(fontsize=7, loc="upper right"); ax[0].set_title("AUROC: CV vs test")
    for i, te in enumerate(m.test_auroc):
        ax[0].text(i + w / 2, te, f"{te:.3f}", ha="center", va="bottom", fontsize=6)
    w3 = 0.27
    for j, (k, lab) in enumerate([("accuracy", "Accuracy"), ("sensitivity", "Sensitivity"), ("specificity", "Specificity")]):
        ax[1].bar(x + (j - 1) * w3, m[k], w3, label=lab)
    ax[1].set_xticks(x); ax[1].set_xticklabels([SHORT.get(n, n) for n in m.model], fontsize=7)
    ax[1].set_ylim(0, 1.18); ax[1].legend(fontsize=6, ncol=3, loc="upper center"); ax[1].set_title("Metrics at threshold 0.5")
    for n in models:
        p = pr[n].values
        f, t, _ = roc_curve(y, p)
        ax[2].plot(f, t, color=colors[n], label=f"{SHORT[n]} ({roc_auc_score(y, p):.3f})")
        pp, rr, _ = precision_recall_curve(y, p)
        ax[3].plot(rr, pp, color=colors[n], label=f"{SHORT[n]} ({average_precision_score(y, p):.3f})")
    ax[2].plot([0, 1], [0, 1], "k--", lw=0.7); ax[2].set_xlabel("False positive rate"); ax[2].set_ylabel("True positive rate")
    ax[2].legend(fontsize=6, loc="lower right", title="AUROC", title_fontsize=6); ax[2].set_title("ROC curves (test)")
    ax[3].axhline(y.mean(), color="k", ls="--", lw=0.7)
    ax[3].set_xlabel("Recall"); ax[3].set_ylabel("Precision"); ax[3].legend(fontsize=6, title="Avg precision", title_fontsize=6)
    ax[3].set_title("Precision-recall (dashed = prevalence)")
    save(fig, out / "fig_models.png")

    # 4. Confusion matrices ---------------------------------------------------------------------
    fig, ax = plt.subplots(1, len(models), figsize=(2.6 * len(models), 2.5))
    ax = np.atleast_1d(ax)
    for k, n in enumerate(models):
        cmx = confusion_matrix(y, pr[n].values >= 0.5)
        ax[k].imshow(cmx / cmx.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
        for (i, j), v in np.ndenumerate(cmx):
            frac = v / cmx[i].sum()
            ax[k].text(j, i, f"{v:,}\n({frac * 100:.0f}%)", ha="center", va="center",
                       color="white" if frac > 0.5 else "black", fontsize=7)
        ax[k].set_xticks([0, 1]); ax[k].set_yticks([0, 1])
        ax[k].set_xticklabels(["Pred 0", "Pred 1"]); ax[k].set_yticklabels(["True 0", "True 1"])
        ax[k].set_title(n, fontsize=7.5)
        for s in ax[k].spines.values():
            s.set_visible(False)
    save(fig, out / "fig_confusion.png")

    # 5. Diagnostics: importance, subgroup AUROC, calibration ----------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(11, 2.6))
    fi = R / "feature_importance.csv"
    if fi.exists():
        imp = pd.read_csv(fi, index_col=0).importance.sort_values()
        ax[0].barh(imp.index, imp.values, color="#1f77b4")
        src = (R / "importance_source.txt").read_text() if (R / "importance_source.txt").exists() else "model"
        ax[0].set_title(f"{src.split(' (')[0]} feature importance")
        ax[0].tick_params(axis="y", labelsize=6.5)
    else:
        ax[0].axis("off"); ax[0].text(0.5, 0.5, "feature_importance.csv not found", ha="center")
    ax[1].barh(sub.subgroup[::-1], sub.auroc[::-1], color=["#d62728" if "Age" in s else "#1f77b4" for s in sub.subgroup[::-1]])
    for i, v in enumerate(sub.auroc[::-1]):
        ax[1].text(v, i, f" {v:.3f}", va="center", fontsize=7)
    ax[1].set_xlim(max(0.4, sub.auroc.min() - 0.1), sub.auroc.max() + 0.06)
    ax[1].set_title(f"Subgroup AUROC ({SHORT[best]})")
    fr, mp = calibration_curve(y, pr[best].values, n_bins=10 if len(y) > 5000 else 6, strategy="quantile")
    ax[2].plot([0, 1], [0, 1], "k--", lw=0.7); ax[2].plot(mp, fr, "o-", color=colors[best])
    ax[2].set_xlabel("Mean predicted probability"); ax[2].set_ylabel(f"Observed {posname} rate")
    ax[2].set_title(f"Calibration ({SHORT[best]})")
    save(fig, out / "fig_diagnostics.png")


if __name__ == "__main__":
    main()
