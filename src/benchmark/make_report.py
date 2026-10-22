"""Build the combined 2-page PDF write-up (Study A + Study B) from results/*. Run AFTER train.py + visualize.py for BOTH datasets.

    python src/benchmark/make_report.py --names "Name1 (SRN1), Name2 (SRN2)" --team 12
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

import datasets

REPO = datasets.REPO
FW = 15.4 * cm  # figure width
SHORT = {"LogReg (Ridge)": "LR-Ridge", "LogReg (Lasso)": "LR-Lasso", "LogReg (ElasticNet)": "LR-EN",
         "Random Forest": "RF", "XGBoost (GBM)": "XGB"}


def load(R):
    R = Path(R)
    return {"sum": json.load(open(R / "data_summary.json")), "m": pd.read_csv(R / "metrics.csv"),
            "sub": pd.read_csv(R / "subgroups.csv"), "dq": json.load(open(R / "data_quality.json")),
            "cal": json.load(open(R / "calibration.json")), "fig": R / "figures"}


def fig(path, width=FW):
    w, h = ImageReader(str(path)).getSize()
    return Image(str(path), width, width * h / w)


def tbl(rows, widths, fs=7.2, header_rows=1, span=None):
    t = Table(rows, colWidths=widths, hAlign="LEFT")
    style = [("FONTSIZE", (0, 0), (-1, -1), fs), ("BACKGROUND", (0, 0), (-1, header_rows - 1), colors.HexColor("#dde6f5")),
             ("GRID", (0, 0), (-1, -1), 0.3, colors.grey), ("TOPPADDING", (0, 0), (-1, -1), 1.2),
             ("BOTTOMPADDING", (0, 0), (-1, -1), 1.2), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]
    t.setStyle(TableStyle(style + (span or [])))
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cardio", default=str(REPO / "results" / "cardio"))
    ap.add_argument("--framingham", default=str(REPO / "results" / "framingham"))
    ap.add_argument("--out", default=str(REPO / "reports" / "writeup.pdf"))
    ap.add_argument("--names", default="<Team member names and SRNs>")
    ap.add_argument("--team", default="<Team no.>")
    a = ap.parse_args()
    A, B = load(a.cardio), load(a.framingham)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    sa, sb = A["sum"], B["sum"]
    synth = sa["synthetic"] or sb["synthetic"]
    gbm = "XGBoost" if sa["xgboost"] else "gradient boosting"

    def best(D):
        return D["m"].sort_values("test_auroc").iloc[-1]

    def row(D, prefix):
        r = D["m"][D["m"].model.str.startswith(prefix)]
        return r.sort_values("test_auroc").iloc[-1]

    def sub_txt(D):
        s = D["sub"]
        gen, age = s[s.subgroup.isin(["Female", "Male"])], s[s.subgroup.str.startswith("Age")]
        low = age.sort_values("auroc").iloc[0]
        return (f"gender {gen.auroc.min():.3f}-{gen.auroc.max():.3f}; age {age.auroc.min():.3f}-{age.auroc.max():.3f} "
                f"(weakest: {low.subgroup}, {low.auroc:.3f})")

    ss = getSampleStyleSheet()
    body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=8.2, leading=10, spaceAfter=2)
    cap = ParagraphStyle("c", parent=body, fontSize=6.8, leading=8, textColor=colors.HexColor("#444444"), spaceAfter=2)
    h = ParagraphStyle("h", parent=ss["Heading3"], fontSize=9.6, spaceBefore=3, spaceAfter=1)
    title = ParagraphStyle("t", parent=ss["Title"], fontSize=13.5, spaceAfter=0, leading=16)
    doc = SimpleDocTemplate(a.out, pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm, topMargin=1.0 * cm, bottomMargin=0.9 * cm)
    P = lambda t, s=body: Paragraph(t, s)
    note = " <b>NOTE: contains synthetic smoke-test data; do not submit.</b>" if synth else ""

    removed = sa["n_raw"] - sa["n_clean"]
    story = [P("Predicting Cardiovascular Risk using Electronic Health Records", title),
             P(f"UE24CS352A Machine Learning - Mini-Project | Team {a.team} | {a.names}" + note, ss["Normal"]),
             P("1. Problem statement", h),
             P("Cardiovascular disease (CVD) is the leading cause of death worldwide, and identifying at-risk patients early enables prevention. "
               "We predict cardiovascular risk from routinely recorded clinical data (binary classification) and ask two questions: which models work best, "
               "and do the conclusions hold across two quite different datasets? Following Nguyen (Stanford CS229) on EHR-based CVD risk prediction, we compare "
               "interpretable regularised logistic regression with tree ensembles. The original STRIDE EHR data is private, so the project is split into "
               "<b>Study A</b> (prevalent CVD, large balanced data) and <b>Study B</b> (10-year incident CHD, small imbalanced data with missing values), "
               "which together cover the paper's setting."),
             P("2. Datasets", h)]
    rows = [["", "Study A: Kaggle Cardiovascular Disease", "Study B: Framingham Heart Study"],
            ["Task / label", "CVD present now (prevalent)", "CHD within 10 years (incident)"],
            ["Records", f"{sa['n_raw']:,} raw -> {sa['n_clean']:,} after cleaning", f"{sb['n_raw']:,} (none dropped)"],
            ["Features", f"{sa['n_features']} (adds BMI, pulse pressure)", f"{sb['n_features']} (smoking, diabetes, BP meds, cholesterol, ...)"],
            ["Positive class", f"{sa['positive_rate']*100:.1f}% (balanced)", f"{sb['positive_rate']*100:.1f}% (imbalanced)"],
            ["Data issues", f"{removed:,} implausible rows (BP, height, weight) removed; no missing values",
             f"{sb['n_missing_rows']:,} rows ({sb['n_missing_rows']/sb['n_raw']*100:.0f}%) with missing values; median-imputed in-pipeline"],
            ["Train / test", f"{sa['n_train']:,} / {sa['n_test']:,}", f"{sb['n_train']:,} / {sb['n_test']:,}"]]
    cellst = ParagraphStyle("cell", parent=body, fontSize=7.2, leading=8.6, spaceAfter=0)
    rows = [[Paragraph(str(c), cellst) for c in r] for r in rows]
    story += [tbl(rows, [2.2 * cm, 7.5 * cm, 7.5 * cm]), Spacer(1, 2),
              KeepTogether([fig(A["fig"] / "fig_data_quality.png"), fig(B["fig"] / "fig_data_quality.png"),
              P("<b>Fig. 1.</b> Data-quality reports. Top: Study A (rows flagged per rule, outliers before/after cleaning, class balance). "
                "Bottom: Study B (missing values per column, class balance).", cap)]),
              P("3. Approach and protocol (identical for both studies)", h),
              P("<b>Models:</b> logistic regression with Ridge, Lasso and Elastic Net penalties (standardised features); Random Forest; and "
                f"<b>{gbm}</b> as the main boosted model. <b>Protocol:</b> stratified 80/20 split (seed 42); median imputation and scaling fitted inside each "
                f"cross-validation fold (no leakage); {sa['cv_folds']}-fold grid search on AUROC using the training data only; one evaluation on the held-out test set. "
                "Study B uses class weighting for the 15% positive class. <b>Metrics:</b> AUROC (primary), average precision, accuracy, sensitivity, "
                "specificity at threshold 0.5, confusion matrices, calibration, and AUROC within gender and age subgroups."),
              P("4. Implementation", h),
              P("Python (pandas, scikit-learn, XGBoost, matplotlib). <font face='Courier'>src/benchmark/</font> holds the shared pipeline: "
                "<font face='Courier'>datasets.py</font> (loading, cleaning, per-dataset settings), <font face='Courier'>train.py</font>, "
                "<font face='Courier'>visualize.py</font>, <font face='Courier'>make_report.py</font>, <font face='Courier'>make_slides.py</font>. "
                "The Framingham exploration and first baselines (data inspection, quality report, untuned LR/RF, coefficients) are the scripts in "
                "<font face='Courier'>src/</font>; the benchmark reuses their exact split. Commands are in the README.")]

    def mrow(D):
        out = {}
        for r in D["m"].itertuples():
            out[r.model] = [f"{r.cv_auroc:.3f}", f"{r.test_auroc:.3f}", f"{r.avg_precision:.3f}", f"{r.sensitivity:.2f}", f"{r.specificity:.2f}"]
        return out
    ra, rb = mrow(A), mrow(B)
    trows = [["", "Study A (Kaggle, prevalent CVD)", "", "", "", "", "Study B (Framingham, 10-yr CHD)", "", "", "", ""],
             ["Model"] + ["CV AUC", "Test AUC", "Avg prec", "Sens", "Spec"] * 2]
    for mdl in A["m"].model:
        trows.append([mdl] + ra[mdl] + rb.get(mdl, [""] * 5))
    t = tbl(trows, [3.1 * cm] + [1.37 * cm] * 10, fs=6.8, header_rows=2,
            span=[("SPAN", (1, 0), (5, 0)), ("SPAN", (6, 0), (10, 0)), ("ALIGN", (1, 0), (-1, -1), "CENTER")])
    story += [P("5. Results", h), t, Spacer(1, 2),
              KeepTogether([fig(A["fig"] / "fig_models.png"), fig(B["fig"] / "fig_models.png"),
              P("<b>Fig. 2.</b> Model comparison (top: Study A, bottom: Study B): CV vs test AUROC, metrics at threshold 0.5, ROC and precision-recall curves.", cap)]),
              KeepTogether([fig(A["fig"] / "fig_confusion.png"), fig(B["fig"] / "fig_confusion.png"),
              P("<b>Fig. 3.</b> Test-set confusion matrices (threshold 0.5, row-normalised percentages). Class weighting in Study B trades specificity for sensitivity.", cap)]),
              KeepTogether([fig(A["fig"] / "fig_diagnostics.png"), fig(B["fig"] / "fig_diagnostics.png"),
              P("<b>Fig. 4.</b> Feature importance of the main tree model, subgroup AUROC and calibration of the best model (top: Study A, bottom: Study B).", cap)]),
              P("6. Discussion and conclusions", h)]

    ba, bb = best(A), best(B)
    lra, lrb = row(A, "LogReg"), row(B, "LogReg")
    fa, fb = row(A, "XGB"), row(B, "XGB")
    calb = B["cal"]
    init = REPO / "reports" / "model_comparison.csv"
    init_txt = ""
    if init.exists():
        ic = pd.read_csv(init)
        try:
            init_txt = (f" Our first untuned Study B baselines (class-weighted LR / RF) reached AUROC {ic.loc[ic.Model.str.startswith('Logistic'), 'ROC_AUC'].iloc[0]:.3f} / "
                        f"{ic.loc[ic.Model.str.startswith('Random'), 'ROC_AUC'].iloc[0]:.3f} on the same test split, consistent with the tuned results above.")
        except Exception:
            pass
    story += [P(f"<b>Study A:</b> best model {ba.model} (test AUROC {ba.test_auroc:.3f}); best logistic regression {lra.test_auroc:.3f}; {gbm} {fa.test_auroc:.3f}. "
                f"<b>Study B:</b> best model {bb.model} (test AUROC {bb.test_auroc:.3f}); best logistic regression {lrb.test_auroc:.3f}; {gbm} {fb.test_auroc:.3f}. "
                f"CV and test AUROC differ by {abs(ba.cv_auroc-ba.test_auroc):.3f} (A) and {abs(bb.cv_auroc-bb.test_auroc):.3f} (B) for the best models; with only "
                f"{sb['n_test']:,} test patients ({int(round(sb['positive_rate']*sb['n_test']))} events), Study B differences of a few hundredths are within noise. "
                + (f"Absolute AUROC is lower in Study B ({bb.test_auroc:.3f} vs {ba.test_auroc:.3f}), plausibly because it predicts future events from a small cohort "
                   "with fewer, noisier signals." if bb.test_auroc < ba.test_auroc else
                   f"Absolute AUROC is similar or higher in Study B ({bb.test_auroc:.3f} vs {ba.test_auroc:.3f}).") + init_txt),
              P(f"<b>Fairness and calibration.</b> Subgroup AUROC (A: {sub_txt(A)}; B: {sub_txt(B)}) shows that performance is not uniform and should be audited before clinical use. "
                f"In Study B, class weighting raises predicted probabilities (mean {calb['mean_predicted']:.2f} vs observed rate {calb['observed_rate']:.2f}), so rankings (AUROC) are valid "
                "but probabilities need recalibration before being read as risks."),
              P("<b>Limitations / future work:</b> Study A labels are prevalent disease with noisy self-reported fields and no ethnicity, so the paper's race-based fairness "
                "analysis could not be repeated; Study B is small, from one cohort, with a single random split. Next: repeated CV / confidence intervals, probability calibration, "
                "survival-style modelling of incident risk and fairness-aware training.")]
    doc.build(story)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
