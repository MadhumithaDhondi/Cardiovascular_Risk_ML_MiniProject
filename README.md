# Predicting Cardiovascular Risk using Electronic Health Records

UE24CS352A Machine Learning - Mini-Project | Team `25` | `Madhumitha Dhondi (PES1UG24AM923), Anjali Arun (PES1UG24AM421)`

We predict cardiovascular risk from routine clinical data and compare logistic regression (Ridge / Lasso / Elastic Net),
Random Forest and XGBoost. Following Nguyen's Stanford CS229 work on EHR-based CVD risk prediction (the original STRIDE EHR
data is private), the project has **two studies that share one pipeline**:

| | Study A - Kaggle Cardiovascular Disease | Study B - Framingham Heart Study |
|---|---|---|
| Question | Is CVD present now? (prevalent) | Will CHD occur within 10 years? (incident) |
| Size / labels | ~70,000 rows, ~50% positive | ~4,200 rows, ~15% positive |
| Data issues | implausible BP / height / weight values | ~14% of rows have missing values |
| Handling | outliers removed | median imputation inside the CV pipeline, class weighting |

Both studies use the same five models, the same stratified 80/20 split (seed 42), cross-validated grid search on AUROC
(training data only) and the same metrics, so results can be compared across studies.

## Repository layout

```
data/raw/                       framingham_heart_disease.csv (included); cardio_train.csv (download, see data/README.md)
src/                            Study B exploration and first baselines (inspect_data, data_quality, prepare_data,
                                train_baseline, train_random_forest, compare_models, feature_importance)
src/benchmark/                  unified pipeline for BOTH studies
    datasets.py                 loading, cleaning and settings per dataset
    train.py                    tuning + evaluation -> results/<dataset>/
    visualize.py                all figures -> results/<dataset>/figures/
    make_report.py              combined 2-page write-up -> reports/writeup.pdf
    make_slides.py              combined deck -> slides/deck.pptx
results/cardio/  results/framingham/   metrics, predictions, subgroup tables, figures
reports/                        write-up PDF, viva notes, and the outputs of the Study B baseline scripts
slides/                         deck.pptx
models/                         pipelines saved by the Study B baseline scripts
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
# macOS only, if xgboost complains about libomp:  brew install libomp
```

Download the Kaggle file as described in [`data/README.md`](data/README.md). Run everything from the repository root.

## Run

```bash
# Study A (Kaggle) and Study B (Framingham): same commands, different --dataset
python src/benchmark/train.py     --dataset cardio
python src/benchmark/train.py     --dataset framingham
python src/benchmark/visualize.py --dataset cardio
python src/benchmark/visualize.py --dataset framingham
```

Add `--fast` to `train.py` for small grids (quick demo). `--synthetic` generates fake data to check the setup only; never report those numbers.

The original Study B scripts still work independently, e.g. `python src/train_baseline.py`, `python src/compare_models.py`.
Their first, untuned baselines (reports/model_comparison.csv) use the same train/test split as the benchmark, so the numbers are directly comparable.

## Method (both studies)

- Split: stratified 80/20, `random_state=42` (identical to the Study B baseline scripts).
- Preprocessing: median imputation (+ standardisation for logistic regression) inside each cross-validation fold - no leakage.
- Tuning: 10-fold grid search on AUROC, training data only. Class weighting for the imbalanced Framingham labels.
- Models: logistic regression (Ridge, Lasso, Elastic Net), Random Forest, XGBoost.
- Metrics: AUROC, average precision, accuracy, sensitivity, specificity (threshold 0.5), confusion matrices, calibration,
  AUROC by gender and age subgroup.

## Outputs per study (`results/<dataset>/`)

`metrics.csv`, `best_params.json`, `subgroups.csv`, `calibration.json`, `data_summary.json`, `data_quality.json`,
`feature_importance.csv`, `test_predictions.csv`, and `figures/` (data quality, EDA, model comparison + ROC + precision-recall,
confusion matrices, feature importance + subgroup AUROC + calibration).

## Contributions

- Study B data inspection, quality report, preprocessing and first logistic regression / Random Forest baselines
  (`src/*.py`): Madhumitha Dhondi (PES1UG24AM923) 
- Study A (Kaggle), the unified benchmark pipeline (`src/benchmark/`), figures, write-up and slides: Anjali Arun (PES1UG24AM421)
