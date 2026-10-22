# Viva notes: two studies, one project

Fill every `[...]` from `results/cardio/metrics.csv` and `results/framingham/metrics.csv` after you run the pipeline.
Numbers marked (earlier run) are Study A results from the first run; re-check them against the new run.

## Facts to know cold

| | Study A: Kaggle | Study B: Framingham |
|---|---|---|
| Question | CVD present now | CHD within 10 years |
| Rows | 70,000 -> 67,941 after cleaning | 4,238 (none dropped) |
| Features | 13 (adds BMI, pulse pressure) | 15 |
| Positive rate | 49.7% | 15.2% |
| Test set | 13,589 | 848 (about 129 events) |
| Data issue | implausible BP/height/weight rows removed | 582 rows (14%) with missing values; glucose 388 (9.2%) |
| Best model, test AUROC | [fill] (earlier run: XGBoost 0.796, best LR 0.784) | [fill] |
| Random Forest | [fill] | [fill] |
| Subgroups | (earlier run) gender gap 0.006; age <50 0.822, 50-59 0.764, 60+ 0.729 | [fill: check the weakest age group] |

Study B first baselines (untuned, Madhumitha's scripts): LR AUROC 0.699, RF 0.668 on the same split.

## Who presents what

- Shared: problem and plan, approach, implementation, conclusions (split them).
- Study A slides (data quality, EDA, model comparison, confusion, diagnostics): the Kaggle owner.
- Study B slides: the Framingham owner.
- Each of you must still be able to answer questions on the whole pipeline.

## Likely questions

**Why two datasets?** They answer different halves of the question. Kaggle is large and balanced, so model comparison is stable (prevalent disease).
Framingham predicts future (10-year) events from a small, imbalanced dataset with missing values - the setting of the paper. Checking both
tells us whether the conclusions are specific to one dataset.

**Why a unified pipeline instead of separate scripts?** So results are comparable: same split, folds, models and metrics. The earlier Framingham
scripts were first baselines (untuned, 5-fold); the benchmark reuses their exact split and adds tuning and the other models.

**How did you handle missing values?** Median imputation fitted inside each cross-validation fold (and on the training set only for the final
model), so no information from validation/test rows leaks into training. Only Framingham has missing values.

**Why class weighting?** Framingham has 15% positives, so an unweighted model can score well by predicting "no event". Weighting makes errors on
the rare class cost more. It lowers specificity and accuracy and raises sensitivity.

**What does class weighting do to probabilities?** It inflates them (mean predicted probability is far above the observed 15% rate, visible in the
calibration plot). AUROC (ranking) is unaffected, but the outputs are not true risks until recalibrated.

**Why AUROC?** Threshold independent, standard in the literature and in the paper. We also report average precision (more informative when positives
are rare), sensitivity/specificity at 0.5, and confusion matrices.

**Why is accuracy only about 0.7 if AUROC is about 0.8 (Study A)?** Accuracy uses one threshold (0.5); AUROC measures ranking over all thresholds.
For screening you would lower the threshold to raise sensitivity.

**Framingham: LR vs Random Forest vs XGBoost - does the ranking matter?** With 848 test patients (129 events), AUROC differences of a few hundredths are
within noise. Say that rather than over-claiming a winner. Repeated CV or bootstrap confidence intervals would settle it (future work).

**Why is AUROC lower on Framingham?** It predicts future events (harder), from a small cohort, with fewer and noisier signals. This is a hypothesis,
not something we tested.

**Why regularise logistic regression with so few features?** A cheap way to reduce variance. In practice Ridge, Lasso and Elastic Net perform almost
identically here, which is expected with 13-15 features.

**Why XGBoost vs Random Forest?** Random Forest averages many deep decorrelated trees (bagging, lower variance). XGBoost builds shallow trees
sequentially, each fitting the previous errors, with L1/L2 regularisation (boosting, lower bias). Comparing them shows whether gains come from
trees in general or from boosting.

**Did you over-fit?** Compare CV and test AUROC in your tables. We kept the test set untouched during tuning. Study B's gap is larger because the
test set is small and noisy.

**What does the fairness analysis show?** Gender gaps are small. Age is not uniform - report your measured numbers, and say we did not test why.
No ethnicity column exists in either dataset, so the paper's race-based analysis could not be repeated.

**Grid-edge caveat:** if tuned XGBoost values sit at the edge of the grid (smallest depth / learning rate, largest number of trees), a wider grid could
help slightly; expect small gains.

**How does this differ from the paper?** Paper: private Stanford EHR, 256k patients, 39k sparse binary features, 1.7% positive incident CVD,
AUROC 0.782. Ours: public data, 4k and 68k patients, 13-15 tabular features. Our AUROCs are not directly comparable to theirs.

**Limitations?** Study A labels are prevalent disease, self-reported fields, noisy labels; Study B is one small cohort; one random split each;
no external validation.

**What would you do next?** Repeated CV / confidence intervals, probability calibration, survival-style modelling of incident risk, fairness-aware training,
validation on a third dataset.

**What is your own contribution?** Prepare one or two sentences naming specific files and matching your commit history
(Study A + `src/benchmark/` pipeline, or the Study B scripts in `src/`).
