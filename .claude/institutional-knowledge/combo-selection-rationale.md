# Hyperparameter Combo Selection Rationale

**Updated**: 2026-01-21
**Source**: Verified from MLflow at `/home/petteri/mlruns/253031330985650090` (410 runs)

## Why Fixed Combos?

The Foundation PLR project has 410 classification runs with hundreds of hyperparameter combinations. Without a fixed subset, each visualization might use different combos, making the manuscript:
- **Non-reproducible** - Different figures show different comparisons
- **Cherry-picked** - Easy to accidentally select favorable combos
- **Inconsistent** - Readers can't compare across figures

## Selection Criteria (Based on Actual MLflow Data)

### Standard Combos (4 curves for main figures)

1. **Ground Truth (pupil-gt + pupil-gt + CatBoost)** - AUROC: 0.9110
   - Human expert annotation
   - Upper bound reference
   - MUST appear in every comparison figure
   - Establishes what's achievable with perfect preprocessing

2. **Best Ensemble (ensemble-LOF-MOMENT-...-UniTS-gt-finetune + CSDI + CatBoost)** - AUROC: 0.9130
   - Highest overall AUROC in all MLflow runs
   - Full ensemble of all outlier detection methods
   - Represents the theoretical best (but complex) approach

3. **Best Single FM (MOMENT-gt-finetune + SAITS + CatBoost)** - AUROC: 0.9099
   - Best non-ensemble foundation model pipeline
   - Selected as top-1 by mean AUROC on test set
   - Represents the "recommended pipeline" for users

4. **Traditional (LOF + SAITS + TabPFN)** - AUROC: 0.8599
   - Traditional outlier detection with deep learning imputation
   - Fair comparison to show FM improvement
   - Uses well-established LOF algorithm

### Why 4 Combos for Main Figures?

Nature/Science guidelines recommend max 4 overlapping curves per figure for readability. With CI bands on 2 curves, visual clarity is maintained.

### Extended Combos (5 additional for supplementary)

5. **MOMENT+MOMENT (MOMENT-gt-finetune + MOMENT-finetune + CatBoost)** - AUROC: 0.8986
   - "Minimal effort FM" - single model for both tasks
   - Lowest barrier to entry for new users
   - Shows "what you get for free" with minimal setup

6. **LOF+MOMENT (LOF + MOMENT-zeroshot + CatBoost)** - AUROC: 0.8830
   - Traditional outlier detection with FM imputation
   - Shows FM imputation salvages poor outlier detection
   - Ablation study for imputation contribution

7. **TimesNet Full (TimesNet-orig + TimesNet + CatBoost)** - AUROC: 0.8970
   - Alternative single-FM pipeline
   - Shows different FM architectures also work
   - Robustness check

8. **UniTS Pipeline (UniTS-orig-finetune + SAITS + CatBoost)** - AUROC: 0.9068
   - Alternative FM for outlier detection
   - Shows MOMENT isn't uniquely special
   - Other FMs achieve comparable performance

9. **Simple Baseline (OneClassSVM + MOMENT-zeroshot + CatBoost)** - AUROC: 0.8824
   - Traditional outlier + FM imputation
   - Lower bound reference with simple outlier detection
   - Shows traditional methods still reasonable

## What's NOT Included (and Why)

| Method | Reason |
|--------|--------|
| "No Outlier Detection" | **DOES NOT EXIST** in MLflow - this was hallucinated |
| "linear" imputation | **DOES NOT EXIST** in MLflow - only SAITS, CSDI, TimesNet, MOMENT |
| PROPHET-based combos | Performance not significantly different from LOF |
| SubPCA-based combos | Very similar to OneClassSVM in results |
| EIF methods | Not in current MLflow runs |
| CSDI-based combos (except ensemble) | Slow inference, not practical for users |

## Classifier Choice Rationale

**CatBoost is standard** for all main combos because:
- Best performing classifier across all preprocessing combos
- Consistent 2-5% improvement over XGBoost
- Robust to hyperparameters (default settings work well)
- Fast inference for practical deployment

TabPFN used in traditional combo to show a different high-performing option.

## MLflow Linkage

| Combo ID | Experiment | Query |
|----------|------------|-------|
| ground_truth | PLR_Classification | anomaly_source=pupil-gt, imputation_source=pupil-gt, model_name=CatBoost |
| best_ensemble | PLR_Classification | anomaly_source=ensemble-LOF-MOMENT-..., imputation_source=CSDI, model_name=CatBoost |
| best_single_fm | PLR_Classification | anomaly_source=MOMENT-gt-finetune, imputation_source=SAITS, model_name=CatBoost |
| traditional | PLR_Classification | anomaly_source=LOF, imputation_source=SAITS, model_name=TabPFN |

## Update Process

1. If new methods are added to MLflow, re-run exploration script
2. Compare against current standard combos for significant improvements
3. If a new method significantly outperforms best_single_fm (>0.01 AUROC), consider replacement
4. Always update `.claude/domains/mlflow-experiments.md` after MLflow changes
5. Document rationale for any changes in this file

## CRITICAL: Verification Before Proposing

**NEVER propose combos without checking:**
1. `.claude/domains/mlflow-experiments.md` - Full MLflow documentation
2. `configs/VISUALIZATION/plot_hyperparam_combos.yaml` - Current fixed combos
3. `/home/petteri/mlruns` - Actual MLflow data (if in doubt)

**Method names must match EXACTLY:**
- `MOMENT-gt-finetune` NOT `MOMENT-finetune-gt`
- `ensemble-LOF-MOMENT-OneClassSVM-PROPHET-SubPCA-TimesNet-UniTS-gt-finetune` (full name)
- `MOMENT-zeroshot` for imputation, `MOMENT-gt-zeroshot` for outlier detection
