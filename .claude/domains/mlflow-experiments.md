# MLflow Experiments Documentation

## 🚨🚨🚨 CRITICAL: REGISTRY IS SINGLE SOURCE OF TRUTH 🚨🚨🚨

**STOP! Before using ANY method name from this document:**

1. **CHECK** `configs/mlflow_registry/parameters/classification.yaml`
2. **USE** `src/data_io/registry.py` to get valid methods programmatically
3. **NEVER** parse MLflow run names to discover methods

**The registry defines EXACTLY:**
- **11 outlier methods** (NOT 17!)
- **8 imputation methods**
- **5 classifiers**

**If code produces different counts, IT IS BROKEN.**

See: `.claude/rules/05-registry-source-of-truth.md` for the full rule.

---

**Generated from actual MLflow data at `/home/petteri/mlruns`**
**Last updated**: 2026-01-21

**⚠️ WARNING**: The data below reflects what EXISTS in MLflow, which includes
orphan runs, test runs, and invalid configurations. The REGISTRY defines what
is VALID. Always use the registry, not raw MLflow scans.

## Experiment Structure

| Experiment | ID | Run Count | Purpose |
|------------|---|-----------|---------|
| PLR_OutlierDetection | 996740926475477194 | 31 | Outlier detection training |
| PLR_Imputation | 940304421003085572 | 138 | Imputation model training |
| PLR_Featurization | 143964216992376241 | 162 | Feature extraction |
| PLR_Classification | 253031330985650090 | 410 | Final classifier training |

## Classification Run Parameters

Each classification run has three key parameters:

1. **`anomaly_source`**: Outlier detection method used
2. **`imputation_source`**: Imputation method used
3. **`model_name`**: Classifier used

## Available Outlier Detection Methods (anomaly_source)

### ✅ VALID Methods (11 total - from Registry)

These are the ONLY valid outlier methods per `configs/mlflow_registry/parameters/classification.yaml`:

| Method | Type | Description |
|--------|------|-------------|
| `pupil-gt` | Ground Truth | Human expert annotation |
| `MOMENT-gt-finetune` | Foundation Model | MOMENT finetuned on gt labels |
| `MOMENT-gt-zeroshot` | Foundation Model | MOMENT zero-shot on gt labels |
| `UniTS-gt-finetune` | Foundation Model | UniTS finetuned |
| `TimesNet-gt` | Foundation Model | TimesNet ground-truth trained |
| `LOF` | Traditional | Local Outlier Factor |
| `OneClassSVM` | Traditional | One-Class SVM |
| `SubPCA` | Traditional | Subspace PCA |
| `PROPHET` | Traditional | Facebook Prophet |
| `ensemble-LOF-MOMENT-OneClassSVM-PROPHET-SubPCA-TimesNet-UniTS-gt-finetune` | Ensemble | All methods combined |
| `ensembleThresholded-MOMENT-TimesNet-UniTS-gt-finetune` | Ensemble | FM-only ensemble |

### ❌ INVALID Methods (DO NOT USE)

These exist in MLflow but are NOT valid for analysis:

| Method | Status | Reason |
|--------|--------|--------|
| `anomaly` | **GARBAGE** | Placeholder from test runs |
| `exclude` | **GARBAGE** | Placeholder from test runs |
| `MOMENT-orig-finetune` | **NOT IN REGISTRY** | Original (not ground-truth) variant |
| `UniTS-orig-finetune` | **NOT IN REGISTRY** | Original variant |
| `UniTS-orig-zeroshot` | **NOT IN REGISTRY** | Original variant |
| `TimesNet-orig` | **NOT IN REGISTRY** | Original variant |

**Python validation:**
```python
from src.data_io.registry import validate_outlier_method
if not validate_outlier_method("anomaly"):
    raise ValueError("Invalid outlier method!")  # This WILL raise
```

## Available Imputation Methods (imputation_source)

### ✅ VALID Methods (8 total - from Registry)

These are the ONLY valid imputation methods per `configs/mlflow_registry/parameters/classification.yaml`:

| Method | Type | Description |
|--------|------|-------------|
| `pupil-gt` | Ground Truth | Human annotation |
| `MOMENT-finetune` | Foundation Model | MOMENT finetuned |
| `MOMENT-zeroshot` | Foundation Model | MOMENT zero-shot |
| `CSDI` | Deep Learning | Conditional Score Diffusion |
| `SAITS` | Deep Learning | Self-Attention Imputation |
| `TimesNet` | Deep Learning | TimesNet imputation |
| `ensemble-CSDI-MOMENT-SAITS` | Ensemble | 3-method ensemble |
| `ensemble-CSDI-MOMENT-SAITS-TimesNet` | Ensemble | 4-method ensemble |

### ❌ INVALID Methods (DO NOT USE)

| Method | Status | Reason |
|--------|--------|--------|
| `linear` | **NOT IN REGISTRY** | Not used in final experiments |

## Available Classifiers (model_name)

| Classifier | Runs | Best AUROC |
|------------|------|------------|
| CatBoost | 81 | 0.9130 |
| TabPFN | 81 | 0.8897 |
| TabM | 81 | 0.8955 |
| XGBoost | 81 | 0.8693 |
| LogisticRegression | 81 | 0.7935 |

## Top Performing Combinations (by Test AUROC)

### Best Overall (Ensemble + CatBoost)

| Rank | AUROC | Outlier | Imputation | Classifier |
|------|-------|---------|------------|------------|
| 1 | 0.9130 | ensemble-LOF-MOMENT-...-UniTS-gt-finetune | CSDI | CatBoost |
| 2 | 0.9122 | ensemble-LOF-MOMENT-...-UniTS-gt-finetune | TimesNet | CatBoost |
| 3 | 0.9116 | ensembleThresholded-MOMENT-TimesNet-UniTS-gt-finetune | CSDI | CatBoost |
| 4 | 0.9113 | ensembleThresholded-MOMENT-TimesNet-UniTS-gt-finetune | TimesNet | CatBoost |

### Best Single FM Model (no ensembles)

| Rank | AUROC | Outlier | Imputation | Classifier |
|------|-------|---------|------------|------------|
| 1 | 0.9099 | MOMENT-gt-finetune | SAITS | CatBoost |
| 2 | 0.9086 | MOMENT-gt-finetune | TimesNet | CatBoost |
| 3 | 0.9068 | UniTS-orig-finetune | SAITS | CatBoost |
| 4 | 0.9066 | UniTS-orig-finetune | TimesNet | CatBoost |

### Ground Truth Performance

| AUROC | Outlier | Imputation | Classifier |
|-------|---------|------------|------------|
| 0.9110 | pupil-gt | pupil-gt | CatBoost |
| 0.8946 | pupil-gt | pupil-gt | TabM |
| 0.8892 | pupil-gt | pupil-gt | TabPFN |
| 0.8693 | pupil-gt | pupil-gt | XGBoost |
| 0.7829 | pupil-gt | pupil-gt | LogisticRegression |

### MOMENT+MOMENT (Single FM for both tasks)

| AUROC | Outlier | Imputation | Classifier |
|-------|---------|------------|------------|
| 0.8986 | MOMENT-gt-finetune | MOMENT-finetune | CatBoost |
| 0.8650 | MOMENT-gt-finetune | MOMENT-finetune | TabPFN |
| 0.8623 | MOMENT-gt-finetune | MOMENT-finetune | TabM |
| 0.8467 | MOMENT-gt-finetune | MOMENT-finetune | XGBoost |

### Traditional Methods Only

| AUROC | Outlier | Imputation | Classifier |
|-------|---------|------------|------------|
| 0.8830 | LOF | MOMENT-zeroshot | CatBoost |
| 0.8824 | OneClassSVM | MOMENT-zeroshot | CatBoost |
| 0.8808 | PROPHET | TimesNet | CatBoost |
| 0.8734 | PROPHET | SAITS | CatBoost |

### Low-Quality Outlier + MOMENT Imputation

| AUROC | Outlier | Imputation | Classifier |
|-------|---------|------------|------------|
| 0.8830 | LOF | MOMENT-zeroshot | CatBoost |
| 0.8553 | LOF | MOMENT-finetune | CatBoost |
| 0.8421 | OneClassSVM | MOMENT-finetune | CatBoost |

## Standard Combos for Manuscript (4 curves)

Based on actual MLflow results, these are the recommended standard combos:

| ID | Name | Outlier | Imputation | Classifier | AUROC | Rationale |
|----|------|---------|------------|------------|-------|-----------|
| ground_truth | Ground Truth | pupil-gt | pupil-gt | CatBoost | 0.9110 | Human expert upper bound |
| best_fm | MOMENT + SAITS | MOMENT-gt-finetune | SAITS | CatBoost | 0.9099 | Best single FM pipeline |
| traditional | LOF + SAITS | LOF | SAITS | TabPFN | 0.8599 | Traditional outlier + DL imputation |
| simple_baseline | OneClassSVM + Linear | OneClassSVM | MOMENT-zeroshot | CatBoost | 0.8824 | Traditional methods only |

## Extended Combos (5 additional for supplementary)

| ID | Name | Outlier | Imputation | Classifier | AUROC | Rationale |
|----|------|---------|------------|------------|-------|-----------|
| best_ensemble | Ensemble + CSDI | ensemble-LOF-MOMENT-...-gt-finetune | CSDI | CatBoost | 0.9130 | Best overall performance |
| moment_full | MOMENT + MOMENT | MOMENT-gt-finetune | MOMENT-finetune | CatBoost | 0.8986 | Single FM for both tasks |
| lof_moment | LOF + MOMENT | LOF | MOMENT-zeroshot | CatBoost | 0.8830 | Traditional outlier + FM imputation |
| timesnet_full | TimesNet + TimesNet | TimesNet-orig | TimesNet | CatBoost | 0.8970 | Alternative single FM |
| units_pipeline | UniTS + SAITS | UniTS-orig-finetune | SAITS | CatBoost | 0.9068 | Alternative FM comparison |

## Key Findings

1. **CatBoost dominates**: Best classifier across all outlier/imputation combos
2. **SAITS is top imputation**: Paired with FM outlier detection gives best results
3. **Ensembles marginally better**: +0.003 AUROC over best single model
4. **Ground truth ≈ Best FM**: MOMENT+SAITS nearly matches human annotation
5. **MOMENT+MOMENT viable**: 0.8986 AUROC with single model for both tasks

## MLflow Run ID Lookup

To find a specific run:
```bash
# Find runs with specific combo
cd /home/petteri/mlruns/253031330985650090
for run in */; do
  anomaly=$(cat "$run/params/anomaly_source" 2>/dev/null)
  imp=$(cat "$run/params/imputation_source" 2>/dev/null)
  model=$(cat "$run/params/model_name" 2>/dev/null)
  if [[ "$anomaly" == "MOMENT-gt-finetune" && "$imp" == "SAITS" && "$model" == "CatBoost" ]]; then
    echo "Run ID: ${run%/}"
  fi
done
```

## CRITICAL: Never Hallucinate Combos

When working on this project:

1. **USE THE REGISTRY** - `src/data_io/registry.py` is the SINGLE SOURCE OF TRUTH
2. **NEVER** parse MLflow run names to discover methods
3. **VERIFY** method names against `configs/mlflow_registry/parameters/classification.yaml`
4. **REJECT** any method not in the registry (even if it exists in MLflow)

**Python pattern:**
```python
from src.data_io.registry import get_valid_outlier_methods, validate_outlier_method

# Get all valid methods (returns exactly 11)
valid_methods = get_valid_outlier_methods()

# Validate before use
if not validate_outlier_method(some_method):
    raise ValueError(f"Invalid outlier method: {some_method}")
```

**NEVER:**
```python
# BANNED - leads to garbage like "anomaly"
methods = set(run_name.split("__")[3] for run in mlflow_runs)
```
