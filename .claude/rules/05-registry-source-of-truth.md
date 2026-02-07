# RULE: Registry is the SINGLE SOURCE OF TRUTH

**This rule applies to EVERY task involving experiment parameters.**

## The Ground Truth Location

```
configs/mlflow_registry/parameters/classification.yaml
```

## Exact Counts (MEMORIZE THESE)

| Parameter | Count | If Different = BROKEN |
|-----------|-------|----------------------|
| Outlier methods | **11** | Never 17, never 15, EXACTLY 11 |
| Imputation methods | **8** | |
| Classifiers | **5** | |

## Valid Outlier Methods (THE ONLY 11)

1. `pupil-gt` (ground truth)
2. `MOMENT-gt-finetune`
3. `MOMENT-gt-zeroshot`
4. `UniTS-gt-finetune`
5. `TimesNet-gt`
6. `LOF`
7. `OneClassSVM`
8. `PROPHET`
9. `SubPCA`
10. `ensemble-LOF-MOMENT-OneClassSVM-PROPHET-SubPCA-TimesNet-UniTS-gt-finetune`
11. `ensembleThresholded-MOMENT-TimesNet-UniTS-gt-finetune`

## INVALID Methods (NEVER USE)

- `anomaly` ← GARBAGE placeholder
- `exclude` ← GARBAGE placeholder
- `MOMENT-orig-finetune` ← Not in registry
- `UniTS-orig-finetune` ← Not in registry
- `UniTS-orig-zeroshot` ← Not in registry
- `TimesNet-orig` ← Not in registry

## Python Code Pattern

```python
# ALWAYS do this:
from src.data_io.registry import get_valid_outlier_methods, validate_outlier_method

valid_methods = get_valid_outlier_methods()  # Returns exactly 11

if not validate_outlier_method(some_method):
    raise ValueError(f"Invalid outlier method: {some_method}")

# NEVER do this:
methods = run_name.split("__")[3]  # BANNED - leads to garbage
methods = set(m for m in mlflow_data)  # BANNED - includes orphan runs
```

## Why This Rule Exists

MLflow contains orphan runs, test runs, and experiments with different naming conventions.
Parsing run names produces garbage like "anomaly" (17 methods instead of 11).
The registry is the VERIFIED list of methods used in the final paper.

## See Also

- `configs/mlflow_registry/README.md`
- `src/data_io/registry.py`
- `CLAUDE.md` (root) - Registry section
