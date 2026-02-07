# MLflow Run Naming Convention

**Purpose**: Document the naming conventions used in MLflow runs

---

## Run Name Format

### Classification Runs (4 fields)
```
{CLASSIFIER}_eval-{metric}__{featurization}__{imputation}__{outlier}
```

Example: `CATBOOST_eval-auc__simple1.0__SAITS__pupil-gt`

### Legacy Format (2 fields)
```
{CLASSIFIER}__{imputation}
```
**WARNING**: When only 2 fields exist, the code defaults `outlier = "anomaly"`.

---

## Outlier Detection Field Values

| Value | Description | Ground Truth? |
|-------|-------------|---------------|
| `pupil-gt` | Human-annotated masks | YES |
| `LOF`, `OneClassSVM`, etc. | Traditional methods | NO |
| `MOMENT-gt-finetune`, etc. | Foundation model methods | NO |
| `ensemble-*` | Ensemble of methods | NO |
| **`anomaly`** | **LEGACY DEFAULT - SOURCE UNCLEAR** | **UNKNOWN** |
| **`exclude`** | **Similar - SOURCE UNCLEAR** | **UNKNOWN** |

---

## Key Warnings

### "anomaly" and "exclude" Outlier Sources

When `mlflow_run_outlier_detection: None` and `Outlier_f1: nan`:
- No explicit outlier detection was run
- The actual data source is unclear
- **Recommendation**: Exclude from analysis unless verified

### Parsing Logic

```python
# From ensemble_utils.py:1016-1018
if len(fields) == 2:
    cls, imput = run_name.split("__")
    outlier = "anomaly"  # DEFAULT when missing!
```

---

## Ensemble Naming

### Imputation Ensemble
```
ensemble-{model1}-{model2}-{...}__{anomaly_source}
```

### Outlier Detection Ensemble
```
ensemble-{model1}-{model2}-{...}
```

---

## Related Files

- `src/ensemble/ensemble_utils.py` - Parsing logic
- `src/classification/classifier_log_utils.py` - Logging
- `src/ensemble/ensemble_logging.py` - Name generation

---

*See also*: `/manuscripts/foundationPLR/background-research/mlflow-naming-convention.md` for full documentation.
