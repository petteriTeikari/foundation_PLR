# Visualization Domain Context

## 🚨 CRITICAL: Registry is Single Source of Truth

**All method names in visualizations MUST come from the registry.**

- **Registry location**: `configs/mlflow_registry/parameters/classification.yaml`
- **Python API**: `src/data_io/registry.py`
- **Rule file**: `.claude/rules/05-registry-source-of-truth.md`

**Exact counts (if figures show different numbers, DATA IS BROKEN):**
- 11 outlier methods
- 8 imputation methods
- 5 classifiers

---

## Hyperparameter Combo Reference

### MANDATORY: Load from configs/VISUALIZATION/plot_hyperparam_combos.yaml

```python
import yaml
from pathlib import Path

def load_standard_combos():
    """Load fixed hyperparam combos. NEVER hardcode these values."""
    config_path = Path("configs/VISUALIZATION/plot_hyperparam_combos.yaml")
    with open(config_path, encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config["standard_combos"]
```

### Standard Combos (4 curves for main figures)

| ID | Name | Outlier | Imputation | Line Style | Color |
|----|------|---------|------------|------------|-------|
| ground_truth | Ground Truth | pupil-gt | pupil-gt | solid thick | gray |
| best_fm | MOMENT + SAITS | MOMENT-gt | SAITS | solid | blue |
| traditional | LOF + Linear | LOF | linear | dashed | orange |
| simple_baseline | OC-SVM + Linear | OneClassSVM | linear | dotted | pink |

### Visual Hierarchy

1. **Ground Truth**: Solid thick gray (2pt) - Reference standard, anchoring element
2. **FM Methods**: Solid colored (1.5pt) - Main results
3. **Traditional**: Dashed colored (1.5pt) - Comparison category
4. **Baseline**: Dotted thin (1pt) - Lower bound

### Color Variables (CSS)

```css
--color-ground-truth: #666666;  /* Gray */
--color-fm-primary: #0072B2;    /* Blue - Paul Tol */
--color-fm-secondary: #56B4E9;  /* Light blue */
--color-traditional: #E69F00;   /* Orange - Paul Tol */
--color-baseline: #CC79A7;      /* Pink - Paul Tol */
```

### CI Band Rules

- Show CI bands for maximum 2 combos
- Priority: ground_truth and best_fm
- Use alpha=0.25 for band fill
- Alternative: Error bars at discrete x-values (0.3, 0.5, 0.7)

### Figure Type Specifics

#### Retention Curves
- X: Retention rate (0-100%)
- Y: Metric value (AUROC, Brier, etc.)
- Label from data.metric field, NOT hardcoded

#### Calibration Plots
- Include 45° perfect calibration line (gray dashed)
- Histogram of predicted probabilities at bottom

#### DCA Curves
- Include "treat all" reference line
- Include "treat none" reference line (y=0)
- X: Threshold probability
- Y: Net benefit

### Subject Selection for Individual Traces

Use `configs/demo_subjects.yaml`:
- 6 control subjects (PLR1xxx)
- 6 glaucoma subjects (PLR4xxx)
- Stratified by outlier percentage

### FORBIDDEN

- Hardcoding method names (MOMENT, SAITS, LOF, etc.)
- Hardcoding color hex values in components
- Using more than 4 curves in main figures
- Omitting ground_truth from comparison figures
- Using "AUROC" as hardcoded label (use data.metric)
