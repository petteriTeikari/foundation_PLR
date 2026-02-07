# Running Experiments

This tutorial walks through running a complete preprocessing sensitivity experiment.

## Overview

We'll:

1. Configure preprocessing methods
2. Run the classification pipeline
3. Analyze results with MLflow
4. Compare STRATOS metrics

## Step 1: Configure the Experiment

Create a custom configuration:

```yaml
# configs/my_experiment.yaml
defaults:
  - defaults

# Preprocessing configuration
outlier_method: MOMENT-gt-finetune
imputation_method: SAITS

# Classifier (keep fixed for preprocessing analysis)
classifier: CatBoost

# Bootstrap settings
CLS_EVALUATION:
  BOOTSTRAP:
    n_iterations: 1000
```

## Step 2: Run the Pipeline

```bash
# Activate environment
source .venv/bin/activate

# Run with custom config
python -m src.classification.flow_classification \
    --config-name=my_experiment
```

## Step 3: Monitor Progress

```bash
# In another terminal, start MLflow UI
mlflow ui --port 5000
```

Watch the experiment at [http://localhost:5000](http://localhost:5000).

## Step 4: Compare Results

### View Metrics

In MLflow UI:

1. Select experiments to compare
2. Click "Compare"
3. View STRATOS metrics side-by-side

### Export Results

```bash
# Export to DuckDB for analysis
python -m src.data_io.duckdb_export export \
    --mlruns /home/petteri/mlruns \
    --output results.db
```

## Step 5: Analyze STRATOS Metrics

The pipeline computes all STRATOS-compliant metrics:

| Metric | Description | Target |
|--------|-------------|--------|
| AUROC | Discrimination | Higher better |
| Brier | Overall performance | Lower better |
| Calibration slope | Should be ~1.0 | Close to 1.0 |
| Net Benefit | Clinical utility | Higher better |

## Next Steps

- Try different [preprocessing combinations](../user-guide/pipeline-overview.md)
- Learn how to [add new methods](adding-new-methods.md)
