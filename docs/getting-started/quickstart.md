# Quick Start

Run your first Foundation PLR experiment in 5 minutes.

## Prerequisites

Ensure you have completed the [installation](installation.md) steps.

## Step 1: Activate Environment

```bash
cd foundation-PLR/foundation_PLR
source .venv/bin/activate
```

## Step 2: Run a Classification Experiment

```bash
# Run with default configuration
python -m src.classification.flow_classification
```

This will:

1. Load PLR data from the database
2. Apply default preprocessing (outlier detection + imputation)
3. Extract handcrafted features
4. Train a CatBoost classifier
5. Evaluate with bootstrap validation
6. Log results to MLflow

## Step 3: View Results

```bash
# Start MLflow UI
mlflow ui --port 5000
```

Open [http://localhost:5000](http://localhost:5000) to view experiment results.

## Configuration

Override defaults with Hydra:

```bash
# Change classifier
python -m src.classification.flow_classification classifier=XGBoost

# Change preprocessing
python -m src.classification.flow_classification \
    outlier_method=MOMENT-gt-finetune \
    imputation_method=SAITS
```

## Available Methods

The pipeline supports:

| Stage | Methods | Registry |
|-------|---------|----------|
| Outlier Detection | 11 methods | `configs/mlflow_registry/parameters/classification.yaml` |
| Imputation | 8 methods | `configs/mlflow_registry/parameters/classification.yaml` |
| Classification | 5 classifiers | CatBoost recommended |

**Note**: The registry is the single source of truth for method names.

## Next Steps

- Learn about the [pipeline architecture](../user-guide/pipeline-overview.md)
- Understand [configuration options](configuration.md)
- Explore [API reference](../api-reference/index.md)
- **New to software tools?** See [Concepts for Researchers](../concepts-for-researchers.md)
