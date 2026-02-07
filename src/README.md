# Source Code (`src/`)

This directory contains the core implementation of the Foundation PLR pipeline.

## Pipeline Stages

| Stage | Module | Entry Point | Description |
|-------|--------|-------------|-------------|
| **0. Data Import** | `data_io/` | `flow_data.py` | Load PLR data from DuckDB |
| **1. Outlier Detection** | `anomaly_detection/` | `flow_anomaly_detection.py` | Detect artifacts in PLR signals |
| **2. Imputation** | `imputation/` | `flow_imputation.py` | Reconstruct missing segments |
| **3. Featurization** | `featurization/` | `flow_featurization.py` | Extract handcrafted features |
| **4. Classification** | `classification/` | `flow_classification.py` | Train CatBoost, evaluate STRATOS metrics |

## Supporting Modules

| Module | Description |
|--------|-------------|
| `stats/` | STRATOS-compliant metrics (calibration, clinical utility) |
| `viz/` | Figure generation for manuscript |
| `log_helpers/` | MLflow experiment tracking integration |
| `metrics/` | Imputation evaluation metrics utilities |
| `orchestration/` | Prefect workflow utilities and reproducibility flows |
| `extraction/` | Production guardrails for MLflow extraction (memory, stall detection) |
| `ensemble/` | Ensemble methods for anomaly detection, imputation, and classification |
| `preprocess/` | Signal preprocessing and normalization |
| `decomposition/` | PLR waveform decomposition (PCA, GED, template fitting) |
| `config/` | Typed configuration loading and validation (Pydantic) |
| `synthetic/` | Synthetic PLR data generation for testing and demos |
| `summarization/` | Experiment summarization and cross-run analysis |
| `deploy/` | Model deployment (placeholder) |
| `r/` | R scripts for statistical analyses (pminternal, DCA) |
| `tools/` | Development and debugging utilities |
| `utils/` | Shared utility functions |
| `figures/` | Output directory for generated figures |

## Module Index

```
src/
├── pipeline_PLR.py              # Main entry point (Hydra + Prefect)
├── utils.py                     # Shared utilities
│
├── anomaly_detection/           # Stage 1: 11 outlier methods
│   ├── flow_anomaly_detection.py
│   ├── anomaly_detection.py     # Core detection logic
│   ├── outlier_sklearn.py       # LOF, SVM methods
│   ├── outlier_prophet.py       # Prophet-based detection
│   ├── outlier_tsb_ad.py        # TSB-AD benchmark
│   ├── timesnet_wrapper.py      # TimesNet integration
│   └── momentfm_outlier/        # MOMENT foundation model
│
├── imputation/                  # Stage 2: 7 imputation methods
│   ├── flow_imputation.py
│   ├── imputation_main.py       # Core imputation logic
│   ├── impute_with_models.py    # Model dispatch
│   ├── momentfm/                # MOMENT imputation
│   ├── pypots/                  # [VENDORED] PyPOTS (SAITS, CSDI)
│   └── nuwats/                  # [VENDORED] NuwaTS
│
├── featurization/               # Stage 3: Feature extraction
│   ├── flow_featurization.py
│   ├── featurize_PLR.py         # Amplitude bins + latency
│   ├── featurizer_PLR_subject.py
│   └── embedding/               # FM embedding alternatives
│
├── classification/              # Stage 4: Classification + evaluation
│   ├── flow_classification.py
│   ├── train_classifier.py      # CatBoost training
│   ├── bootstrap_evaluation.py  # Bootstrap CI
│   ├── classifier_evaluation.py # Evaluation pipeline
│   ├── tabpfn/                  # [VENDORED] TabPFN v2
│   └── tabpfn_v1/               # [VENDORED] TabPFN v1
│
├── stats/                       # STRATOS-compliant metrics
│   ├── calibration_extended.py  # Calibration slope/intercept
│   ├── clinical_utility.py      # Net Benefit, DCA
│   ├── scaled_brier.py          # IPA (Scaled Brier)
│   ├── pminternal_wrapper.py    # R pminternal for stability
│   └── ...
│
├── viz/                         # Visualization
│   ├── generate_all_figures.py  # Main figure generator
│   ├── plot_config.py           # Style configuration
│   ├── metric_registry.py       # Metric definitions
│   └── ...                      # Individual figure modules
│
├── data_io/                     # Data import/export
│   ├── flow_data.py             # Data flow orchestration
│   ├── duckdb_export.py         # DuckDB export utilities
│   └── ...
│
└── log_helpers/                 # MLflow integration
    ├── mlflow_utils.py          # MLflow setup
    ├── mlflow_artifacts.py      # Artifact logging
    └── ...
```

## Entry Points

### Main Pipeline

```bash
python src/pipeline_PLR.py
```

### Figure Generation

```bash
python src/viz/generate_all_figures.py
python src/viz/generate_all_figures.py --figure R7  # Specific figure
```

## Module Documentation

Each subdirectory has its own README:

**Pipeline stages:**
- [anomaly_detection/README.md](anomaly_detection/README.md)
- [imputation/README.md](imputation/README.md)
- [featurization/README.md](featurization/README.md)
- [classification/README.md](classification/README.md)

**Analysis and visualization:**
- [stats/README.md](stats/README.md)
- [viz/README.md](viz/README.md)
- [figures/README.md](figures/README.md)
- [decomposition/README.md](decomposition/README.md)
- [summarization/README.md](summarization/README.md)

**Infrastructure:**
- [data_io/README.md](data_io/README.md)
- [log_helpers/README.md](log_helpers/README.md)
- [orchestration/README.md](orchestration/README.md)
- [extraction/README.md](extraction/README.md)
- [ensemble/README.md](ensemble/README.md)
- [preprocess/README.md](preprocess/README.md)
- [metrics/README.md](metrics/README.md)
- [config/README.md](config/README.md)
- [synthetic/README.md](synthetic/README.md)
- [deploy/README.md](deploy/README.md)
- [r/README.md](r/README.md)
- [tools/README.md](tools/README.md)
- [utils/README.md](utils/README.md)

## Vendored Code

The following directories contain third-party code and are **excluded from documentation**:

| Directory | Source | Purpose |
|-----------|--------|---------|
| `anomaly_detection/extra_eval/TSB_AD/` | TSB-AD benchmark | Outlier evaluation |
| `imputation/pypots/` | PyPOTS | SAITS, CSDI imputation |
| `imputation/nuwats/` | NuwaTS | NuwaTS imputation |
| `classification/tabpfn/` | TabPFN | TabPFN v2 classifier |
| `classification/tabpfn_v1/` | TabPFN | TabPFN v1 classifier |

## Code Style

- **Docstrings**: NumPy style
- **Linting**: ruff
- **Type hints**: Required for public functions
