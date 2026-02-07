# API Architecture Reference

> **Complete UML/Mermaid documentation for all Foundation PLR modules**

This document provides comprehensive visual documentation of the codebase architecture, class hierarchies, and data flows.

## Table of Contents

1. [Pipeline Overview](#pipeline-overview)
2. [Module Dependency Graph](#module-dependency-graph)
3. [Class Hierarchies](#class-hierarchies)
4. [Sequence Diagrams](#sequence-diagrams)
5. [Data Flow Diagrams](#data-flow-diagrams)
6. [API Reference by Module](#api-reference-by-module)

---

## Pipeline Overview

### High-Level Architecture

```mermaid
graph TB
    subgraph "Input Layer"
        DB[(SERI_PLR_GLAUCOMA.db<br/>507 subjects)]
        CFG[configs/defaults.yaml]
    end

    subgraph "Orchestration Layer"
        HYDRA[Hydra Config Manager]
        PREFECT[Prefect Flow Orchestrator]
        MLFLOW[MLflow Experiment Tracker]
    end

    subgraph "Pipeline Stages"
        S1[Stage 1: Outlier Detection<br/>flow_anomaly_detection.py]
        S2[Stage 2: Imputation<br/>flow_imputation.py]
        S3[Stage 3: Featurization<br/>flow_featurization.py]
        S4[Stage 4: Classification<br/>flow_classification.py]
    end

    subgraph "Support Modules"
        DATA[data_io/]
        STATS[stats/]
        VIZ[viz/]
        LOG[log_helpers/]
    end

    subgraph "Output Layer"
        METRICS[STRATOS Metrics]
        FIGS[Generated Figures]
        ARTIFACTS[MLflow Artifacts]
    end

    DB --> HYDRA
    CFG --> HYDRA
    HYDRA --> PREFECT
    PREFECT --> MLFLOW

    PREFECT --> S1
    S1 --> S2
    S2 --> S3
    S3 --> S4

    DATA -.-> S1 & S2 & S3 & S4
    LOG -.-> S1 & S2 & S3 & S4

    S4 --> METRICS
    METRICS --> STATS
    STATS --> VIZ
    VIZ --> FIGS
    MLFLOW --> ARTIFACTS
```

### Error Propagation Model

```mermaid
graph LR
    subgraph "Stage 1"
        O_ERR[Outlier Detection Error<br/>FN: Missed artifacts]
    end

    subgraph "Stage 2"
        I_ERR[Imputation Error<br/>Corrupted reconstruction]
    end

    subgraph "Stage 3"
        F_ERR[Feature Error<br/>Distorted bins/latencies]
    end

    subgraph "Stage 4"
        C_ERR[Classification Degradation<br/>Reduced AUROC, poor calibration]
    end

    O_ERR -->|"Propagates"| I_ERR
    I_ERR -->|"Propagates"| F_ERR
    F_ERR -->|"Propagates"| C_ERR

    style O_ERR fill:#ff6b6b
    style I_ERR fill:#ffa07a
    style F_ERR fill:#ffd93d
    style C_ERR fill:#ff4757
```

---

## Module Dependency Graph

### Core Module Dependencies

```mermaid
graph TD
    subgraph "Entry Points"
        MAIN[pipeline_PLR.py]
        GENALL[generate_all_figures.py]
    end

    subgraph "Flow Modules"
        FLOW_AD[flow_anomaly_detection.py]
        FLOW_IMP[flow_imputation.py]
        FLOW_FEAT[flow_featurization.py]
        FLOW_CLS[flow_classification.py]
        FLOW_DATA[flow_data.py]
    end

    subgraph "Core Processing"
        AD_MAIN[anomaly_detection.py]
        IMP_MAIN[imputation_main.py]
        FEAT_MAIN[featurize_PLR.py]
        CLS_EVAL[classifier_evaluation.py]
        BOOT[bootstrap_evaluation.py]
    end

    subgraph "Utilities"
        DATA_UTILS[data_utils.py]
        MLFLOW_U[mlflow_utils.py]
        PLOT_CFG[plot_config.py]
        METRIC_REG[metric_registry.py]
    end

    subgraph "Statistics"
        CALIB[calibration_extended.py]
        CLINICAL[clinical_utility.py]
        PMINT[pminternal_wrapper.py]
    end

    MAIN --> FLOW_AD & FLOW_IMP & FLOW_FEAT & FLOW_CLS
    FLOW_AD --> AD_MAIN
    FLOW_IMP --> IMP_MAIN
    FLOW_FEAT --> FEAT_MAIN
    FLOW_CLS --> CLS_EVAL --> BOOT

    FLOW_DATA --> DATA_UTILS
    BOOT --> CALIB & CLINICAL

    GENALL --> VIZ_MODS[viz/*.py]
    VIZ_MODS --> PLOT_CFG & METRIC_REG

    AD_MAIN & IMP_MAIN & CLS_EVAL --> MLFLOW_U
```

### Detailed Module Map

```mermaid
graph LR
    subgraph "anomaly_detection/"
        AD_FLOW[flow_anomaly_detection]
        AD_MAIN[anomaly_detection]
        AD_UTILS[anomaly_utils]
        AD_SKL[outlier_sklearn]
        AD_PROP[outlier_prophet]
        AD_TIME[timesnet_wrapper]
        AD_MOM[momentfm_outlier/]
        AD_UNITS[units/]
    end

    subgraph "imputation/"
        IMP_FLOW[flow_imputation]
        IMP_MAIN[imputation_main]
        IMP_UTILS[imputation_utils]
        IMP_MOM[momentfm/]
    end

    subgraph "classification/"
        CLS_FLOW[flow_classification]
        CLS_EVAL[classifier_evaluation]
        CLS_BOOT[bootstrap_evaluation]
        CLS_CAT[catboost/]
        CLS_XGB[xgboost_main]
        CLS_TABM[tabm_main]
        CLS_TABPFN[tabpfn_main]
    end

    subgraph "stats/"
        STATS_CALIB[calibration_extended]
        STATS_CLIN[clinical_utility]
        STATS_PM[pminternal_wrapper]
        STATS_EFFECT[effect_sizes]
        STATS_STREAM[streaming_exporter]
    end

    subgraph "viz/"
        VIZ_GEN[generate_all_figures]
        VIZ_CFG[plot_config]
        VIZ_INST[fig_instability_plots]
        VIZ_CALIB[fig_calibration]
        VIZ_DCA[fig_dca_curves]
    end

    AD_FLOW --> AD_MAIN
    IMP_FLOW --> IMP_MAIN
    CLS_FLOW --> CLS_EVAL
    CLS_EVAL --> CLS_BOOT
    CLS_BOOT --> STATS_CALIB & STATS_CLIN
    VIZ_GEN --> VIZ_CFG & VIZ_INST & VIZ_CALIB & VIZ_DCA
```

---

## Class Hierarchies

### Outlier Detection Classes

```mermaid
classDiagram
    class BaseOutlierDetector {
        <<abstract>>
        +fit(X, y)
        +predict(X)
        +fit_predict(X, y)
        +decision_function(X)
    }

    class SklearnOutlierWrapper {
        +model: sklearn estimator
        +contamination: float
        +fit(X, y)
        +predict(X)
    }

    class LOFOutlier {
        +n_neighbors: int
        +novelty: bool
    }

    class OneClassSVMOutlier {
        +kernel: str
        +nu: float
    }

    class ProphetOutlier {
        +seasonality_mode: str
        +changepoint_prior: float
        +detect_anomalies(df)
    }

    class MomentOutlier {
        +model_path: str
        +mode: str
        +finetune(data)
        +zeroshot_detect(data)
    }

    class TimesNetOutlier {
        +seq_len: int
        +pred_len: int
        +train_and_detect(data)
    }

    class UniTSOutlier {
        +model_config: dict
        +detect(data)
    }

    class EnsembleOutlier {
        +methods: list
        +voting: str
        +threshold: float
        +combine_predictions(preds)
    }

    BaseOutlierDetector <|-- SklearnOutlierWrapper
    SklearnOutlierWrapper <|-- LOFOutlier
    SklearnOutlierWrapper <|-- OneClassSVMOutlier
    BaseOutlierDetector <|-- ProphetOutlier
    BaseOutlierDetector <|-- MomentOutlier
    BaseOutlierDetector <|-- TimesNetOutlier
    BaseOutlierDetector <|-- UniTSOutlier
    BaseOutlierDetector <|-- EnsembleOutlier
```

### Imputation Classes

```mermaid
classDiagram
    class BaseImputer {
        <<abstract>>
        +fit(X, mask)
        +transform(X, mask)
        +fit_transform(X, mask)
    }

    class LinearImputer {
        +method: str
        +interpolate(signal, mask)
    }

    class SAITSImputer {
        +n_layers: int
        +d_model: int
        +train(X, mask)
        +impute(X, mask)
    }

    class CSDIImputer {
        +diffusion_steps: int
        +denoise(X, mask)
    }

    class TimesNetImputer {
        +configs: dict
        +reconstruct(X, mask)
    }

    class MomentImputer {
        +model_name: str
        +mode: str
        +finetune_impute(X, mask)
        +zeroshot_impute(X, mask)
    }

    class EnsembleImputer {
        +methods: list
        +aggregation: str
        +combine_imputations(results)
    }

    BaseImputer <|-- LinearImputer
    BaseImputer <|-- SAITSImputer
    BaseImputer <|-- CSDIImputer
    BaseImputer <|-- TimesNetImputer
    BaseImputer <|-- MomentImputer
    BaseImputer <|-- EnsembleImputer
```

### Classification Classes

```mermaid
classDiagram
    class BaseClassifier {
        <<abstract>>
        +fit(X, y)
        +predict(X)
        +predict_proba(X)
    }

    class CatBoostWrapper {
        +iterations: int
        +learning_rate: float
        +depth: int
        +fit(X, y)
        +predict_proba(X)
    }

    class XGBoostWrapper {
        +n_estimators: int
        +max_depth: int
        +fit(X, y)
    }

    class TabPFNWrapper {
        +device: str
        +N_ensemble_configurations: int
        +predict_proba(X)
    }

    class TabMWrapper {
        +config: dict
        +train(X, y)
    }

    class LogisticRegressionWrapper {
        +C: float
        +penalty: str
    }

    class BootstrapEvaluator {
        +n_iterations: int
        +ci_alpha: float
        +evaluate(model, X, y)
        +compute_metrics(y_true, y_prob)
    }

    BaseClassifier <|-- CatBoostWrapper
    BaseClassifier <|-- XGBoostWrapper
    BaseClassifier <|-- TabPFNWrapper
    BaseClassifier <|-- TabMWrapper
    BaseClassifier <|-- LogisticRegressionWrapper

    BaseClassifier ..> BootstrapEvaluator : uses
```

### Stats Module Classes

```mermaid
classDiagram
    class CalibrationAnalyzer {
        +compute_slope(y_true, y_prob)
        +compute_intercept(y_true, y_prob)
        +compute_oe_ratio(y_true, y_prob)
        +plot_calibration_curve(y_true, y_prob)
    }

    class ClinicalUtilityAnalyzer {
        +compute_net_benefit(y_true, y_prob, threshold)
        +compute_dca_curve(y_true, y_prob, thresholds)
        +plot_dca(curves)
    }

    class PmInternalWrapper {
        +validate_model(model, data)
        +compute_instability(predictions)
        +plot_instability(results)
    }

    class StreamingExporter {
        +compute_stratos_metrics(y_true, y_prob)
        +export_to_duckdb(metrics, path)
    }

    class EffectSizeCalculator {
        +cohens_d(group1, group2)
        +eta_squared(groups)
        +interpret_effect(value)
    }

    CalibrationAnalyzer --> StreamingExporter : exports to
    ClinicalUtilityAnalyzer --> StreamingExporter : exports to
    PmInternalWrapper --> StreamingExporter : exports to
```

---

## Sequence Diagrams

### Full Pipeline Execution

```mermaid
sequenceDiagram
    participant User
    participant Hydra
    participant Prefect
    participant MLflow
    participant Stage1 as Outlier Detection
    participant Stage2 as Imputation
    participant Stage3 as Featurization
    participant Stage4 as Classification
    participant Stats

    User->>Hydra: python pipeline_PLR.py
    Hydra->>Hydra: Load configs/defaults.yaml
    Hydra->>Prefect: Initialize flow
    Prefect->>MLflow: Create experiment run

    Prefect->>Stage1: flow_anomaly_detection()
    Stage1->>Stage1: Load raw PLR signals
    Stage1->>Stage1: Apply 11 outlier methods
    Stage1->>MLflow: Log outlier masks
    Stage1-->>Prefect: Return processed data

    Prefect->>Stage2: flow_imputation()
    Stage2->>Stage2: Apply 7 imputation methods
    Stage2->>MLflow: Log imputed signals
    Stage2-->>Prefect: Return imputed data

    Prefect->>Stage3: flow_featurization()
    Stage3->>Stage3: Extract amplitude bins
    Stage3->>Stage3: Compute latency features
    Stage3->>MLflow: Log feature matrix
    Stage3-->>Prefect: Return features

    Prefect->>Stage4: flow_classification()
    Stage4->>Stage4: Train CatBoost
    Stage4->>Stage4: Bootstrap evaluation (1000 iter)
    Stage4->>Stats: Compute STRATOS metrics
    Stats-->>Stage4: Return metrics
    Stage4->>MLflow: Log all metrics + artifacts
    Stage4-->>Prefect: Complete

    Prefect-->>User: Pipeline complete
```

### Bootstrap Evaluation Flow

```mermaid
sequenceDiagram
    participant Evaluator as BootstrapEvaluator
    participant Model as CatBoost
    participant Calib as CalibrationAnalyzer
    participant Clinical as ClinicalUtilityAnalyzer
    participant Export as StreamingExporter

    loop 1000 iterations
        Evaluator->>Evaluator: Sample with replacement
        Evaluator->>Model: fit(X_train, y_train)
        Model-->>Evaluator: Fitted model
        Evaluator->>Model: predict_proba(X_test)
        Model-->>Evaluator: y_prob

        Evaluator->>Calib: compute_metrics(y_true, y_prob)
        Calib-->>Evaluator: slope, intercept, oe_ratio

        Evaluator->>Clinical: compute_net_benefit(y_true, y_prob, 0.15)
        Clinical-->>Evaluator: net_benefit

        Evaluator->>Evaluator: Store iteration metrics
    end

    Evaluator->>Evaluator: Compute 95% CI
    Evaluator->>Export: export_stratos_metrics(all_metrics)
    Export-->>Evaluator: Saved to DuckDB
```

### Figure Generation Flow

```mermaid
sequenceDiagram
    participant User
    participant GenAll as generate_all_figures.py
    participant Registry as figure_registry.yaml
    participant PlotCfg as plot_config.py
    participant FigMod as Figure Module
    participant Output as figures/generated/

    User->>GenAll: python generate_all_figures.py
    GenAll->>Registry: Load figure specifications
    GenAll->>PlotCfg: setup_style()

    loop For each figure
        GenAll->>FigMod: Generate figure
        FigMod->>FigMod: Load data from MLflow/DuckDB
        FigMod->>FigMod: Create visualization
        FigMod->>Output: save_figure(fig, name, data)
        Output-->>FigMod: Saved .pdf, .png, .json
    end

    GenAll-->>User: All figures generated
```

---

## Data Flow Diagrams

### Data Transformation Pipeline

```mermaid
graph TD
    subgraph "Raw Data"
        RAW[Raw PLR Signal<br/>shape: (N, T)]
        MASK_GT[Ground Truth Mask<br/>shape: (N, T)]
    end

    subgraph "Outlier Detection Output"
        MASK_PRED[Predicted Outlier Mask<br/>shape: (N, T)]
        OUTLIER_METRICS[Outlier Metrics<br/>F1, Precision, Recall]
    end

    subgraph "Imputation Output"
        IMPUTED[Imputed Signal<br/>shape: (N, T)]
        IMP_METRICS[Imputation Metrics<br/>MAE, RMSE]
    end

    subgraph "Feature Extraction Output"
        BINS[Amplitude Bins<br/>shape: (N, n_bins)]
        LATENCY[Latency Features<br/>shape: (N, 1)]
        FEATURES[Feature Matrix<br/>shape: (N, n_features)]
    end

    subgraph "Classification Output"
        Y_PROB[Predictions<br/>shape: (N,)]
        STRATOS[STRATOS Metrics<br/>Dict]
    end

    RAW --> MASK_PRED
    MASK_GT -.-> OUTLIER_METRICS
    MASK_PRED --> OUTLIER_METRICS

    RAW --> IMPUTED
    MASK_PRED --> IMPUTED

    IMPUTED --> BINS
    IMPUTED --> LATENCY
    BINS --> FEATURES
    LATENCY --> FEATURES

    FEATURES --> Y_PROB
    Y_PROB --> STRATOS
```

### MLflow Artifact Structure

```mermaid
graph TD
    subgraph "MLflow Run"
        RUN[Run ID]

        subgraph "Parameters"
            P1[outlier_method]
            P2[imputation_method]
            P3[classifier]
            P4[n_bootstrap]
        end

        subgraph "Metrics"
            M1[auroc_mean]
            M2[auroc_ci_lo]
            M3[auroc_ci_hi]
            M4[brier_mean]
            M5[calibration_slope]
            M6[net_benefit_15pct]
        end

        subgraph "Artifacts"
            A1[bootstrap_results.pkl]
            A2[confusion_matrix.png]
            A3[roc_curve.json]
            A4[predictions.parquet]
        end
    end

    RUN --> P1 & P2 & P3 & P4
    RUN --> M1 & M2 & M3 & M4 & M5 & M6
    RUN --> A1 & A2 & A3 & A4
```

---

## API Reference by Module

### src/anomaly_detection/

| File | Key Functions | Purpose |
|------|--------------|---------|
| `flow_anomaly_detection.py` | `flow_anomaly_detection(cfg)` | Orchestrates all outlier methods |
| `anomaly_detection.py` | `detect_outliers(signal, method)` | Main detection interface |
| `anomaly_utils.py` | `apply_mask()`, `evaluate_detection()` | Utility functions |
| `outlier_sklearn.py` | `run_lof()`, `run_ocsvm()` | sklearn-based methods |
| `outlier_prophet.py` | `prophet_anomaly_detection()` | Facebook Prophet |
| `timesnet_wrapper.py` | `train_timesnet()`, `detect_timesnet()` | TimesNet wrapper |

### src/imputation/

| File | Key Functions | Purpose |
|------|--------------|---------|
| `flow_imputation.py` | `flow_imputation(cfg)` | Orchestrates all imputation methods |
| `imputation_main.py` | `impute_signal(signal, mask, method)` | Main imputation interface |
| `imputation_utils.py` | `linear_interpolate()`, `evaluate_imputation()` | Utility functions |
| `momentfm/moment_imputation.py` | `moment_impute()` | MOMENT foundation model |

### src/classification/

| File | Key Functions | Purpose |
|------|--------------|---------|
| `flow_classification.py` | `flow_classification(cfg)` | Orchestrates classification |
| `classifier_evaluation.py` | `evaluate_classifier()` | Single classifier evaluation |
| `bootstrap_evaluation.py` | `bootstrap_evaluate()`, `compute_ci()` | Bootstrap confidence intervals |
| `catboost/catboost_main.py` | `train_catboost()`, `predict_catboost()` | CatBoost training |
| `stats_metric_utils.py` | `compute_auroc()`, `compute_brier()` | Metric computation |

### src/stats/

| File | Key Functions | Purpose |
|------|--------------|---------|
| `calibration_extended.py` | `calibration_slope()`, `calibration_intercept()`, `oe_ratio()` | STRATOS calibration |
| `clinical_utility.py` | `net_benefit()`, `decision_curve()` | Clinical utility metrics |
| `pminternal_wrapper.py` | `validate_model()`, `instability_plot()` | R pminternal integration |
| `effect_sizes.py` | `cohens_d()`, `eta_squared()` | Effect size calculations |
| `streaming_exporter.py` | `export_stratos_metrics()` | DuckDB export |

### src/viz/

| File | Key Functions | Purpose |
|------|--------------|---------|
| `generate_all_figures.py` | `main()` | Entry point for all figures |
| `plot_config.py` | `setup_style()`, `COLORS`, `save_figure()` | Plot configuration |
| `metric_registry.py` | `MetricRegistry.get()` | Metric definitions |
| `fig_instability_plots.py` | `plot_riley_instability()`, `plot_kompa_uncertainty()` | Instability figures |
| `fig_calibration.py` | `plot_calibration_curve()` | Calibration plots |
| `fig_dca_curves.py` | `plot_dca()` | Decision curve analysis |

### src/data_io/

| File | Key Functions | Purpose |
|------|--------------|---------|
| `flow_data.py` | `flow_import_data(cfg)` | Data loading flow |
| `data_utils.py` | `load_plr_database()`, `export_to_duckdb()` | Database operations |
| `define_sources_for_flow.py` | `get_data_sources()` | Data source configuration |

### src/log_helpers/

| File | Key Functions | Purpose |
|------|--------------|---------|
| `mlflow_utils.py` | `init_mlflow()`, `log_metrics()`, `log_artifact()` | MLflow integration |
| `mlflow_artifacts.py` | `save_bootstrap_results()`, `load_bootstrap_results()` | Artifact management |
| `log_naming_uris_and_dirs.py` | `get_artifact_path()`, `get_run_name()` | Naming conventions |

---

## Configuration Reference

### Hydra Config Structure

```mermaid
graph TD
    subgraph "configs/"
        DEFAULTS[defaults.yaml]

        subgraph "Overrides"
            OUTLIER[outlier/]
            IMPUTE[imputation/]
            CLASSIFY[classification/]
        end

        subgraph "Visualization"
            VIZ_CFG[VISUALIZATION/plot_config.yaml]
            FIG_REG[figure_registry.yaml]
            COMBOS[plot_hyperparam_combos.yaml]
        end

        subgraph "MLflow"
            MLFLOW_REG[mlflow_registry/experiments.yaml]
        end
    end

    DEFAULTS --> OUTLIER & IMPUTE & CLASSIFY
    DEFAULTS --> VIZ_CFG & FIG_REG & COMBOS
    DEFAULTS --> MLFLOW_REG
```

### Key Configuration Values

| Path | Type | Description |
|------|------|-------------|
| `CLS_EVALUATION.glaucoma_params.prevalence` | float | Disease prevalence (0.0354) |
| `CLS_EVALUATION.BOOTSTRAP.n_iterations` | int | Bootstrap iterations (1000) |
| `CLS_EVALUATION.BOOTSTRAP.alpha_CI` | float | Confidence level (0.95) |
| `VISUALIZATION.dpi` | int | Figure DPI (100) |
| `VISUALIZATION.figure_sizes.paper` | tuple | Paper figure size |

---

## Cross-References

- [ARCHITECTURE.md](https://github.com/petteriTeikari/foundation_PLR/blob/main/ARCHITECTURE.md) - High-level overview
- [KNOWLEDGE_GRAPH.md](KNOWLEDGE_GRAPH.md) - Entity relationships
- [CLAUDE.md](https://github.com/petteriTeikari/foundation_PLR/blob/main/CLAUDE.md) - AI assistant context
- [README.md](https://github.com/petteriTeikari/foundation_PLR/blob/main/README.md) - Getting started

---

*Generated: 2026-01-23*
*Last Updated: 2026-01-23*
