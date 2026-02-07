# Foundation PLR Knowledge Graph

> **Structured knowledge representation for Graph RAG retrieval**

This document defines entities, relationships, and semantic connections for intelligent codebase navigation.

## Graph Schema

```mermaid
erDiagram
    MODULE ||--o{ FILE : contains
    FILE ||--o{ FUNCTION : defines
    FILE ||--o{ CLASS : defines
    CLASS ||--o{ METHOD : has
    FUNCTION ||--o{ PARAMETER : accepts
    FUNCTION ||--o{ RETURN_TYPE : returns
    MODULE ||--o{ DEPENDENCY : depends_on
    FUNCTION ||--o{ FUNCTION : calls
    CONCEPT ||--o{ MODULE : implemented_by
    CONCEPT ||--o{ PAPER : referenced_in
    METRIC ||--o{ FUNCTION : computed_by
```

---

## Entity Definitions

### Modules (Top-Level Packages)

```yaml
entities:
  - id: mod_anomaly_detection
    type: MODULE
    name: anomaly_detection
    path: src/anomaly_detection/
    description: "Outlier detection methods for PLR signals (11 methods)"
    entry_point: flow_anomaly_detection.py
    tags: [stage1, preprocessing, foundation_models, traditional]

  - id: mod_imputation
    type: MODULE
    name: imputation
    path: src/imputation/
    description: "Signal reconstruction methods (7 methods)"
    entry_point: flow_imputation.py
    tags: [stage2, preprocessing, deep_learning]

  - id: mod_featurization
    type: MODULE
    name: featurization
    path: src/featurization/
    description: "Feature extraction from PLR signals"
    entry_point: flow_featurization.py
    tags: [stage3, features, amplitude_bins, latency]

  - id: mod_classification
    type: MODULE
    name: classification
    path: src/classification/
    description: "Classifier training and evaluation"
    entry_point: flow_classification.py
    tags: [stage4, catboost, bootstrap, stratos]

  - id: mod_stats
    type: MODULE
    name: stats
    path: src/stats/
    description: "Statistical metrics and STRATOS compliance"
    entry_point: null
    tags: [metrics, calibration, clinical_utility, effect_sizes]

  - id: mod_viz
    type: MODULE
    name: viz
    path: src/viz/
    description: "Figure generation for manuscript"
    entry_point: generate_all_figures.py
    tags: [visualization, mermaid, matplotlib]

  - id: mod_data_io
    type: MODULE
    name: data_io
    path: src/data_io/
    description: "Database loading and export"
    entry_point: flow_data.py
    tags: [database, duckdb, sqlite, etl]

  - id: mod_log_helpers
    type: MODULE
    name: log_helpers
    path: src/log_helpers/
    description: "MLflow integration and logging"
    entry_point: mlflow_utils.py
    tags: [mlflow, logging, artifacts]
```

### Files (Key Source Files)

```yaml
entities:
  # Anomaly Detection
  - id: file_anomaly_detection
    type: FILE
    name: anomaly_detection.py
    path: src/anomaly_detection/anomaly_detection.py
    module: mod_anomaly_detection
    description: "Main outlier detection dispatcher"
    functions: [detect_outliers, apply_outlier_method, evaluate_outlier_detection]

  - id: file_outlier_sklearn
    type: FILE
    name: outlier_sklearn.py
    path: src/anomaly_detection/outlier_sklearn.py
    module: mod_anomaly_detection
    description: "sklearn-based outlier methods (LOF, OCSVM, SubPCA)"
    functions: [run_lof, run_ocsvm, run_subpca]

  - id: file_outlier_prophet
    type: FILE
    name: outlier_prophet.py
    path: src/anomaly_detection/outlier_prophet.py
    module: mod_anomaly_detection
    description: "Facebook Prophet anomaly detection"
    functions: [prophet_anomaly_detection, fit_prophet_model]

  - id: file_timesnet_wrapper
    type: FILE
    name: timesnet_wrapper.py
    path: src/anomaly_detection/timesnet_wrapper.py
    module: mod_anomaly_detection
    description: "TimesNet foundation model wrapper"
    functions: [train_timesnet, detect_timesnet_anomalies]

  # Imputation
  - id: file_imputation_main
    type: FILE
    name: imputation_main.py
    path: src/imputation/imputation_main.py
    module: mod_imputation
    description: "Main imputation dispatcher"
    functions: [impute_signal, apply_imputation_method, evaluate_imputation]

  - id: file_imputation_utils
    type: FILE
    name: imputation_utils.py
    path: src/imputation/imputation_utils.py
    module: mod_imputation
    description: "Imputation utility functions"
    functions: [linear_interpolate, cubic_interpolate, validate_imputation]

  # Classification
  - id: file_bootstrap_evaluation
    type: FILE
    name: bootstrap_evaluation.py
    path: src/classification/bootstrap_evaluation.py
    module: mod_classification
    description: "Bootstrap confidence interval computation"
    functions: [bootstrap_evaluate, compute_ci, aggregate_metrics]

  - id: file_classifier_evaluation
    type: FILE
    name: classifier_evaluation.py
    path: src/classification/classifier_evaluation.py
    module: mod_classification
    description: "Classifier evaluation with STRATOS metrics"
    functions: [evaluate_classifier, train_and_evaluate, cross_validate]

  - id: file_catboost_main
    type: FILE
    name: catboost_main.py
    path: src/classification/catboost/catboost_main.py
    module: mod_classification
    description: "CatBoost training and prediction"
    functions: [train_catboost, predict_catboost, optimize_hyperparams]

  # Stats
  - id: file_calibration_extended
    type: FILE
    name: calibration_extended.py
    path: src/stats/calibration_extended.py
    module: mod_stats
    description: "STRATOS calibration metrics"
    functions: [calibration_slope, calibration_intercept, oe_ratio, calibration_plot]

  - id: file_clinical_utility
    type: FILE
    name: clinical_utility.py
    path: src/stats/clinical_utility.py
    module: mod_stats
    description: "Clinical utility (Net Benefit, DCA)"
    functions: [net_benefit, decision_curve, standardized_net_benefit]

  - id: file_pminternal_wrapper
    type: FILE
    name: pminternal_wrapper.py
    path: src/stats/pminternal_wrapper.py
    module: mod_stats
    description: "R pminternal integration (Riley 2023)"
    functions: [validate_model, compute_instability, plot_instability]

  - id: file_effect_sizes
    type: FILE
    name: effect_sizes.py
    path: src/stats/effect_sizes.py
    module: mod_stats
    description: "Effect size calculations"
    functions: [cohens_d, hedges_g, eta_squared, partial_eta_squared]

  # Viz
  - id: file_plot_config
    type: FILE
    name: plot_config.py
    path: src/viz/plot_config.py
    module: mod_viz
    description: "Plot configuration and styling"
    functions: [setup_style, save_figure]
    constants: [COLORS, FIGURE_SIZES]

  - id: file_metric_registry
    type: FILE
    name: metric_registry.py
    path: src/viz/metric_registry.py
    module: mod_viz
    description: "Metric definitions for visualization"
    classes: [MetricRegistry, MetricDefinition]

  - id: file_fig_instability
    type: FILE
    name: fig_instability_plots.py
    path: src/viz/fig_instability_plots.py
    module: mod_viz
    description: "Riley 2023 and Kompa 2021 instability plots"
    functions: [plot_riley_instability, plot_kompa_uncertainty]

  # Data I/O
  - id: file_data_utils
    type: FILE
    name: data_utils.py
    path: src/data_io/data_utils.py
    module: mod_data_io
    description: "Database loading and export utilities"
    functions: [load_plr_database, export_to_duckdb, query_mlflow]

  - id: file_mlflow_utils
    type: FILE
    name: mlflow_utils.py
    path: src/log_helpers/mlflow_utils.py
    module: mod_log_helpers
    description: "MLflow experiment tracking"
    functions: [init_mlflow, log_metrics, log_artifact, log_params]
```

### Functions (Key Entry Points)

```yaml
entities:
  # Pipeline Entry Points
  - id: func_flow_anomaly_detection
    type: FUNCTION
    name: flow_anomaly_detection
    file: src/anomaly_detection/flow_anomaly_detection.py
    signature: "flow_anomaly_detection(cfg: DictConfig) -> DataFrame"
    description: "Orchestrates all outlier detection methods for entire dataset"
    calls: [detect_outliers, log_metrics, save_artifact]
    parameters:
      - name: cfg
        type: DictConfig
        description: "Hydra configuration"
    returns: "DataFrame with outlier masks"

  - id: func_flow_imputation
    type: FUNCTION
    name: flow_imputation
    file: src/imputation/flow_imputation.py
    signature: "flow_imputation(cfg: DictConfig) -> DataFrame"
    description: "Orchestrates all imputation methods"
    calls: [impute_signal, log_metrics, save_artifact]

  - id: func_flow_classification
    type: FUNCTION
    name: flow_classification
    file: src/classification/flow_classification.py
    signature: "flow_classification(cfg: DictConfig) -> Dict"
    description: "Orchestrates classification with bootstrap evaluation"
    calls: [bootstrap_evaluate, compute_stratos_metrics, log_metrics]

  # Core Functions
  - id: func_detect_outliers
    type: FUNCTION
    name: detect_outliers
    file: src/anomaly_detection/anomaly_detection.py
    signature: "detect_outliers(signal: np.ndarray, method: str, **kwargs) -> np.ndarray"
    description: "Apply outlier detection method to signal"
    methods_dispatched: [LOF, OCSVM, PROPHET, MOMENT, UniTS, TimesNet, Ensemble]

  - id: func_impute_signal
    type: FUNCTION
    name: impute_signal
    file: src/imputation/imputation_main.py
    signature: "impute_signal(signal: np.ndarray, mask: np.ndarray, method: str) -> np.ndarray"
    description: "Apply imputation method to masked signal"
    methods_dispatched: [linear, SAITS, CSDI, MOMENT, TimesNet, Ensemble]

  - id: func_bootstrap_evaluate
    type: FUNCTION
    name: bootstrap_evaluate
    file: src/classification/bootstrap_evaluation.py
    signature: "bootstrap_evaluate(model, X, y, n_iterations: int) -> Dict"
    description: "Compute bootstrap confidence intervals for all metrics"
    returns: "Dict with mean, CI_lo, CI_hi for each metric"

  # STRATOS Metric Functions
  - id: func_calibration_slope
    type: FUNCTION
    name: calibration_slope
    file: src/stats/calibration_extended.py
    signature: "calibration_slope(y_true: np.ndarray, y_prob: np.ndarray) -> float"
    description: "Compute weak calibration slope via logistic regression"
    reference: "Van Calster 2024"

  - id: func_net_benefit
    type: FUNCTION
    name: net_benefit
    file: src/stats/clinical_utility.py
    signature: "net_benefit(y_true, y_prob, threshold: float) -> float"
    description: "Compute net benefit at decision threshold"
    formula: "TP/n - FP/n * (threshold / (1 - threshold))"
    reference: "Vickers 2006"

  - id: func_decision_curve
    type: FUNCTION
    name: decision_curve
    file: src/stats/clinical_utility.py
    signature: "decision_curve(y_true, y_prob, thresholds: list) -> Dict"
    description: "Compute DCA curve over threshold range"
    returns: "Dict with thresholds, nb_model, nb_all, nb_none"
```

### Concepts (Research Concepts)

```yaml
entities:
  - id: concept_error_propagation
    type: CONCEPT
    name: Error Propagation
    description: "Errors at preprocessing stages cascade to downstream metrics"
    implemented_by: [mod_anomaly_detection, mod_imputation, mod_classification]
    evidence: "Suboptimal outlier detection → imputation errors → feature distortion → AUROC degradation"

  - id: concept_stratos_compliance
    type: CONCEPT
    name: STRATOS Compliance
    description: "Reporting standards for predictive AI (Van Calster 2024)"
    implemented_by: [mod_stats, mod_classification]
    metrics: [AUROC, calibration_slope, calibration_intercept, oe_ratio, net_benefit, brier, scaled_brier]
    reference: paper_van_calster_2024

  - id: concept_foundation_models
    type: CONCEPT
    name: Foundation Models for Preprocessing
    description: "Using pretrained time-series models (MOMENT, UniTS, TimesNet) for outlier detection and imputation"
    implemented_by: [mod_anomaly_detection, mod_imputation]
    finding: "Competitive with traditional methods for preprocessing, but embeddings underperform handcrafted features"

  - id: concept_bootstrap_ci
    type: CONCEPT
    name: Bootstrap Confidence Intervals
    description: "1000-iteration bootstrap for robust metric estimation"
    implemented_by: [file_bootstrap_evaluation]
    parameters:
      n_iterations: 1000
      alpha_CI: 0.95

  - id: concept_model_instability
    type: CONCEPT
    name: Model Instability (Riley 2023)
    description: "Per-patient prediction variability across bootstrap samples"
    implemented_by: [file_pminternal_wrapper, file_fig_instability]
    reference: paper_riley_2023
    metrics: [MAPE, calibration_instability, classification_instability]
```

### Papers (Reference Papers)

```yaml
entities:
  - id: paper_najjar_2023
    type: PAPER
    name: "Najjar et al. 2023"
    title: "Handheld chromatic pupillometry can accurately and rapidly reveal functional loss in glaucoma"
    journal: "Br J Ophthalmol"
    doi: "10.1136/bjophthalmol-2021-319938"
    relevance: "SOURCE DATA - Our dataset is a subset of this study"
    n_subjects: 322
    auroc: 0.94

  - id: paper_van_calster_2024
    type: PAPER
    name: "Van Calster et al. 2024"
    title: "Performance evaluation of predictive AI models to support medical decisions"
    source: "STRATOS Initiative Topic Group 6"
    relevance: "MANDATORY GUIDELINES - Defines which metrics to report"
    metrics_mandated: [AUROC, calibration_plot, calibration_slope, oe_ratio, net_benefit, dca]

  - id: paper_riley_2023
    type: PAPER
    name: "Riley et al. 2023"
    title: "Clinical prediction models and the multiverse of madness"
    journal: "BMC Medicine"
    relevance: "Model instability analysis via pminternal R package"
    r_package: "pminternal"

  - id: paper_kompa_2021
    type: PAPER
    name: "Kompa et al. 2021"
    title: "Second opinion needed: communicating uncertainty in medical machine learning"
    journal: "npj Digital Medicine"
    relevance: "Per-patient uncertainty visualization"
```

### Metrics (STRATOS Metrics)

```yaml
entities:
  - id: metric_auroc
    type: METRIC
    name: AUROC
    full_name: "Area Under ROC Curve"
    category: discrimination
    computed_by: func_bootstrap_evaluate
    range: [0.0, 1.0]
    interpretation: ">0.9 excellent, 0.8-0.9 good, 0.7-0.8 fair"
    stratos_status: REQUIRED

  - id: metric_brier
    type: METRIC
    name: Brier Score
    category: overall
    computed_by: func_bootstrap_evaluate
    range: [0.0, 1.0]
    interpretation: "Lower is better; 0 = perfect"
    stratos_status: REQUIRED

  - id: metric_calibration_slope
    type: METRIC
    name: Calibration Slope
    category: calibration
    computed_by: func_calibration_slope
    range: [0.0, infinity]
    interpretation: "1.0 = perfect calibration; <1 = overfit; >1 = underfit"
    stratos_status: REQUIRED

  - id: metric_oe_ratio
    type: METRIC
    name: O:E Ratio
    full_name: "Observed to Expected Ratio"
    category: calibration
    computed_by: file_calibration_extended
    interpretation: "1.0 = perfect; <1 = overprediction; >1 = underprediction"
    stratos_status: REQUIRED

  - id: metric_net_benefit
    type: METRIC
    name: Net Benefit
    category: clinical_utility
    computed_by: func_net_benefit
    formula: "TP/n - FP/n * (pt / (1 - pt))"
    interpretation: "Higher is better; compare to treat-all and treat-none"
    stratos_status: REQUIRED

  - id: metric_scaled_brier
    type: METRIC
    name: Scaled Brier (IPA)
    full_name: "Index of Prediction Accuracy"
    category: overall
    computed_by: file_calibration_extended
    range: [-infinity, 1.0]
    interpretation: "1 = perfect; 0 = null model; <0 = worse than null"
    stratos_status: RECOMMENDED
```

---

## Relationship Definitions

### Module Dependencies

```yaml
relationships:
  - type: DEPENDS_ON
    from: mod_classification
    to: mod_stats
    description: "Classification uses stats for metric computation"

  - type: DEPENDS_ON
    from: mod_classification
    to: mod_featurization
    description: "Classification requires extracted features"

  - type: DEPENDS_ON
    from: mod_featurization
    to: mod_imputation
    description: "Featurization requires imputed signals"

  - type: DEPENDS_ON
    from: mod_imputation
    to: mod_anomaly_detection
    description: "Imputation requires outlier masks"

  - type: DEPENDS_ON
    from: mod_viz
    to: mod_stats
    description: "Visualization uses stat functions for plots"

  - type: DEPENDS_ON
    from: mod_viz
    to: mod_log_helpers
    description: "Visualization loads data from MLflow"
```

### Implementation Relationships

```yaml
relationships:
  - type: IMPLEMENTS
    from: file_calibration_extended
    to: concept_stratos_compliance
    what: "calibration_slope, calibration_intercept, oe_ratio"

  - type: IMPLEMENTS
    from: file_clinical_utility
    to: concept_stratos_compliance
    what: "net_benefit, decision_curve"

  - type: IMPLEMENTS
    from: file_pminternal_wrapper
    to: concept_model_instability
    what: "Riley 2023 instability metrics via rpy2"

  - type: IMPLEMENTS
    from: file_bootstrap_evaluation
    to: concept_bootstrap_ci
    what: "1000-iteration bootstrap with 95% CI"
```

### Reference Relationships

```yaml
relationships:
  - type: REFERENCES
    from: concept_stratos_compliance
    to: paper_van_calster_2024

  - type: REFERENCES
    from: concept_model_instability
    to: paper_riley_2023

  - type: DATA_SOURCE
    from: mod_data_io
    to: paper_najjar_2023
    description: "Dataset is subset of Najjar 2023"
```

---

## Query Patterns

### Find Implementation Location

```
Q: Where is calibration slope computed?
A: Look for metric_calibration_slope → computed_by → func_calibration_slope → file → calibration_extended.py
   Path: src/stats/calibration_extended.py
```

### Find Related Concepts

```
Q: What does STRATOS compliance require?
A: Look for concept_stratos_compliance → metrics → [AUROC, calibration_slope, ...]
   Implementation: mod_stats, file_calibration_extended, file_clinical_utility
```

### Trace Error Propagation

```
Q: How do outlier errors affect classification?
A: Follow concept_error_propagation:
   mod_anomaly_detection (outlier errors)
   → mod_imputation (imputation errors)
   → mod_featurization (feature distortion)
   → mod_classification (metric degradation)
```

### Find Reference Paper

```
Q: What paper defines net benefit?
A: metric_net_benefit → stratos_status → concept_stratos_compliance → paper_van_calster_2024
   Also: Vickers 2006 (original DCA paper)
```

---

## Navigation Index

### By Task

| Task | Start Entity | Path |
|------|--------------|------|
| "Add new outlier method" | mod_anomaly_detection | → file_outlier_sklearn → see pattern |
| "Compute STRATOS metrics" | concept_stratos_compliance | → implemented_by → file_calibration_extended |
| "Generate figures" | mod_viz | → file_plot_config → setup_style() |
| "Load MLflow data" | mod_log_helpers | → file_mlflow_utils → query_mlflow() |

### By Metric

| Metric | Entity | Implementation |
|--------|--------|----------------|
| AUROC | metric_auroc | func_bootstrap_evaluate |
| Calibration Slope | metric_calibration_slope | func_calibration_slope |
| Net Benefit | metric_net_benefit | func_net_benefit |
| Scaled Brier | metric_scaled_brier | file_calibration_extended |

### By Paper

| Paper | Relevance | Implementation |
|-------|-----------|----------------|
| Najjar 2023 | Data source | mod_data_io |
| Van Calster 2024 | STRATOS metrics | mod_stats |
| Riley 2023 | Instability | file_pminternal_wrapper |
| Kompa 2021 | Uncertainty viz | file_fig_instability |

---

## Usage Notes

### For Claude Code / AI Assistants

1. **Start with concepts** when user asks "why" questions
2. **Start with files/functions** when user asks "where" or "how" questions
3. **Follow relationship chains** to find related implementations
4. **Check paper references** for methodology background

### For Graph RAG Integration

This document can be parsed into a property graph with:
- Nodes: All entities (modules, files, functions, concepts, papers, metrics)
- Edges: All relationships (DEPENDS_ON, IMPLEMENTS, REFERENCES, etc.)
- Properties: All YAML attributes (description, path, signature, etc.)

### Update Protocol

When adding new code:
1. Add FILE entity if new file
2. Add FUNCTION entities for public functions
3. Update DEPENDS_ON relationships
4. Link to CONCEPT if implementing known concept
5. Link to METRIC if computing STRATOS metric

---

*Generated: 2026-01-23*
*Entity Count: ~60 entities, ~30 relationships*
