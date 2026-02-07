# Figure Alt Text & Caption Catalog

Alt text and captions for all repository documentation figures. Describes what each figure shows and cites academic sources where the underlying methods originate. Use these when embedding figures in README.md files.

## Embedding Template

```markdown
![ALT TEXT](assets/fig-repo-XX-name.jpg)

*Caption text here.*
```

---

## fig-repo-01 to 12: Core Repository Overview

### fig-repo-01: What Does This Repo Do?

**Alt text**: "Infographic showing a raw pupillary light reflex signal with blink artifacts being cleaned by either traditional methods (LOF, SVM) or foundation models (MOMENT, UniTS, TimesNet), then fed through handcrafted feature extraction and CatBoost classification for glaucoma screening. Data from Najjar et al. 2023 (Br J Ophthalmol), 507 subjects."

**Caption**: *Overview of the foundation_PLR preprocessing pipeline: raw PLR signals are cleaned and classified to screen for glaucoma. Based on Najjar et al. 2023 (Br J Ophthalmol).*

---

### fig-repo-02: The Preprocessing Pipeline

**Alt text**: "Five-stage horizontal pipeline: raw PLR signal flows through outlier detection (11 methods), imputation (8 methods), feature extraction (fixed handcrafted features), and classification (fixed CatBoost), with arrows showing how errors at each stage propagate downstream to affect STRATOS-compliant metrics."

**Caption**: *The preprocessing pipeline: outlier detection and imputation choices propagate through to all downstream metrics. Classifier is fixed (CatBoost).*

---

### fig-repo-03: Why Foundation Models?

**Alt text**: "Comparison of traditional rule-based preprocessing (LOF, OneClassSVM, linear interpolation) versus time-series foundation models (MOMENT, UniTS, TimesNet) that learn temporal patterns from large pretraining corpora and transfer them to pupil signal artifact removal."

**Caption**: *Traditional methods rely on hand-tuned rules; foundation models learn temporal patterns from pretraining data and apply them to new signals.*

---

### fig-repo-04: MLflow = Smart Lab Notebook

**Alt text**: "Comparison of a manual lab notebook with incomplete records versus MLflow's automatic experiment tracking, showing how each of 542 preprocessing runs stores parameters, metrics, and 1000-bootstrap prediction pickles in a searchable database."

**Caption**: *MLflow tracks every experiment automatically. 542 runs with 1000 bootstrap iterations each are fully reproducible.*

---

### fig-repo-05: Hydra Configuration System

**Alt text**: "Diagram showing Hydra composing multiple YAML configuration files (outlier model, imputation model, classifier, evaluation settings) into a single config object, with CLI override capability for factorial experiment sweeps."

**Caption**: *Hydra composes YAML configs hierarchically. Change experiments by editing config files, not source code.*

---

### fig-repo-06: How to Add a New Classifier

**Alt text**: "Two-column infographic. Left: four steps to add a classifier (Python wrapper implementing scikit-learn API, CLS_MODELS YAML config, CLS_HYPERPARAMS search space, pytest tests). Right: the mlflow_registry is updated only after experiments complete, not before. Shows Hydra config composition flow."

**Caption**: *Adding a classifier requires 4 files: wrapper, model config, HPO space, tests. The registry is updated post-hoc, not during training.*

---

### fig-repo-07: How to Create a Figure

**Alt text**: "Six-step flowchart for creating a publication-ready figure: check figure_registry.yaml for specification, write a script in src/viz/ that reads DuckDB, load colors from colors.yaml, apply setup_style(), save with save_figure() including JSON sidecar, and pass pre-commit hooks."

**Caption**: *Figure creation follows a strict pipeline: register, write script (DuckDB read-only), apply style, save with JSON data, pass QA.*

---

### fig-repo-08: Pre-commit Quality Gates

**Alt text**: "Diagram showing pre-commit hooks as a gate between the developer and git history. Hooks check for hardcoded hex colors in R code, sklearn imports in visualization code, registry integrity (11/8/5 method counts), and code formatting via ruff."

**Caption**: *Pre-commit hooks catch hardcoded values, banned imports, and registry violations before they reach the repository.*

---

### fig-repo-09: Test-Driven Development Workflow

**Alt text**: "Circular red-green-refactor TDD diagram: write a failing pytest test (red), implement just enough code to pass (green), refactor while tests stay green. Includes a concrete example from the GED decomposition implementation."

**Caption**: *The red-green-refactor cycle: write the test first, make it pass, clean up. Every feature starts with a failing test.*

---

### fig-repo-10: Prefect Experiment Pipeline

**Alt text**: "Six Prefect subflows arranged in a pipeline: Data Import (data engineer), Outlier Detection (signal processing), Imputation (signal processing), Featurization (domain expert), Classification (biostatistician), Deployment (MLOps). Each reads and writes MLflow artifacts, enabling independent work across roles."

**Caption**: *Six Prefect subflows with role-based ownership. MLflow serves as the contract between teams.*

---

### fig-repo-11: STRATOS Metrics Explained

**Alt text**: "Three-panel infographic based on STRATOS guidelines (Van Calster et al. 2024): Discrimination (AUROC), Calibration (slope, intercept, Brier score), and Clinical Utility (Net Benefit and Decision Curve Analysis). Warning that AUROC alone is insufficient for clinical model evaluation."

**Caption**: *STRATOS requires three metric domains for clinical models, not just AUROC. Based on Van Calster et al. 2024.*

---

### fig-repo-12: Future Experiments

**Alt text**: "Roadmap diagram showing the current baseline (11 outlier methods, 8 imputation methods, 5 classifiers) branching into three future directions: new foundation models (Chronos, Moirai), systematic embedding dimension study (8 to 256 dimensions), and signal decomposition analysis (template fitting, PCA, GED)."

**Caption**: *Extension roadmap: new models, embedding dimension optimization, and signal decomposition. The Hydra config system makes factorial experiments straightforward.*

---

## fig-repo-13 to 43: Extended Technical Documentation

### fig-repo-13: End-to-End Pipeline

**Alt text**: "Horizontal pipeline showing how 500+ scattered CSV files from MLflow are consolidated into a single DuckDB database, then processed through 4-stage preprocessing to produce 40+ publication-quality ggplot2 and matplotlib figures with JSON sidecars for reproducibility."

**Caption**: *From 500+ MLflow CSVs to 40+ publication figures via a single DuckDB database.*

---

### fig-repo-14: Why uv?

**Alt text**: "Side-by-side comparison of pip (requirements.txt lists direct dependencies only, sub-dependency versions float) versus uv (uv.lock pins every transitive dependency to exact versions, ensuring byte-identical environments across machines)."

**Caption**: *uv.lock pins every dependency exactly. pip's requirements.txt only lists what you asked for.*

---

### fig-repo-15: Polars vs Pandas

**Alt text**: "Performance comparison showing Polars achieving 3-5x faster execution and 2-3x less memory usage than optimized pandas for batch processing of PLR bootstrap data, with caveats that the widely-cited 10x claims are inflated. References Patrick Hoefler's benchmarks."

**Caption**: *Polars is 3-5x faster than optimized pandas for our batch workload, with 2-3x less memory. The 10x marketing claims are inflated.*

---

### fig-repo-16: DuckDB Explained

**Alt text**: "Before-and-after diagram: 507 scattered CSV files (difficult to query, no ACID guarantees) replaced by a single DuckDB file (SQL queries in milliseconds, ACID transactions, portable, no server required). Includes comparison table of SQLite, PostgreSQL, and DuckDB trade-offs."

**Caption**: *DuckDB replaces hundreds of CSV files with one portable database. SQL queries on 316 experiment configs return in milliseconds.*

---

### fig-repo-17: Logging Levels

**Alt text**: "Comparison of print() debugging (messages lost in noise during 1000 bootstrap iterations) versus structured logging with Loguru (timestamped, color-coded by severity, rotated files, thread-safe). Shows the log level hierarchy: DEBUG, INFO, WARNING, ERROR, CRITICAL."

**Caption**: *Loguru replaces print() with structured, color-coded, thread-safe logging. Essential for debugging 1000-iteration bootstrap loops.*

---

### fig-repo-18: Two-Block Architecture

**Alt text**: "Two-block diagram: Block 1 (Extraction) computes all metrics from MLflow and writes to DuckDB. Block 2 (Analysis) reads from DuckDB only and generates figures. An enforcement boundary ensures visualization code never recomputes metrics. References CRITICAL-FAILURE-003 meta-learning."

**Caption**: *Block 1 computes metrics (once). Block 2 reads DuckDB (never recomputes). This separation prevents CRITICAL-FAILURE-003.*

---

### fig-repo-19: Subject Stratification

**Alt text**: "Nested rectangle diagram showing 507 total subjects from Najjar et al. 2023, all with ground truth outlier/imputation masks. Subset of 208 (152 control, 56 glaucoma) have classification labels. The remaining 299 contribute to preprocessing evaluation only."

**Caption**: *507 subjects for preprocessing, 208 for classification. The difference is available labels, not lost data.*

---

### fig-repo-20: Error Propagation

**Alt text**: "Four-stage pipeline showing how a missed blink artifact in outlier detection propagates through imputation, feature extraction, and classification. Clean path achieves ground truth AUROC; poor preprocessing degrades downstream metrics progressively."

**Caption**: *Errors cascade: a missed artifact at stage 1 becomes corrupted features at stage 3 and wrong predictions at stage 4.*

---

### fig-repo-21: Bootstrap Confidence Intervals

**Alt text**: "Bootstrap resampling process: 208 subjects are resampled 1000 times with replacement. Each resample produces an AUROC estimate. The 1000 estimates form a distribution, with 2.5th and 97.5th percentiles defining the 95% confidence interval."

**Caption**: *1000 bootstrap resamples produce a distribution of AUROC estimates. The 95% CI shows where the true value likely falls.*

---

### fig-repo-22: The 4 Standard Combos

**Alt text**: "Four preprocessing pipeline cards used for consistent comparison across all figures: (1) Ground truth (pupil-gt + pupil-gt + CatBoost), (2) Best ensemble (7-method ensemble + CSDI + CatBoost), (3) Best single foundation model (MOMENT-gt-finetune + SAITS + CatBoost), (4) Traditional baseline (LOF + SAITS + CatBoost). Loaded from plot_hyperparam_combos.yaml."

**Caption**: *Four standard preprocessing pipelines compared in every figure: ground truth ceiling, best ensemble, best single FM, and traditional baseline.*

---

### fig-repo-23: Data Privacy

**Alt text**: "Two-column classification diagram. Public (committed to GitHub): aggregate metrics, DuckDB with re-anonymized subject IDs (Hxxx/Gxxx codes), figure JSON with ROC/DCA curve coordinates. Private (gitignored): individual PLR traces, subject lookup table mapping Hxxx/Gxxx to original PLRxxxx codes, SERI database. Compliant with Singapore PDPA."

**Caption**: *Aggregate data is public; individual subject data is private. Re-anonymization protects patient identity. SERI institutional data rights and PDPA compliance.*

---

### fig-repo-25: Handcrafted vs Embedding Features

**Alt text**: "Comparison of two featurization approaches: handcrafted features (YAML-configured physiological features like constriction amplitude and recovery slope, producing interpretable feature vectors) versus MOMENT embedding features (768-dimensional learned representations, less interpretable but requiring no domain expertise)."

**Caption**: *Handcrafted features are YAML-configured and interpretable. MOMENT embeddings require no domain expertise but are less interpretable.*

---

### fig-repo-26: Classifier Configuration Architecture

**Alt text**: "Architecture showing the two-file pattern for each of the 5 classifiers: CLS_MODELS/catboost.yaml (fixed model parameters like iterations=1000) and CLS_HYPERPARAMS/catboost_hyperparam_space.yaml (search ranges for Optuna HPO: depth 3-8, learning_rate 0.01-0.3). References Prokhorenkova et al. 2018 (CatBoost) and Chen & Guestrin 2016 (XGBoost)."

**Caption**: *Each classifier has two config files: fixed parameters and HPO search space. CatBoost uses Optuna, XGBoost uses hyperopt.*

---

### fig-repo-27: How to Read Critical Difference Diagrams

**Alt text**: "Annotated critical difference diagram showing method rankings along a number line, with horizontal bars connecting methods that are not statistically significantly different (Nemenyi post-hoc test). Lower rank = better. Connected methods are statistically equivalent."

**Caption**: *Critical difference diagrams: lower rank is better. Connected bars mean no statistically significant difference between methods.*

---

### fig-repo-27b: How to Read Raincloud Plots

**Alt text**: "Annotated raincloud plot combining three visualization layers: a half-violin showing the distribution shape, individual data points (jittered), and a box plot with median and interquartile range. Shows how to read density, spread, and outliers simultaneously."

**Caption**: *Raincloud plots show distribution shape (violin), individual observations (dots), and summary statistics (box) in one view.*

---

### fig-repo-27c: How to Read Specification Curve Analysis

**Alt text**: "Annotated specification curve analysis plot with two panels: upper panel shows the sorted effect sizes for all (outlier x imputation) specifications, lower panel shows a binary matrix indicating which analytical choices were active for each specification."

**Caption**: *Specification curves show how robust a finding is across all analytical choices. Consistent effects across specifications indicate robustness.*

---

### fig-repo-27d: How to Read Instability Plots

**Alt text**: "Annotated pminternal instability plot (Riley 2023, BMC Medicine) showing how individual predictions vary across 200 bootstrap samples. Each subject is a row; horizontal spread indicates prediction instability. Wide spread means the model is unreliable for that patient."

**Caption**: *Instability plots (pminternal, Riley 2023) show prediction consistency across bootstrap samples. Wide spread = unreliable for that patient.*

---

### fig-repo-27e: How to Read SHAP Values

**Alt text**: "Annotated SHAP (Shapley Additive Explanations) beeswarm plot showing feature importance and direction: each dot is a prediction, x-axis is SHAP value (impact on output), color indicates feature value (high/low). Features ranked by mean absolute SHAP value."

**Caption**: *SHAP plots show each feature's contribution to individual predictions. Red = high feature value, blue = low.*

---

### fig-repo-27f: How to Read Risk-Coverage Plots

**Alt text**: "Annotated risk-coverage plot for selective classification (Geifman & El-Yaniv 2017): x-axis is coverage (fraction of predictions made), y-axis is risk (error rate on accepted predictions). Good uncertainty estimates produce a steep initial drop. Measured by AURC (area under the curve, lower is better)."

**Caption**: *Risk-coverage plots: abstaining on uncertain predictions reduces error. AURC measures the trade-off. Based on Geifman & El-Yaniv 2017.*

---

### fig-repo-28: STRATOS Metrics Overview

**Alt text**: "Overview of the five STRATOS metric categories (Van Calster et al. 2024): discrimination (AUROC), calibration (slope, intercept, O:E ratio), overall performance (Brier, scaled Brier), clinical utility (net benefit, DCA curves), and probability distributions. Banned metrics listed: F1, AUPRC, pAUROC, accuracy, Youden."

**Caption**: *Five STRATOS metric categories for clinical prediction models. Banned: F1, AUPRC, accuracy. Based on Van Calster et al. 2024.*

---

### fig-repo-29: MLflow Experiment Tracking

**Alt text**: "MLflow tracking architecture for this repository: 542 runs organized by experiment, each storing parameters (outlier method, imputation method, classifier), metrics (AUROC, Brier, calibration slope), and artifacts (1000-bootstrap prediction pickles). Searchable via MLflow UI or Python API."

**Caption**: *542 MLflow runs store all parameters, metrics, and bootstrap predictions. Searchable and reproducible.*

---

### fig-repo-30: Python-R Interoperability

**Alt text**: "Interop architecture: Python handles ML training, metric extraction, and most visualization. R handles specialized statistical packages (pminternal for model stability per Riley 2023, dcurves for DCA, pmcalibration) and ggplot2 figures. DuckDB and CSV serve as data bridge between languages."

**Caption**: *Python for ML and extraction, R for pminternal stability analysis (Riley 2023) and ggplot2 figures. DuckDB bridges both.*

---

### fig-repo-31: Foundation Model Taxonomy

**Alt text**: "Categorization of time-series foundation models used in this repository: MOMENT and UniTS for outlier detection (anomaly scoring), SAITS and CSDI for imputation (gap filling), TimesNet for both tasks. Each model's pretraining strategy and fine-tuning approach described."

**Caption**: *Foundation model roles: MOMENT/UniTS for outlier detection, SAITS/CSDI for imputation, TimesNet for both.*

---

### fig-repo-32: Virtual Environments Explained

**Alt text**: "Diagram explaining Python virtual environments (.venv) as isolated toolboxes: each project gets its own package versions, keeping the system Python clean. Shows the .venv directory structure and activation command."

**Caption**: *Virtual environments isolate project dependencies. Each project's .venv has its own packages.*

---

### fig-repo-33: Critical Failures Meta-Learning

**Alt text**: "Timeline showing how each critical failure in the project became a documented meta-learning with an automated prevention mechanism: CRITICAL-FAILURE-001 (synthetic data in figures) led to test_data_provenance.py, CRITICAL-FAILURE-002 (mixed featurization) led to registry validation, etc."

**Caption**: *Each critical failure produced a documented meta-learning and an automated guardrail. Mistakes become permanent protections.*

---

### fig-repo-34: README Hierarchy

**Alt text**: "Tree diagram of README.md files in the repository: root README (project overview), configs/README.md (configuration guide), src/*/README.md (module documentation), tests/README.md (testing guide), docs/tutorials/ (learning paths). Each serves a different audience depth."

**Caption**: *Multiple READMEs serve different purposes. Start at root, drill down by topic.*

---

### fig-repo-35: Makefile Commands Overview

**Alt text**: "Grouped listing of key Makefile targets: `make reproduce` runs the full pipeline, `make extract` runs Block 1 only, `make analyze` runs Block 2 only, `make test-local` runs Tier 1 tests, `make figures` generates all figures, `make verify-registry-integrity` checks method counts."

**Caption**: *Key Make targets: `reproduce` (full pipeline), `extract` (Block 1), `analyze` (Block 2), `test-local` (fast feedback).*

---

### fig-repo-36: Hydra Configuration System (Detailed)

**Alt text**: "Detailed Hydra configuration composition diagram: defaults.yaml provides base values, domain configs (CLS_MODELS, OUTLIER_MODELS, MODELS) extend it via config groups, experiment configs compose multiple groups, and CLI arguments override everything. Shows the OmegaConf merge order."

**Caption**: *Hydra composition: defaults → domain configs → experiment → CLI overrides. Based on OmegaConf merge semantics.*

---

### fig-repo-37: Prefect Orchestration (Detailed)

**Alt text**: "Prefect flow coordination diagram for the two-block pipeline: extraction_flow reads MLflow runs and writes DuckDB tables, analysis_flow reads DuckDB and generates figures/stats/LaTeX. Shows task dependencies, retry logic, and checkpoint management."

**Caption**: *Prefect coordinates extraction and analysis flows with retry logic and checkpoint management.*

---

### fig-repo-38: Figure JSON Data Structure

**Alt text**: "Anatomy of a JSON sidecar file accompanying each generated figure: contains figure_id, generation timestamp, source database path with SHA-256 hash, combo identifiers, and all numeric data used to render the figure. Enables exact figure reproduction."

**Caption**: *Every figure's JSON sidecar contains all data needed for exact reproduction: source hash, combos, and numeric values.*

---

### fig-repo-39: Calibration Metrics Explained

**Alt text**: "Educational diagram explaining calibration for clinical prediction models: a well-calibrated model predicts 30% probability when approximately 30% of patients truly have the disease. Shows calibration slope (ideal: 1.0), intercept (ideal: 0.0), O:E ratio (ideal: 1.0), and a smoothed calibration curve with confidence interval."

**Caption**: *Calibration measures whether predicted probabilities match observed frequencies. Slope=1, intercept=0, O:E=1 is perfect.*

---

### fig-repo-40: Net Benefit and DCA

**Alt text**: "Decision Curve Analysis (Vickers & Elkin 2006) for clinical prediction models: net benefit plotted against threshold probability. A useful model provides more net benefit than both the 'treat all' and 'treat none' strategies. Shows how to interpret the threshold range where the model adds clinical value."

**Caption**: *Decision Curve Analysis (Vickers & Elkin 2006): a useful model beats both 'treat all' and 'treat none' across clinically relevant thresholds.*

---

### fig-repo-43: Hierarchical Experiment Configuration

**Alt text**: "Hydra composition tree: paper_2026.yaml references 5 sub-config categories via defaults (data, combos, subjects, figures_config, mlflow_config), each resolving to specific YAML files. Pydantic validation catches config errors at load time. Frozen experiments (frozen: true) ensure publication reproducibility."

**Caption**: *Hierarchical experiment configs: one file composes from 5 sub-configs. Pydantic validates at load time. Frozen configs ensure reproducibility.*

---

## fig-repo-51 to 59: Configuration Documentation

### fig-repo-51: Classifier Configuration Architecture

**Alt text**: "Three-column architecture diagram showing the classifier configuration system: config file locations (CLS_MODELS/*.yaml for fixed parameters, CLS_HYPERPARAMS/*.yaml for search spaces), hyperparameter table for all 5 classifiers (CatBoost, XGBoost, TabPFN, TabM, LogisticRegression), and search space visualization. References Prokhorenkova et al. 2018 (CatBoost), Chen & Guestrin 2016 (XGBoost), Hollmann et al. 2023 (TabPFN)."

**Caption**: *Classifier config architecture: 5 classifiers, each with model config and HPO search space. CatBoost (Prokhorenkova 2018), XGBoost (Chen 2016), TabPFN (Hollmann 2023).*

---

### fig-repo-52: Classifier Paradigms

**Alt text**: "Timeline showing the evolution of tabular classifiers from linear models (logistic regression, 1950s) through tree ensembles (random forests, Breiman 2001; gradient boosting, Chen & Guestrin 2016) to foundation tabular models (TabPFN, Hollmann et al. 2023). Each paradigm shows data requirements, interpretability, and computational cost. References Grinsztajn et al. 2022 (NeurIPS) on why tree-based models still outperform deep learning on tabular data."

**Caption**: *Classifier evolution: linear → trees → boosting → foundation models. Tree-based models remain competitive on tabular data (Grinsztajn 2022 NeurIPS).*

---

### fig-repo-53: Outlier Detection Methods

**Alt text**: "Categorized grid of the 11 registered outlier detection methods: Ground Truth (pupil-gt), Foundation Models (MOMENT-gt-finetune, MOMENT-gt-zeroshot, UniTS-gt-finetune), Deep Learning (TimesNet-gt), Traditional (LOF per Breunig et al. 2000, OneClassSVM per Scholkopf et al. 2001, PROPHET per Taylor & Letham 2018, SubPCA), and 2 Ensembles (7-method majority vote, 4-method FM-only threshold vote). Exactly 11 methods per the registry."

**Caption**: *11 registered outlier methods across 5 categories. LOF (Breunig 2000), OneClassSVM (Scholkopf 2001), MOMENT (Goswami 2024), UniTS (Gao 2024).*

---

### fig-repo-54: Imputation Model Landscape

**Alt text**: "Complexity spectrum of the 8 registered imputation methods: Ground Truth (pupil-gt), Traditional (linear interpolation), Deep Learning (SAITS per Du et al. 2023, CSDI per Tashiro et al. 2021, TimesNet per Wu et al. 2023), Foundation Models (MOMENT-finetune per Goswami et al. 2024, MOMENT-zeroshot), and Ensemble. Shows config file locations under configs/MODELS/."

**Caption**: *8 imputation methods from simple (linear) to complex (CSDI diffusion model, Tashiro 2021). Config files in configs/MODELS/.*

---

### fig-repo-55: Registry as Single Source of Truth

**Alt text**: "Data flow diagram showing the registry pattern: the problem (parsing MLflow run names produces 17 methods including garbage entries like 'anomaly'), the solution (YAML registry defines exactly 11 outlier, 8 imputation, 5 classifier methods), and the enforcement (Python validation functions, pytest assertions, pre-commit hooks all check against the registry)."

**Caption**: *The registry pattern: YAML defines valid methods, Python loads them, 5 layers of validation enforce consistency.*

---

### fig-repo-56: Experiment Configuration Hierarchy

**Alt text**: "Hydra config group composition diagram: reusable config groups (CLS_MODELS, MODELS, OUTLIER_MODELS) function as building blocks that multiple experiments reference. Shows how paper_2026 and a hypothetical paper_2027 can share the same CatBoost config. Override hierarchy: CLI > experiment > combos > groups > defaults. References Hydra documentation."

**Caption**: *Config groups as reusable blocks: define CatBoost once, reference from any experiment. Hydra handles composition and overrides.*

---

### fig-repo-57: DuckDB Schema Diagram

**Alt text**: "Entity-relationship diagram of foundation_plr_results.db: central essential_metrics table (316 rows, composite key: outlier_method + imputation_method + classifier) with columns for all STRATOS metrics, linked 1:N to 5 child tables (predictions, calibration_curves, dca_curves, retention_metrics, distribution_stats). This single ~50MB database replaces ~200GB of ephemeral MLflow artifacts."

**Caption**: *DuckDB schema: essential_metrics at center with 5 child tables. One 50MB database replaces 200GB of MLflow artifacts.*

---

### fig-repo-58: Pre-Commit Enforcement Matrix

**Alt text**: "Matrix of pre-commit hooks and what each catches: ruff (Python lint and formatting), check_r_hardcoding (hex colors, ggsave(), hardcoded dimensions in R code), check_computation_decoupling (sklearn/scipy imports in src/viz/), registry-integrity (method count verification), renv-sync (R lockfile consistency). Includes violation examples and fix instructions."

**Caption**: *Pre-commit hook matrix: what each hook catches, example violations, and how to fix them.*

---

### fig-repo-59: STRATOS Computation Flow

**Alt text**: "Computation pipeline for STRATOS-compliant metrics (Van Calster et al. 2024): for each (outlier x imputation x classifier) config, the trained model's predictions are bootstrap-resampled 1000 times. Each sample computes AUROC, calibration slope/intercept (logistic regression on logit-transformed probabilities), O:E ratio, Brier score, and net benefit (using prevalence=0.0354 from Tham 2014). Aggregated statistics stored in DuckDB essential_metrics table."

**Caption**: *STRATOS metrics (Van Calster 2024) computed via 1000-bootstrap resampling. All computation in Block 1 (extraction), never in visualization.*

---

## fig-repo-61 to 98: Testing, CI/CD, Quality Enforcement, and Onboarding

### fig-repo-61: Test Skip Groups A-H

**Alt text**: "Diagnostic map of 181 pytest test skips organized into 7 groups (A-H), each with a root cause (missing DuckDB databases, R output data not generated, figure filename mismatches, demo data not created, manuscript artifacts, TDD stubs) and the fix command (make extract, make analyze, make r-figures-all) that resolves it."

**Caption**: *Test skip diagnostic: 181 skips across 7 groups. Run `make extract && make analyze && make r-figures-all` to resolve all.*

---

### fig-repo-62: Test Tier to CI Job Mapping

**Alt text**: "Mapping of 4 local test tiers (Tier 0 ruff lint, Tier 1 unit+guardrail, Quality Gates, Tier 3 integration+e2e) to 5 GitHub Actions CI jobs (lint, test-fast with pytest-xdist, quality-gates with registry and decoupling checks, test-integration, r-lint), showing dependency chains and ~35 minute total wall time."

**Caption**: *4 local test tiers map to 5 CI jobs with dependency chains. Total wall time ~35 minutes.*

---

### fig-repo-63: pytest Marker System

**Alt text**: "Venn diagram of 7 pytest markers partitioning 2000+ tests: unit (239, pure functions), integration (96, demo data), e2e (18, full pipeline), slow (31, >30s), guardrail (code quality), data (needs DuckDB), r_required (needs Rscript). Includes command cheatsheet for selective execution."

**Caption**: *7 pytest markers partition 2000+ tests. Use `pytest -m unit` for fast feedback, `pytest -m "not slow"` for CI.*

---

### fig-repo-64: Test Fixture Architecture

**Alt text**: "Tree hierarchy of conftest.py fixtures: root conftest provides session-scoped DuckDB connections, data path constants, and skipif decorators. Per-directory conftest files specialize: test_figure_qa (figure_dir, json_files), test_figure_generation (function-scoped db_connection), unit (pure function fixtures), integration (synthetic_db, demo_subjects)."

**Caption**: *Fixture hierarchy: session-scoped DB connections at root, module-scoped specializations at leaf. Graceful skip-if-absent.*

---

### fig-repo-65: Figure QA Test Categories

**Alt text**: "Figure QA test matrix with 7 pytest test files organized by priority: P0 test_data_provenance.py (catches synthetic data fraud, per CRITICAL-FAILURE-001), P1 test_statistical_validity.py and test_no_nan_ci.py (invalid metrics), P2 test_publication_standards.py and test_rendering_artifacts.py (DPI, dimensions, fonts), P3 test_accessibility.py (color blindness safety). Zero tolerance across all priorities."

**Caption**: *7 QA test files, zero tolerance. P0 catches synthetic data (CRITICAL-FAILURE-001), P1 validates statistics, P2 enforces publication standards.*

---

### fig-repo-66: Test Data Flow

**Alt text**: "Four test data channels: synthetic SYNTH_PLR_DEMO.db (isolated, for unit/guardrail tests), demo_subjects.yaml (8 curated subjects: 4 control H001-H004, 4 glaucoma G001-G004), foundation_plr_results.db (real production data, optional for integration/e2e), and in-memory conftest.py fixtures. Isolation rules prevent cross-contamination."

**Caption**: *4 isolated test data channels: synthetic DB, 8 demo subjects, real DuckDB (optional), in-memory fixtures. No cross-contamination.*

---

### fig-repo-67: Local vs Docker vs CI Tests

**Alt text**: "Three test execution paths: make test-local (~90s, Tier 1 only), make test-all (Docker, all tiers, ~10 min), GitHub Actions automatic on PR (5 jobs: lint, test-fast, quality-gates, test-integration, r-lint). Decision tree for choosing the right path based on development context."

**Caption**: *Three paths: local (fastest), Docker (CI parity), GitHub Actions (PR gate). Choose based on feedback speed needed.*

---

### fig-repo-68: GitHub Actions CI Pipeline

**Alt text**: "Directed acyclic graph of the CI pipeline: 3 parallel starting jobs (lint ~5min with ruff, test-fast ~10min with pytest-xdist, r-lint ~10min), quality-gates after test-fast (registry verification, computation decoupling check), then test-integration with xdist parallel execution. Total wall time ~35 minutes."

**Caption**: *CI DAG: 3 parallel starts, quality gates after fast tests. Defined in `.github/workflows/ci.yml`.*

---

### fig-repo-69: Docker Multi-Image Architecture

**Alt text**: "Four Docker images optimized for different purposes: full development (~2GB, rocker/tidyverse + Python 3.11 + Node.js 20), R-only figure generation (~1GB, rocker/tidyverse + renv 1.1.6 with pinned lockfile), lean test-only (~400MB, python:3.11-slim multi-stage build), and Shiny interactive tools (rocker/shiny on port 3838)."

**Caption**: *4 Docker images: full dev (2GB), R figures (1GB), fast CI tests (400MB), Shiny tools. Each optimized for its use case.*

---

### fig-repo-70: Docker Compose Service Map

**Alt text**: "Docker Compose topology: 7 services in docker-compose.yml with 3 always-available (dev, r-figures, test) and 2 profile-gated (viz on port 3000, shiny on port 3838). Volume mounts map src/ to containers for live code changes during development."

**Caption**: *docker-compose.yml: 7 services, 2 profiles. Volume mounts enable live code changes.*

---

### fig-repo-71: Docker Build Pipeline (CI)

**Alt text**: "Docker CI pipeline: 3 parallel image builds (build-r ~45min, build-test ~30min, build-full ~60min) converge to test-python (~30min), then conditional push to GitHub Container Registry on main branch only. Defined in .github/workflows/docker.yml."

**Caption**: *Docker CI: 3 parallel builds, then test, then push to GHCR on main only.*

---

### fig-repo-72: Pre-commit Hook Chain

**Alt text**: "Sequential execution of 7 pre-commit hooks: (1) ruff format+lint, (2) registry-integrity verifying 11/8/5 method counts across 5 sources, (3) registry-validation running pytest, (4) r-hardcoding-check for hex colors and ggsave() in R, (5) computation-decoupling for sklearn in src/viz/, (6) extraction-isolation-check, (7) figure-isolation-check. First failure stops the chain."

**Caption**: *7 hooks run sequentially. First failure blocks the commit. Known bypass: `SKIP=renv-sync-check` for pre-existing renv issue.*

---

### fig-repo-73: Registry Anti-Cheat System

**Alt text**: "5-layer verification ensuring method count integrity (exactly 11 outlier, 8 imputation, 5 classifier): Layer 1 registry_canary.yaml, Layer 2 mlflow_registry/parameters/ YAML definitions, Layer 3 registry.py EXPECTED_*_COUNT constants, Layer 4 test_registry.py pytest assertions, Layer 5 pre-commit hook. All 5 must agree."

**Caption**: *5-layer registry verification: canary YAML, registry defs, Python constants, pytest, pre-commit. All must agree.*

---

### fig-repo-74: Computation Decoupling Enforcement

**Alt text**: "Two-block enforcement diagram: Block 1 (Extraction) is allowed to import sklearn, scipy, src/stats and write to DuckDB. Block 2 (Visualization) is banned from those imports and may only read DuckDB via SELECT. Enforced by pre-commit hook and pytest tests."

**Caption**: *src/viz/ reads DuckDB only, never computes metrics. Enforced by pre-commit hook and pytest.*

---

### fig-repo-75: Hardcoding Prevention Matrix

**Alt text**: "Four types of hardcoding violations and their enforcement: hex colors (caught by r-hardcoding-check, fix: COLORS dict), literal paths (caught by test_absolute_paths.py, fix: save_figure()), method names (caught by test_method_abbreviations.py, fix: load from YAML), dimensions (caught by test_no_hardcoded_values.py, fix: fig_config)."

**Caption**: *4 hardcoding types, 4 detection mechanisms, 4 correct patterns.*

---

### fig-repo-76: Color System Architecture

**Alt text**: "Color resolution data flow: configs/VISUALIZATION/colors.yaml defines color_definitions (economist_palette, metric_colors) and combo_colors with color_ref pointers. plot_config.py provides the canonical COLORS dict, resolve_color() for CSS-variable-style references, and get_combo_color() resolving via YAML. R reads the same YAML for cross-language consistency."

**Caption**: *YAML defines colors, Python COLORS dict is canonical source. R reads the same YAML for cross-language consistency.*

---

### fig-repo-77: Data Isolation Gates

**Alt text**: "Boundary diagram separating production data (data/public/ with foundation_plr_results.db, /home/petteri/mlruns/ with 410 real runs, figures/generated/) from synthetic data (data/synthetic/ with SYNTH_PLR_DEMO.db, test fixtures). 4 isolation gates enforce the boundary: 2 pre-commit hooks, config validation, and pytest checks."

**Caption**: *4 gates prevent synthetic-production cross-contamination: pre-commit hooks, config validation, pytest.*

---

### fig-repo-78: Data Path Resolution

**Alt text**: "Path resolution chain: Path(__file__).parent.parent resolves to PROJECT_ROOT, which defines canonical constants (RESULTS_DB, CD_DIAGRAM_DB, R_DATA_DIR, FIGURES_DIR, DEMO_DB). conftest.py fixtures consume these with pytest.mark.skipif decorators that gracefully skip when databases are absent."

**Caption**: *PROJECT_ROOT to canonical constants to conftest.py fixtures. Missing databases trigger skips, not failures.*

---

### fig-repo-79: DuckDB Table Relationships

**Alt text**: "Entity-relationship diagram of foundation_plr_results.db: essential_metrics table (316 rows, composite key: outlier_method + imputation_method + classifier) with STRATOS metrics (Van Calster 2024), linked 1:N to predictions (316x208 rows), calibration_curves, dca_curves, retention_metrics, and distribution_stats."

**Caption**: *DuckDB schema: essential_metrics (316 configs) at center with 5 child tables. All STRATOS metrics (Van Calster 2024) stored.*

---

### fig-repo-80: Configuration Space Anatomy

**Alt text**: "3D grid of the experiment configuration space: 11 outlier x 8 imputation x 5 classifiers = 440 total configs. Main analysis fixes CatBoost (88 configs). 316 configs available in DuckDB. Visualizes which dimensions are fixed (classifier) versus varied (outlier, imputation)."

**Caption**: *Configuration space: 11 x 8 x 5 = 440 configs. CatBoost fixed for main analysis (88 configs). 316 in DuckDB.*

---

### fig-repo-81: Figure Registry & Combo System

**Alt text**: "Data flow from figure_registry.yaml (figure ID, script, combo source, privacy level) and plot_hyperparam_combos.yaml (4 standard combos: ground_truth, best_ensemble, best_single_fm, traditional) through generate_all_figures.py which loads both YAMLs, calls plot scripts, and saves figures with JSON sidecars."

**Caption**: *Figure registry: YAML defines figures and combos. generate_all_figures.py orchestrates with JSON sidecar auto-creation.*

---

### fig-repo-82: Metric Registry API

**Alt text**: "Class diagram of MetricRegistry from src/viz/metric_registry.py, grouping all STRATOS metrics (Van Calster et al. 2024) by domain: DISCRIMINATION (auroc), CALIBRATION (slope, intercept, o_e_ratio), CLINICAL_UTILITY (net_benefit at thresholds), OVERALL (brier, scaled_brier). API: has(), get_display_name(), get_domain(), get_duckdb_column()."

**Caption**: *MetricRegistry: Python class grouping STRATOS metrics (Van Calster 2024) by domain with programmatic access.*

---

### fig-repo-83: Repository Directory Map

**Alt text**: "Annotated directory tree of the repository: src/ (Python source with anomaly_detection, classification, extraction, stats, viz, r/ subdirectories), configs/ (Hydra configs with mlflow_registry as source of truth), tests/ (2000+ tests), data/ (public DuckDB databases, private subject lookup, synthetic test data), figures/generated/ (outputs with JSON sidecars), .claude/ (AI agent instructions)."

**Caption**: *Repository at a glance: src/ for code, configs/ for settings, tests/ for validation, data/ for databases, figures/ for outputs.*

---

### fig-repo-84: CLAUDE.md Instruction Hierarchy

**Alt text**: "4-level hierarchy of Claude Code AI agent instructions: Level 1 always-loaded (~30K chars total: root CLAUDE.md project overview, .claude/CLAUDE.md behavior contract, 6 rules files, auto-context.yaml), Level 2 on-demand (.claude/domains/ for MLflow/visualization/manuscript context), Level 3 per-directory overrides (e.g., docs/repo-figures/CLAUDE.md enforcing code-not-results)."

**Caption**: *Claude Code instruction hierarchy: ~30K chars always loaded across 3 levels. Rules files enforce project-specific constraints.*

---

### fig-repo-85: New Developer Quick Start

**Alt text**: "5-step onboarding flowchart: git clone, run setup-dev-environment.sh (installs uv, Python 3.11, creates .venv, runs uv sync, installs pre-commit hooks), make test-local (2042 passed, 181 skipped), optionally make extract + make analyze to resolve skips, ready to contribute. No manual package installation required."

**Caption**: *Clone to first test in 5 minutes. The setup script handles everything: uv, Python, .venv, pre-commit hooks.*

---

### fig-repo-86: How to Add a New Method

**Alt text**: "6-step checklist: add to configs/mlflow_registry/ YAML and increment count, create config in OUTLIER_MODELS/ or MODELS/, implement the detection/imputation interface in src/, write tests, re-extract with make extract + make analyze, update all 5 anti-cheat layers (canary, registry.py constants, test assertions, pre-commit args)."

**Caption**: *Adding a method: 5 places must change (registry, config, source, tests, anti-cheat layers). No shortcuts.*

---

### fig-repo-87: How to Add a New Figure

**Alt text**: "Figure lifecycle: register in figure_registry.yaml, write plot script in src/viz/ (must call setup_style(), load combos from YAML, read DuckDB only, use COLORS dict, call save_figure() with JSON sidecar), generate via generate_all_figures.py, pass pytest tests/test_figure_qa/ with zero tolerance, commit with pre-commit hooks."

**Caption**: *Figure lifecycle: register, write script (DuckDB read-only), generate, pass QA (zero tolerance), commit.*

---

### fig-repo-88: Makefile Target Map

**Alt text**: "Radial map of 40+ Makefile targets in 7 categories: Pipeline (reproduce, extract, analyze), Figures (figures, figure, r-figures), Testing (test-local, test-all, test-figures, test-registry), Docker (docker-build, docker-run), Registry (verify-registry-integrity), Setup (setup-dev-environment), and Utility (clean, type-check)."

**Caption**: *40+ Make targets: `make test-local` for dev, `make reproduce` for full pipeline, `make figures` for publication.*

---

### fig-repo-89: R Figure System Architecture

**Alt text**: "R figure pipeline: renv.lock pins package versions (ggplot2, pminternal per Rhodes 2025, dcurves, pROC, data.table), src/r/figure_system/ provides shared utilities (theme_foundation_plr(), load_color_definitions() from colors.yaml, save_publication_figure()), data flows from DuckDB via Python CSV export to outputs/r_data/ then to R for ggplot2 rendering. Enforced by pre-commit r-hardcoding-check."

**Caption**: *R figure system: renv-pinned packages, shared theme/color utilities, DuckDB-to-CSV-to-ggplot2 pipeline. Pre-commit enforced.*

---

### fig-repo-90: Model Stability Analysis (pminternal)

**Alt text**: "Model stability pipeline using the pminternal R package (Rhodes 2025, based on Riley 2023 BMC Medicine): Python pminternal_wrapper.py sends y_true and y_prob via subprocess with CSV interchange to R's pminternal::validate(), which computes Optimism-corrected Performance Estimates and instability indices with 200 bootstrap iterations."

**Caption**: *Model stability via pminternal (Rhodes 2025, Riley 2023 BMC Medicine): Python-to-R subprocess computing instability indices with 200 bootstrap iterations.*

---

### fig-repo-91: Ensemble Construction

**Alt text**: "Ensemble outlier detector construction: 7 base methods (LOF, MOMENT-gt-finetune, OneClassSVM, PROPHET, SubPCA, TimesNet-gt, UniTS-gt-finetune) combined via majority voting for the full ensemble. A 4-method FM-only subset (MOMENT, TimesNet, UniTS-gt-finetune) uses threshold voting (at least 2 agree) for the thresholded ensemble."

**Caption**: *Full ensemble: 7 methods, majority vote. FM-only ensemble: 4 foundation models, threshold vote. Both in the registry.*

---

### fig-repo-92: Feature Definition YAML

**Alt text**: "Mapping from featuresBaseline.yaml to PLR signal segments: each handcrafted feature defined by time_from (reference: stimulus_onset or minimum_point), time_start/time_end (window in seconds), and stat (min, mean, slope, percentile_75). Example: constriction_amplitude uses stimulus_onset 0.0-1.5s with min. Based on Najjar et al. 2023 (Br J Ophthalmol) feature definitions."

**Caption**: *Feature definitions map YAML configs to PLR signal windows. Configurable, not hardcoded. Based on Najjar et al. 2023 (Br J Ophthalmol).*

---

### fig-repo-93: Common Test Failures & Fixes

**Alt text**: "Diagnostic decision tree for test failures: FileNotFoundError (fix: make extract && make analyze), ModuleNotFoundError (fix: uv sync), pre-commit hook failures (ruff: fix style; registry-integrity: update 5 layers; computation-decoupling: remove banned import; r-hardcoding: use load_color_definitions()), R not found (install R >= 4.4 or use make r-docker-test), genuine failures (read assertion, fix code)."

**Caption**: *Test failure diagnostic: 5 categories with specific fixes. Most resolve with `make extract && make analyze` or `uv sync`.*

---

### fig-repo-94: Extraction Guardrails

**Alt text**: "Three guardrails for the MLflow-to-DuckDB extraction pipeline (410 runs, 1000 bootstraps each): MemoryMonitor (threshold 85% RAM, action: GC + reduce batch), DiskMonitor (threshold 90% disk, action: warn + compact), StallDetector (heartbeat 60s, timeout 300s, action: skip run). CheckpointManager saves progress for resumable extraction."

**Caption**: *3 guardrails protect extraction: memory, disk, stall detection. Checkpoints enable `make reproduce-from-checkpoint`.*

---

### fig-repo-95: Quality Gate Decision Flow

**Alt text**: "Pre-commit hook failure recovery: git commit triggers 7 hooks; all pass = commit created; any fail = commit NOT created (no SHA exists). Recovery: read error, fix issue, re-stage with git add, create a NEW commit. Never use --amend after a hook failure because the failed commit doesn't exist."

**Caption**: *Hook failure recovery: fix, re-stage, NEW commit. Never --amend after a failed hook -- the commit doesn't exist.*

---

### fig-repo-96: Selective Classification & Uncertainty

**Alt text**: "Selective classification for the glaucoma screening pipeline: 208 subjects' predictions sorted by model confidence. High-confidence predictions are accepted; uncertain predictions are referred to a specialist. Measured by AURC (Area Under Risk-Coverage curve, lower is better). Based on Barrenada 2025 and Geifman & El-Yaniv 2017."

**Caption**: *Selective classification: abstain on uncertain predictions, refer to specialist. AURC measures the trade-off. Based on Barrenada 2025, Geifman & El-Yaniv 2017.*

---

### fig-repo-97: Critical Failure Meta-Learnings Timeline

**Alt text**: "Timeline of 6 critical failures and the guardrails they produced: CF-001 synthetic data in figures (→ test_data_provenance.py), CF-002 mixed featurization (→ registry validation), CF-003 computation in visualization (→ decoupling hook), CF-004 hardcoded values (→ r-hardcoding-check), CF-005 visual bug priority (→ bug-first rule), CF-006 shortcuts (→ pre-implementation checklist)."

**Caption**: *6 critical failures, 6 guardrails. Each incident produced a specific automated check.*

---

### fig-repo-98: JSON Sidecar Pattern

**Alt text**: "JSON sidecar reproducibility pattern: every generated figure has a companion JSON file containing figure_id, generation timestamp, source database path with SHA-256 hash, combos used, all numeric data, and summary statistics (n_subjects=208, n_bootstrap=1000). Subject-level JSON files are private and gitignored."

**Caption**: *JSON sidecars: every figure's reproducibility passport. Contains data, source hashes, metadata. Subject-level data is private.*

---

## fig-repro-01 to 24: Reproducibility Series

### fig-repro-01: Reproducibility Crisis in Numbers

**Alt text**: "Infographic presenting statistics on the reproducibility crisis in science: percentage of studies that fail to reproduce across fields, estimated costs of irreproducible research, and the gap between 'code available on request' claims and actual code availability."

**Caption**: *The reproducibility crisis in numbers: failure rates, costs, and the gap between 'available on request' and reality.*

---

### fig-repro-02a: Why Notebooks Fail (ELI5)

**Alt text**: "Simplified explanation of why Jupyter notebooks fail at reproducibility: hidden state from out-of-order cell execution, missing dependency tracking, and the inability to run notebooks end-to-end without manual intervention."

**Caption**: *Notebooks fail at reproducibility: hidden state, out-of-order execution, no dependency tracking.*

---

### fig-repro-02b: Why Notebooks Fail (Expert)

**Alt text**: "Technical analysis of notebook reproducibility failures: non-deterministic cell execution order creates hidden state dependencies, pip freeze captures the environment but not the execution sequence, and notebook diffs are unreadable JSON making code review impractical."

**Caption**: *Expert view: non-deterministic execution, unreadable diffs, and no way to enforce execution order in notebooks.*

---

### fig-repro-03: Five Horsemen of Irreproducibility

**Alt text**: "Five categories of reproducibility failures: environment drift (packages change over time), hidden state (execution order matters), data provenance (unclear data origins), random seeds (non-deterministic results), and manual steps (undocumented human interventions)."

**Caption**: *Five horsemen: environment drift, hidden state, data provenance, random seeds, manual steps.*

---

### fig-repro-04: Levels of Reproducibility

**Alt text**: "Hierarchy of reproducibility levels from weakest to strongest: Level 1 code available (but may not run), Level 2 code runs (but environment differs), Level 3 environment pinned (Docker + lockfiles), Level 4 results match (deterministic execution), Level 5 bitwise identical (byte-for-byte output match)."

**Caption**: *5 levels: from 'code available' (weakest) to bitwise identical results (strongest). This repo targets Level 4.*

---

### fig-repro-05: What Reviewers Check

**Alt text**: "Checklist of what journal reviewers and reproducibility auditors look for: dependency lockfiles, data availability statements, containerized environments, test suites, continuous integration, and documented random seeds."

**Caption**: *Reviewer checklist: lockfiles, data availability, containers, tests, CI, documented seeds.*

---

### fig-repro-06: Cost of Irreproducibility

**Alt text**: "Estimated costs of irreproducible research: wasted researcher time re-implementing others' work, retracted papers, delayed clinical translation, and duplicated effort across labs."

**Caption**: *The cost: wasted time, retractions, delayed translation, duplicated effort across labs.*

---

### fig-repro-07: Docker Is Not Enough

**Alt text**: "Diagram showing why Docker alone doesn't guarantee reproducibility: base images change over time (ubuntu:22.04 today differs from ubuntu:22.04 last month due to apt updates), network dependencies can disappear, and building from Dockerfile is not the same as running from a frozen image."

**Caption**: *Docker helps but isn't sufficient. Base images change, network dependencies disappear. Pinned lockfiles are still needed inside containers.*

---

### fig-repro-08a: Dependency Hell (ELI5)

**Alt text**: "Simplified explanation of dependency conflicts: package A needs version 1.x of a library, package B needs version 2.x, and both cannot coexist. Shows how dependency resolution either fails or makes an arbitrary choice."

**Caption**: *Dependency hell: two packages need incompatible versions of the same library.*

---

### fig-repro-10: System Dependencies

**Alt text**: "Diagram showing the dependency stack beyond Python packages: system libraries (libblas, liblapack), compiler toolchains (gcc), Python interpreter version, OS-level packages (libcurl, libssl), and hardware-specific optimizations (CUDA, MKL) that all affect reproducibility."

**Caption**: *Reproducibility extends beyond pip: system libraries, compilers, OS packages, and hardware all matter.*

---

### fig-repro-11: Version Pinning Strategies

**Alt text**: "Comparison of pinning strategies: no pinning (packages float), requirements.txt (direct deps only), lockfiles (all transitive deps pinned), Docker images (frozen OS + packages), and Nix/Guix (bit-reproducible builds). Shows trade-off between flexibility and reproducibility."

**Caption**: *Pinning strategies from loose (no pin) to strict (Nix). Lockfiles are the practical sweet spot.*

---

### fig-repro-14: Lockfiles as Time Machine

**Alt text**: "Visualization of lockfiles (uv.lock, renv.lock) as a 'time machine': they record the exact versions of every transitive dependency at a point in time, allowing exact environment reconstruction months or years later."

**Caption**: *Lockfiles freeze your entire dependency tree at a point in time. uv.lock for Python, renv.lock for R.*

---

### fig-repro-17: Bitwise vs Functional Reproducibility

**Alt text**: "Comparison of bitwise reproducibility (byte-identical outputs, extremely difficult to achieve due to floating-point non-determinism across hardware) versus functional reproducibility (same conclusions, tolerable numerical differences). This project targets functional reproducibility."

**Caption**: *Bitwise identical is nearly impossible across hardware. Functional reproducibility (same conclusions) is the practical target.*

---

### fig-repro-18: The Base Image Problem

**Alt text**: "Diagram showing how Docker base images (e.g., rocker/tidyverse:4.4.1) change over time as upstream packages are updated. Even with the same tag, apt-get install today produces different packages than last month. Solution: multi-stage builds with explicit version pinning."

**Caption**: *Same Docker tag, different contents over time. Pin everything: base image digest, apt versions, R package versions via renv.*

---

### fig-repro-19: R4R Automatic Artifacts

**Alt text**: "R4R (Ready for Review) automatic artifact generation: when a pipeline completes, it automatically produces all artifacts needed for review: figures with JSON sidecars, DuckDB database, LaTeX tables, and a manifest file listing what was generated and from which data."

**Caption**: *R4R artifacts: figures, JSON sidecars, DuckDB, LaTeX tables, and manifest -- all generated automatically.*

---

### fig-repro-20: DuckDB as Single Source

**Alt text**: "Diagram showing DuckDB as the single archival artifact: MLflow runs (~200GB, ephemeral) are extracted into one DuckDB file (~50MB, permanent). All figures, statistics, and LaTeX tables are generated from this single database, ensuring consistency."

**Caption**: *One DuckDB file replaces 200GB of MLflow runs. All downstream outputs trace back to this single source.*

---

### fig-repro-22: JSON Sidecars for Figure Reproducibility

**Alt text**: "JSON sidecar pattern for figure reproducibility: each PNG figure is accompanied by a JSON file containing the exact data plotted, source database hash, generation parameters, and timestamp. Given the JSON, the figure can be exactly reproduced without re-running the pipeline."

**Caption**: *JSON sidecars make every figure reproducible: data, source hash, and parameters recorded alongside the image.*

---

### fig-repro-23: R4R Success Rate

**Alt text**: "Metrics on R4R (Ready for Review) artifact generation success rate: percentage of pipeline runs producing complete artifact sets, common failure modes, and the relationship between test coverage and artifact completeness."

**Caption**: *R4R success metrics: artifact completeness correlates with test coverage.*

---

### fig-repro-24: Git LFS vs DuckDB

**Alt text**: "Comparison of Git LFS (stores large binary files in Git, requires LFS server, complex setup) versus DuckDB approach (single portable database file, SQL queryable, no special tooling needed). This project chose DuckDB for its queryability and simplicity."

**Caption**: *Git LFS vs DuckDB: LFS needs server infrastructure; DuckDB is a single queryable file.*

---

## fig-trans-01 to 20: Translational Insights

### fig-trans-01: When NOT to Impute

**Alt text**: "Two-panel comparison. Left: PLR signal with a blink gap -- this is a measurement error that should be imputed because the signal existed but couldn't be measured. Right: logistics time series with a warehouse closure gap -- this should NOT be imputed because the gap encodes real information. Decision tree: 'Is the gap a measurement error?' References Van Ness et al. 2023, McTavish et al. 2024 (NeurIPS Proposition 3.1)."

**Caption**: *Not all gaps should be filled. Blink artifacts = impute. Warehouse closures = the gap IS the information. Based on McTavish et al. 2024 (NeurIPS).*

---

### fig-trans-02: TSFM Hype vs Reality

**Alt text**: "Comparison of time-series foundation model marketing claims ('revolutionize everything', '10x better') versus research findings: LLMs fail without periodicity, GPT-4 performs at chance on real forecasting tasks (Schoenegger & Park 2023), normalized metrics can hide poor predictions. Lists where TSFMs genuinely help (anomaly detection, imputation, zero-shot prototyping) versus where they struggle (sparse data, event-driven patterns)."

**Caption**: *TSFM hype vs reality: useful for anomaly detection and imputation, but not universal. GPT-4 is chance-level at forecasting (Schoenegger & Park 2023).*

---

### fig-trans-03: Domain Fit Matrix

**Alt text**: "Heat map of TSFM suitability across domains: dense signal domains (PLR, ECG, EEG, seismic, vibration) score well for anomaly detection and imputation. Sparse domains (EHR, business KPIs, logistics) score poorly, with X marks indicating imputation is harmful, not merely ineffective."

**Caption**: *TSFMs suit dense signals (biosignal, vibration). For sparse data (EHR, logistics), imputation can be harmful.*

---

### fig-trans-04: Sparse vs Dense Time Series

**Alt text**: "Visual comparison: PLR signal at 30 Hz (150 points in 5 seconds, smooth continuous curve where neighbors are correlated) versus supply chain data (11 points over 11 days, each point independent). Different sampling rates require fundamentally different mathematical approaches."

**Caption**: *30 samples/second vs 1 sample/day are different problems. The math that works for one doesn't work for the other.*

---

### fig-trans-05: PLR and Vibration Parallel

**Alt text**: "Side-by-side comparison of pupil signal artifact removal and vibration sensor dropout correction. Both involve detecting anomalies in dense, regularly-sampled signals and reconstructing the true underlying waveform. The preprocessing pipeline is domain-agnostic; what differs are the features and classification targets."

**Caption**: *PLR blink removal and vibration dropout correction are the same mathematical problem. The pipeline transfers; only features differ.*

---

### fig-trans-06: PLR and Seismic Parallel

**Alt text**: "Comparison of PLR anomaly detection (blink artifacts at 30 Hz) with seismic event detection (earthquakes at 100 Hz). Shared methods: LOF, autoencoders, diffusion models, foundation models. Domain-specific: DASFormer for seismic (2025), MOMENT for PLR (Goswami et al. 2024). References DeepDenoiser, Cold Diffusion 2024."

**Caption**: *Earthquake detection is anomaly detection. LOF, autoencoders, and foundation models transfer from PLR to seismic. DASFormer (2025) for seismic, MOMENT (Goswami 2024) for PLR.*

---

### fig-trans-07: PLR and Audio Parallel

**Alt text**: "Three-way comparison of denoising in PLR (30 Hz, time domain), speech (16 kHz, spectrogram), and music (48 kHz, spectrogram). Shared framework: observed = signal + artifact. Same architecture families (autoencoders, masking, diffusion, transformers) apply across all three. PLR works in time domain while audio uses spectrograms due to different sampling rates."

**Caption**: *Denoising PLR, speech, and music share the same math (signal + artifact separation) but operate at different time scales.*

---

### fig-trans-08: Source Separation for Lung Sounds

**Alt text**: "Wearable lung sound source separation: a single microphone captures a mixture of lung sounds, heart sounds, ambient noise, and friction. Analogous to PLR preprocessing but with multiple signals to preserve. References Grooby et al. 2023, McLane et al. 2023, Rennoll et al. 2023. Multiple microphones enable informed separation using reference signals."

**Caption**: *Lung sound source separation: same artifact removal problem as PLR but with multiple overlapping signals. Based on Grooby et al. 2023.*

---

### fig-trans-09: Power Grid Monitoring

**Alt text**: "Parallel between PLR (30 Hz, blink artifacts) and power grid monitoring (60 Hz, voltage sags and transients). Both are dense, regularly-sampled signals with physically interpretable anomalies. The same anomaly detection algorithms apply. Structural similarities: known periodicity, dense sampling, meaningful artifacts."

**Caption**: *PLR at 30 Hz and power grid at 60 Hz: same anomaly detection problem. Dense, periodic, interpretable.*

---

### fig-trans-10: The Dense Signal Club

**Alt text**: "Membership diagram for signals where PLR preprocessing concepts transfer: requirements are >1 sample/second, continuous underlying process, gaps represent errors, neighbors are correlated. Members: biosignals (PLR, ECG, EEG, PPG), engineering (grid, vibration, seismic), audio (speech, music). Not members: EHR, business KPIs, logistics (with recommended alternatives)."

**Caption**: *The 'dense signal club': >1 Hz, continuous process, correlated neighbors. PLR preprocessing transfers to ECG, vibration, seismic, audio.*

---

### fig-trans-11: GMAN for Event-Conditioned Time Series

**Alt text**: "Architecture diagram for GMAN (Graph Mixing Additive Networks, Bechler-Speicher et al. 2025): for event-conditioned time series (missile strikes causing supply chain gaps), GMAN conditions predictions on external events rather than imputing over them. Three trajectories as directed graphs flow through ExtGNAN processing, then DeepSet aggregation, then interpretable prediction. Achieved 76.64% AUROC on PhysioNet."

**Caption**: *GMAN (Bechler-Speicher 2025): conditions on external events instead of imputing over them. Interpretable, handles irregularity natively.*

---

### fig-trans-12: M-GAM for Missing Values as Features

**Alt text**: "M-GAM approach (McTavish et al. 2024, NeurIPS): when missingness is informative (store closed, patient didn't visit), M-GAM treats missing values as features rather than imputing them. Proposition 3.1: perfect imputation can reduce model performance when missingness carries information. Uses adjusted univariate shape curves while maintaining GAM interpretability."

**Caption**: *M-GAM (McTavish 2024 NeurIPS): when gaps ARE information, treat missingness as a feature. Imputation can hurt if gaps are informative (Prop 3.1).*

---

### fig-trans-13: When Simple Baselines Win

**Alt text**: "Decision framework for method selection: simple baselines (constant, linear, moving average) win when data is small, SNR is low, interpretability is required, or real-time constraints exist. Foundation models win when there's a large training corpus, complex temporal patterns, or zero-shot transfer is needed. Five-step practitioner protocol for choosing. References Zeng 2023, Makridakis et al. 2022."

**Caption**: *Simple baselines often win. Use them for small data, low SNR, interpretability, or real-time. FMs for complex patterns and zero-shot transfer.*

---

### fig-trans-14: Domain-Specific vs Generic Models

**Alt text**: "Trade-off between generic foundation models (MOMENT, TimesFM -- broad applicability, zero-shot, opaque) and domain-specific approaches (handcrafted features, EchoNet-Dynamic per Ouyang et al. 2020 -- interpretable, data-efficient, require expertise). Decision matrix: high-stakes clinical applications favor domain-specific; prototyping can use generic. References Grinsztajn et al. 2022 on tabular data."

**Caption**: *Generic FMs trade depth for breadth. Domain-specific models win on high-stakes applications. Prototyping can start generic. Based on Grinsztajn 2022.*

---

### fig-trans-15: PLR Code: What's Domain-Specific?

**Alt text**: "Color-coded directory tree: green (70% domain-agnostic: data_io, models, evaluation, visualization, statistics) vs red (30% domain-specific: PLR features, outlier thresholds, classification targets). For a new domain (e.g., vibration monitoring), only the red 30% needs replacing."

**Caption**: *70% of this codebase is domain-agnostic. To adapt for vibration or seismic, replace the 30% that's PLR-specific.*

---

### fig-trans-16: Configuration vs Hardcoding

**Alt text**: "Diagram contrasting hardcoded values (three files with inconsistent prevalence: 0.035, 0.04, 3.5%) versus centralized configuration (defaults.yaml with prevalence=0.0354 and Tham 2014 citation, all files reading from it). Shows how CRITICAL-FAILURE-002 and CRITICAL-FAILURE-004 motivated this pattern."

**Caption**: *Every hardcoded value is a reproducibility bug. Central YAML config with citations prevents inconsistency (CRITICAL-FAILURE-002, 004).*

---

### fig-trans-17: The Registry Pattern

**Alt text**: "Software pattern diagram: MLflow contained 17 unique outlier method strings including garbage ('anomaly'). The registry YAML defines exactly 11 valid methods. Python accessor functions load from YAML, and multiple consumers (visualization, extraction, tests) validate against the registry. If count differs from registry, the code is broken."

**Caption**: *Registry pattern: YAML defines truth (11 methods), Python loads it, all consumers validate. Prevents garbage from MLflow parsing.*

---

### fig-trans-18: Fork Guide

**Alt text**: "Five-phase fork guide for adapting the repository to a new domain: Phase 0 setup (1 hour), Phase 1 data layer (2-4 hours), Phase 2 feature engineering (4-8 hours, most effort, with mapping table: PLR constriction_amplitude → vibration fft_peak_frequency), Phase 3 registry update (1 hour), Phase 4 threshold calibration (2-4 hours), Phase 5 validation (2-4 hours). 70% of code unchanged."

**Caption**: *Fork, don't rewrite. 5 phases, 70% of code unchanged. Phase 2 (features) is the main effort.*

---

### fig-trans-19: Data Quality Manifesto

**Alt text**: "Six data quality principles derived from project critical failures: (1) fix at source, not downstream (CRITICAL-FAILURE-002), (2) validate against ground truth, (3) no synthetic data in figures (CRITICAL-FAILURE-001), (4) computation in extraction not visualization (CRITICAL-FAILURE-003), (5) document assumptions with citations, (6) test figures with pytest."

**Caption**: *6 data quality principles from hard-won failures. Each principle has an automated guardrail.*

---

### fig-trans-20: Choose Your Approach (Decision Tree)

**Alt text**: "Decision tree for method selection: start with sampling rate. Dense signals (>1 Hz) with measurement-error gaps lead to TSFMs for preprocessing and handcrafted features for classification. Sparse signals with meaningful missingness lead to M-GAM or GMAN. Special cases: event-conditioned (GMAN), real-time (<100ms, simple baselines), small data (<1000 samples, simple baselines), interpretability required (GAMs). Summary table by domain: biosignals favor handcrafted, industrial favors domain + DL, business favors GMAN/M-GAM."

**Caption**: *No universal best method. Dense signals: TSFMs + handcrafted features. Sparse signals: M-GAM/GMAN. Real-time: simple baselines.*

---
