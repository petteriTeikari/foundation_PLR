# Data Directory

This directory contains all data artifacts for the Foundation PLR project: DuckDB databases (results and rankings), privacy-controlled subject lookups, synthetic test data, and R visualization exports. The extraction pipeline (`make extract`) populates `public/` from MLflow; the analysis pipeline (`make analyze`) populates `r_data/` from DuckDB.

> **See also:** [fig-repo-79](../docs/repo-figures/figure-plans/fig-repo-79-duckdb-table-relationships.md) -- DuckDB ER diagram showing all table relationships.

---

## Directory Layout

```
data/
├── public/                          # Anonymized, shareable artifacts
│   ├── foundation_plr_results.db    # Main DuckDB (406 configs, 10 tables)
│   ├── cd_diagram_data.duckdb       # Critical difference diagram rankings
│   └── DATA_MANIFEST.yaml           # File inventory and provenance
│
├── private/                         # PII mappings (gitignored)
│   └── subject_lookup.yaml          # Hxxx/Gxxx → PLRxxxx mapping
│
├── synthetic/                       # Testing only (32 subjects)
│   ├── SYNTH_PLR_DEMO.db            # Synthetic DuckDB for tests
│   └── generation_params.yaml       # Seed + outlier parameters
│
├── r_data/                          # R/ggplot2 visualization exports
│   ├── essential_metrics.csv        # Tabular STRATOS metrics
│   ├── featurization_comparison.json
│   ├── calibration_data.json
│   ├── dca_data.json
│   └── DATA_MANIFEST.yaml           # File inventory
│
├── _checksums.sha256                # Integrity verification
└── README.md                        # This file
```

> **See also:** [fig-repo-77](../docs/repo-figures/figure-plans/fig-repo-77-data-isolation-gates.md) -- Synthetic vs production isolation boundary with 4 enforcement gates.

---

## DuckDB Schema Reference

The main database `public/foundation_plr_results.db` contains these tables. The central table `essential_metrics` has a `config_id` primary key; all auxiliary tables reference it via `config_id` foreign key:

| Table | Rows | Purpose | Consumed By |
|-------|------|---------|-------------|
| `essential_metrics` | 316 | One row per config: AUROC + CI, calibration slope/intercept, O:E ratio, Brier, net benefit at 4 thresholds. Indexed on `(outlier_method, imputation_method, classifier)` | Most `src/viz/` modules |
| `supplementary_metrics` | 316 | Extended metrics (F1, accuracy, AUPR, sensitivity, specificity, PPV, NPV) -- improper per Van Calster, kept for STRATOS appendix | Statistical analyses |
| `calibration_curves` | ~3K | Calibration bin data (`bin_midpoint`, `observed_proportion`, `predicted_mean`, `n_samples`) | `src/viz/calibration_plot.py` |
| `probability_distributions` | ~6K | Predicted probability histograms per outcome class (`bin_start`, `bin_end`, `count`) | `src/viz/prob_distribution.py` |
| `dca_curves` | ~16K | DCA threshold sweep (`threshold`, `net_benefit_model`, `net_benefit_all`, `sensitivity`, `specificity`) | `src/viz/dca_plot.py` |
| `predictions` | 65,728 | Per-subject predictions (`subject_code`, `y_true`, `y_prob`) for 316 configs x 208 subjects | `src/viz/prob_distribution.py`, `src/viz/uncertainty_scatter.py` |
| `retention_metrics` | varies | Selective classification (`retention_rate`, `metric_name`, `metric_value`) | `src/viz/retained_metric.py` |
| `cohort_metrics` | varies | Cohort-level performance (`cohort_fraction`, `metric_name`, `metric_value`) | `src/viz/metric_vs_cohort.py` |
| `distribution_stats` | 316 | Summary statistics per config (`median_cases`, `median_controls`, `mean_cases`, `mean_controls`) | `src/viz/prob_distribution.py` |
| `extraction_checkpoints` | varies | Crash recovery metadata (`status`, `started_at`, `completed_at`, `error_message`) | `src/data_io/streaming_duckdb_export.py` |

**Companion database:** `cd_diagram_data.duckdb` contains rank data for Friedman/Nemenyi post-hoc tests, consumed by `src/r/figures/cd_diagram.R`. Separate from the main DB because CD diagrams require a different data structure (ranks, not raw metrics).

**Key counts:**

| Dimension | Count |
|-----------|-------|
| Unique outlier methods | 11 (from registry) |
| Unique imputation methods | 8 (from registry) |
| Unique classifiers | 5 (from registry) |
| Theoretical full grid | 11 x 8 x 5 = 440 |
| Available configurations | 316 (not all combos ran) |
| Subjects per config | 208 (152 control + 56 glaucoma) |
| Bootstrap iterations | 1000 per config |

**Schema source:** `src/data_io/streaming_duckdb_export.py` defines all table schemas and indices.

> **See also:** [fig-repo-79](../docs/repo-figures/figure-plans/fig-repo-79-duckdb-table-relationships.md) -- Full ER diagram. [fig-repro-20](../docs/repo-figures/figure-plans/fig-repro-20-duckdb-single-source.md) -- Why DuckDB as single source of truth.

---

## Data Provenance

**Source:** Najjar et al. 2023, *Br J Ophthalmol* (DOI: [10.1136/bjophthalmol-2021-319938](https://doi.org/10.1136/bjophthalmol-2021-319938))

| Dataset | N | Purpose |
|---------|---|---------|
| Najjar original | 322 | Full Singapore dataset (SNEC) |
| Our subset (preprocess) | 507 | All subjects with ground truth outlier masks |
| Our subset (classify) | 208 | 152 control + 56 glaucoma (labeled subset) |
| Synthetic (testing) | 32 | 16 train + 16 test, generated from `generation_params.yaml` |

### Three-tier privacy model

| Tier | Identifier | Example | Location | Git-tracked |
|------|-----------|---------|----------|-------------|
| Public | Anonymized codes | H001, G003 | `data/public/` | Yes (DB) |
| Private | Original codes | PLR0042 | `data/private/subject_lookup.yaml` | No (gitignored) |
| Synthetic | Synthetic codes | SYNTH_H001 | `data/synthetic/` | Yes |

> **See also:** [fig-repo-23](../docs/repo-figures/figure-plans/fig-repo-23-data-privacy.md) -- Data privacy classification (public vs private). [fig-repo-78](../docs/repo-figures/figure-plans/fig-repo-78-data-path-resolution.md) -- How tests find their data files.

---

## TSFM Data Format Requirements

Each foundation model expects a different input format. The raw PLR signal (507 subjects x 1981 timepoints, ~30 Hz, univariate) must be adapted to each model's requirements:

| Model | NaN Handling | Seq Length | Windowing | Output Format | Code Path |
|-------|-------------|------------|-----------|---------------|-----------|
| **MOMENT** | NaN-tolerant + `input_mask` | 512 (pad to 2048) | 4 x 512 windows | `TensorDataset` + `DataLoader` | `data_utils.py`, `torch_data.py` |
| **UniTS** | Median fill | Native (1981) | None | PSM-format CSV | `ts_format.py:write_as_psm()` |
| **TimesNet** | Median fill | Native (1981) | None | PyPOTS dict (`X`, `missing_mask`) | `ts_format.py` |
| **SAITS** | NaN-tolerant + mask | Native (1981) | None | PyPOTS dict (`X`, `missing_mask`) | `ts_format.py` |

**Config locations:** `configs/MODELS/{MOMENT,TimesNet,SAITS,CSDI}.yaml`

![TSFM Data Adaptation Pipeline: raw PLR signal (507 subjects, 1981 timepoints) flows through common preprocessing into four model-specific branches -- MOMENT (NaN+mask, pad 2048, 4x512 windows), UniTS (median fill, PSM CSV), TimesNet (median fill, PyPOTS dict), SAITS (NaN+mask, PyPOTS dict).](../docs/repo-figures/assets/fig-data-01-tsfm-data-adaptation-pipeline.jpg)

> **See also:** [fig-data-01 plan](../docs/repo-figures/figure-plans/fig-data-01-tsfm-data-adaptation-pipeline.md) -- Full figure specification with code paths and extension guide.

---

## The `data_provider` Pattern

Many TSFMs (UniTS, Time-Series-Library models) share a common benchmark format called `data_provider`:

```
dataset_folder/
├── train.csv          # Time-series data (columns: timestamp, feature1, ..., label)
├── test.csv
├── dataset_config.yaml  # Metadata (length, channels, frequency)
└── data_factory.py      # DataLoader construction
```

This pattern originates from the [Time-Series-Library](https://github.com/thuml/Time-Series-Library) and is used by benchmark datasets like PSM, MSL, SMAP, SMD. Our PLR data maps to this format via `ts_format.py:write_as_psm()`, which writes the univariate pupil signal as a PSM-compatible CSV.

**Key adaptation:** PLR is a short (1,981 points), univariate, 30 Hz medical biosignal. Standard benchmarks are long (17K-700K), multivariate, low-frequency industrial/weather data. This domain gap requires padding, windowing, and format conversion.

![Common TS Benchmark Dataset Landscape: table comparing 8 standard TSFM benchmarks (ETTh, Weather, ECL, PSM, MSL, SMAP, SMD, SWaT) with PLR highlighted as outlier -- short (2K vs 17K-700K), univariate (1 vs 7-321 channels), medical biosignal vs industrial/weather.](../docs/repo-figures/assets/fig-data-02-benchmark-dataset-landscape.jpg)

> **See also:** [fig-data-02 plan](../docs/repo-figures/figure-plans/fig-data-02-benchmark-dataset-landscape.md) -- Full figure specification with model-dataset matrix.

---

## How to Onboard a New TSFM

Seven-step checklist for adding a new time-series foundation model:

1. **Identify expected input format** -- DataLoader, CSV, PyPOTS dict, or custom?
   - Check the model's training scripts and tutorials
2. **Determine NaN handling** -- Does the model tolerate NaN natively (like MOMENT), or require fill?
   - If NaN-tolerant: generate an `input_mask` (see `torch_data.py`)
   - If fill required: use median fill (see `ts_format.py`)
3. **Determine sequence length constraints** -- Fixed (e.g., MOMENT: 512) or variable?
   - If fixed: implement padding/windowing in `data_utils.py`
4. **Add format writer** to `src/data_io/ts_format.py`
   - Follow the `write_as_psm()` pattern for file-based formats
   - Follow the PyPOTS dict pattern for in-memory formats
5. **Add model config YAML** to `configs/MODELS/{MODEL}.yaml`
   - Include: model hyperparameters, sequence length, batch size, epochs
6. **Wire into pipeline** via `src/anomaly_detection/` or `src/imputation/`
   - Register the model's outlier/imputation method name in `configs/mlflow_registry/`
7. **Validate with registry counts** -- After running, verify:
   - Outlier methods: exactly 11 (or 12 with your addition)
   - Imputation methods: exactly 8 (or 9 with your addition)

![PLR Data Dictionary Structure: tree diagram of the data_dict hierarchy (train/test splits, each containing data/labels/metadata branches with typed columns). Three consumer cards show which branches each task uses -- Outlier Detection (507 subjects), Imputation (507 subjects), Classification (208 subjects).](../docs/repo-figures/assets/fig-data-03-data-dictionary-structure.jpg)

> **See also:** [fig-data-03 plan](../docs/repo-figures/figure-plans/fig-data-03-data-dictionary-structure.md) -- Full figure specification with code paths and extension guide.

---

## Data Versioning

### Current approach: SHA256 checksums

The dataset is **frozen for publication**. We use SHA256 checksums for integrity verification:

```bash
# Verify all checksums
cd /path/to/repo && sha256sum -c data/_checksums.sha256

# Currently tracked files:
#   data/public/foundation_plr_results.db
#   data/public/DATA_MANIFEST.yaml
#   data/synthetic/SYNTH_PLR_DEMO.db
#   data/synthetic/generation_params.yaml
```

### When to upgrade

| Feature | SHA256 | DVC | Datalad |
|---------|--------|-----|---------|
| Integrity verification | Yes | Yes | Yes |
| Version history | No (git only) | Yes | Yes |
| Remote storage (S3/GCS) | No | Yes | Yes (git-annex) |
| Pipeline DAG tracking | No | Yes (`dvc.yaml`) | No |
| Complexity | Zero | Low-Medium | Medium-High |
| Dependencies | `coreutils` | `pip install dvc` | `pip` + `git-annex` |
| **Frozen dataset fit** | **Perfect** | Overkill | Overkill |

**Recommendation:** SHA256 is sufficient during publication freeze. Adopt DVC when:
- Dataset changes between experiments
- Multiple dataset versions need tracking
- External collaborators need data access via cloud storage
- Pipeline DAG tracking becomes valuable

![Data Versioning Decision Tree: flowchart showing frozen dataset leads to SHA256 checksums (current approach, zero dependencies). Changing dataset leads to DVC. Comparison table of SHA256 vs DVC vs Datalad across 9 dimensions. Upgrade triggers and current status callouts.](../docs/repo-figures/assets/fig-data-04-data-versioning-decision-tree.jpg)

> **See also:**
> - [fig-data-04 plan](../docs/repo-figures/figure-plans/fig-data-04-data-versioning-decision-tree.md) -- Data versioning decision tree
> - [fig-repro-24](../docs/repo-figures/figure-plans/fig-repro-24-git-lfs-vs-duckdb.md) -- Git LFS vs DuckDB for large data
> - [fig-repro-17](../docs/repo-figures/figure-plans/fig-repro-17-bitwise-vs-functional.md) -- Bitwise vs functional reproducibility
> - [fig-repro-11](../docs/repo-figures/figure-plans/fig-repro-11-version-pinning-strategies.md) -- Version pinning strategies
> - [fig-repro-14](../docs/repo-figures/figure-plans/fig-repro-14-lockfiles-time-machine.md) -- Lockfiles as time machine
> - [fig-repro-22](../docs/repo-figures/figure-plans/fig-repro-22-json-sidecars-figure-reproducibility.md) -- JSON sidecars for figure reproducibility
> - [docs/planning/reproducibility-and-mlsecops-improvements.md](../docs/planning/reproducibility-and-mlsecops-improvements.md) -- Full DVC deferral rationale

---

## Reproducibility Commands

```bash
# Full pipeline: MLflow → DuckDB → Figures
make reproduce

# Extraction only: MLflow → DuckDB
make extract

# Analysis only: DuckDB → Figures (uses existing DB)
make analyze
make reproduce-from-checkpoint

# R figure generation
make r-figures

# Verify data integrity
sha256sum -c data/_checksums.sha256
```

---

## Cross-References

| Resource | Location |
|----------|----------|
| DuckDB export schema | `src/data_io/streaming_duckdb_export.py` |
| Format conversion code | `src/data_io/ts_format.py` |
| Data loading + windowing | `src/data_io/torch_data.py` |
| Data dictionary creation | `src/data_io/data_wrangler.py` |
| Model configs | `configs/MODELS/` |
| Registry (method validation) | `configs/mlflow_registry/parameters/classification.yaml` |
| Test data path resolution | `tests/conftest.py` |
| Figure registry | `configs/VISUALIZATION/figure_registry.yaml` |
| Architecture overview | `ARCHITECTURE.md` |
| Data I/O module README | `src/data_io/README.md` |

### Figure Plans Referenced in This Document

| Figure | Topic |
|--------|-------|
| [fig-repo-79](../docs/repo-figures/figure-plans/fig-repo-79-duckdb-table-relationships.md) | DuckDB ER diagram |
| [fig-repo-77](../docs/repo-figures/figure-plans/fig-repo-77-data-isolation-gates.md) | Synthetic vs production isolation |
| [fig-repo-78](../docs/repo-figures/figure-plans/fig-repo-78-data-path-resolution.md) | Test data path resolution |
| [fig-repo-23](../docs/repo-figures/figure-plans/fig-repo-23-data-privacy.md) | Data privacy classification |
| [fig-repro-20](../docs/repo-figures/figure-plans/fig-repro-20-duckdb-single-source.md) | Why DuckDB |
| [fig-repro-24](../docs/repo-figures/figure-plans/fig-repro-24-git-lfs-vs-duckdb.md) | Git LFS vs DuckDB |
| [fig-repro-17](../docs/repo-figures/figure-plans/fig-repro-17-bitwise-vs-functional.md) | Bitwise vs functional reproducibility |
| [fig-repro-22](../docs/repo-figures/figure-plans/fig-repro-22-json-sidecars-figure-reproducibility.md) | JSON sidecars |
| [fig-repro-14](../docs/repo-figures/figure-plans/fig-repro-14-lockfiles-time-machine.md) | Lockfiles |
| [fig-repro-11](../docs/repo-figures/figure-plans/fig-repro-11-version-pinning-strategies.md) | Version pinning |
| [fig-data-01](../docs/repo-figures/figure-plans/fig-data-01-tsfm-data-adaptation-pipeline.md) | TSFM data adaptation pipeline |
| [fig-data-02](../docs/repo-figures/figure-plans/fig-data-02-benchmark-dataset-landscape.md) | Benchmark dataset landscape |
| [fig-data-03](../docs/repo-figures/figure-plans/fig-data-03-data-dictionary-structure.md) | Data dictionary structure |
| [fig-data-04](../docs/repo-figures/figure-plans/fig-data-04-data-versioning-decision-tree.md) | Data versioning decision tree |
