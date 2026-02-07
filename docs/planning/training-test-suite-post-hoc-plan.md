# Training Test Suite Post-Hoc Plan

## User Request (Verbatim)

> When you are done with previous testing and everything runs
> We have been now mostly working on the mlflow -> duckdb -> ggplot2 pipeline with the json sidecar. We need to next improve the test coverage for the actual training pipeline and the whole experiment reproduction.
>
> As we cannot share this /home/petteri/Dropbox/github-personal/foundation-PLR/SERI_PLR_GLAUCOMA.db to the public with the repo, we need to create a bare minimum synthetic dataset with similar characteristics: a) all the same columns, b) as many time points, c) similar artifacts. You don't have to get ultrarealistic with the actual signal fidelity per se. We are not now creating a paper for a journal specializing in synthetic medical data generation, the data just needs to have 2 classes (healthy vs. glaucoma) and looking like PLR with some artifacts added with ground truth. Then the orig signal can have some noise added on top of the smooth pupil-gt nonlinear trend. So First task is to create this synthetic data module for testing the pipeline, and using it is a golden data for the test suite, and for the people using the repo to tests its behavior. Remember to create some tests to ensure that you just don't take the easiest route and take PII data of the time series! You actually need to create some PLR-like data that is not exactly like input, but you can obviously use the input data and add some noise, do DTW, whatever data augmentation to it in order to further anonymise the data. the whole point of the exercise is that no PII data can be shared in public!
>
> The test for the whole "end-to-end" pipeline and for the pipeline until end of classifier Prefect flow should use the existing /home/petteri/Dropbox/github-personal/foundation-PLR/foundation_PLR/configs/debug_run.yaml that then uses a subset of /home/petteri/Dropbox/github-personal/foundation-PLR/SERI_PLR_GLAUCOMA.db when ran with actual data (analyze /home/petteri/Dropbox/github-personal/foundation-PLR/foundation_PLR/src/data_io) and you should use the same number of subjects then for the synthetic dataset as we did in the debug mode (/home/petteri/Dropbox/github-personal/foundation-PLR/foundation_PLR/configs/defaults.yaml)
>
> DEBUG:
>   # Setting this to None gives you full dataset that could be useful for an end-to-end CI test
>   # Number of epochs or iterations are set to minimum still
>   debug_n_subjects: 8 # this is per label (2) and per split (2) (thus with 4, you get 4x2x2=16 subjects)
>
> The MLflow artifacts should go to the same server as the actual experiment artifacts, but then we should use similar prefix "synth" as we use for the debug runs so that the actual mlflow runs don't get mixed with the actual results. As debug as the name implies used for quick debugging so that we don't have to use the whole dataset and all the epochs/n_training_rounds when just developing the code.
>
> Let's create the plan /home/petteri/Dropbox/github-personal/foundation-PLR/foundation_PLR/docs/planning/training-test-suite-post-hoc-plan.md as we did not really have test coverage when we ran this experiment (which can be checked from commits), but we could make the repo more useful for publication and other researchers and ensure that the pipeline actually works, and that it keeps working with excellent test coverage! And that we actually verify that all the hyperparameter combos work.
>
> Note that the UniTS cannot be reproduced with this same framework, and you should run the experiment from a separate repo https://github.com/petteriTeikari/UniTS for this! We can be a bit lazy for this and not apply the same rigor for running this with the synthetic data as it required too much work for integration for this repo. We should document this very clearly for the repo reproducibility then and note that you need to run this external repo (that is public, forked by me, and some minor tweaks added by me)
>
> Create this plan and optimize it with multiple rounds of code reviewer agents until the plan converges into an optimal plan! Add my prompt as verbatim as well

---

## Executive Summary

This document outlines the plan to create a comprehensive test suite for the Foundation PLR training pipeline. The key components are:

1. **Synthetic PLR Dataset**: Privacy-safe synthetic data matching real dataset schema
2. **End-to-End Pipeline Tests**: Full pipeline testing from data → classification
3. **Hyperparameter Combo Verification**: Ensure all registered methods work
4. **MLflow Artifact Isolation**: Use `synth_` prefix to separate test runs

---

## Current State Analysis

### Test Coverage Assessment

| Area | Current Coverage | Target |
|------|-----------------|--------|
| MLflow → DuckDB extraction | ~70% | ≥90% |
| DuckDB → ggplot2/figures | ~75% | ≥90% |
| Outlier detection pipeline | ~20% | ≥85% |
| Imputation pipeline | ~15% | ≥85% |
| Classification pipeline | ~25% | ≥85% |
| End-to-end pipeline | ~5% | ≥80% |
| **Overall** | **18%** | **≥85%** |

### Data Constraints

| Data Source | Path | Shareable? |
|------------|------|------------|
| Real PLR data | `/home/petteri/Dropbox/github-personal/foundation-PLR/SERI_PLR_GLAUCOMA.db` | **NO** - PII |
| Demo subjects | `data/private/subject_lookup.yaml` | **NO** - PII mapping |
| Subject traces | `data/private/demo_subjects_traces.pkl` | **NO** - PII |

---

## Phase 1: Synthetic PLR Data Generation

### 1.1 Dataset Specifications

**Subject Count**: Match debug mode configuration
- `debug_n_subjects: 8` per label per split
- Total: 8 × 2 (labels) × 2 (splits) = **32 subjects**
- Timepoints: 32 × 1,981 = **63,392 timepoints**

**Database Schema**: Must match `SERI_PLR_GLAUCOMA.db` exactly

| Column | Type | Description | Generation Method |
|--------|------|-------------|-------------------|
| `time` | DOUBLE | 0.0 to 66.0 seconds | Linspace |
| `pupil_orig` | DOUBLE | Raw with outliers | `pupil_gt` + noise + injected outliers |
| `pupil_raw` | DOUBLE | Outliers as NULL | `pupil_orig` with outliers → NULL |
| `pupil_gt` | DOUBLE | Ground truth | Synthetic PLR curve |
| `pupil_orig_imputed` | DOUBLE | Filled for outlier detection | Linear interpolation of `pupil_orig` |
| `pupil_raw_imputed` | DOUBLE | Filled for downstream | Linear interpolation of `pupil_raw` |
| `Red` | DOUBLE | Red light stimulus | Fixed protocol pattern |
| `Blue` | DOUBLE | Blue light stimulus | Fixed protocol pattern |
| `light_stimuli` | DOUBLE | Combined R+B | `Red + Blue` |
| `time_orig` | DOUBLE | Original time | Same as `time` |
| `subject_code` | VARCHAR | Subject ID | `SYNTH_H001`, `SYNTH_G001`, etc. |
| `no_outliers` | BIGINT | Outlier count | Count from `outlier_mask` |
| `Age` | DECIMAL | Subject age | Uniform(40, 80) |
| `class_label` | VARCHAR | control/glaucoma | Based on subject prefix |
| `outlier_mask` | INTEGER | 1=outlier, 0=valid | Generated with artifact injection |
| `imputation_mask` | BOOLEAN | Imputed indicator | Where `pupil_raw` is NULL |
| `split` | VARCHAR | train/test | Stratified assignment |

### 1.2 PLR Curve Generation

**Ground Truth (`pupil_gt`) Generation:**

```python
def generate_plr_curve(class_label: str, seed: int) -> np.ndarray:
    """
    Generate a synthetic PLR curve with physiologically plausible characteristics.

    Parameters:
        class_label: "control" or "glaucoma"
        seed: Random seed for reproducibility

    Returns:
        1D array of 1981 timepoints representing pupil diameter
    """
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 66, 1981)  # 30 fps, 66 seconds

    # Baseline pupil size (mm): Healthy 4-7mm, Glaucoma often smaller
    if class_label == "control":
        baseline = rng.uniform(4.5, 6.5)
    else:
        baseline = rng.uniform(3.5, 5.5)  # Slightly smaller for glaucoma

    # Red light response (rod-mediated)
    red_on = 10  # seconds
    red_off = 25
    red_amplitude = rng.uniform(0.8, 1.5) if class_label == "control" else rng.uniform(0.5, 1.0)

    # Blue light response (melanopsin-mediated, PIPR)
    blue_on = 35
    blue_off = 50
    blue_amplitude = rng.uniform(1.2, 2.0) if class_label == "control" else rng.uniform(0.8, 1.5)

    # Recovery dynamics differ between healthy and glaucoma
    recovery_tau = rng.uniform(5, 10) if class_label == "control" else rng.uniform(8, 15)

    # Construct the PLR response
    pupil = np.ones_like(t) * baseline

    # Red light constriction and recovery
    red_mask = (t >= red_on) & (t < red_off)
    red_recovery_mask = t >= red_off
    # ... (detailed physiological model)

    return pupil
```

**Noise and Artifact Injection:**

```python
def inject_artifacts(pupil_gt: np.ndarray, outlier_pct: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Inject realistic artifacts into clean PLR signal.

    Parameters:
        pupil_gt: Clean ground truth signal
        outlier_pct: Fraction of timepoints to corrupt (0.02 to 0.40)
        seed: Random seed

    Returns:
        pupil_orig: Corrupted signal with outliers
        outlier_mask: Binary mask (1=outlier)
    """
    rng = np.random.RandomState(seed)
    n = len(pupil_gt)
    outlier_mask = np.zeros(n, dtype=int)
    pupil_orig = pupil_gt.copy()

    n_outliers = int(n * outlier_pct)

    # Type 1: Blinks (sharp drops to zero, ~100-300ms duration)
    n_blinks = rng.poisson(n_outliers * 0.3)  # 30% are blinks
    for _ in range(n_blinks):
        start = rng.randint(0, n - 15)
        duration = rng.randint(3, 9)  # 100-300ms at 30fps
        pupil_orig[start:start+duration] = rng.uniform(0, 0.5)
        outlier_mask[start:start+duration] = 1

    # Type 2: Segmentation noise (random spikes)
    n_spikes = int(n_outliers * 0.5)
    spike_locs = rng.choice(n, size=n_spikes, replace=False)
    pupil_orig[spike_locs] *= rng.uniform(0.7, 1.5, size=n_spikes)
    outlier_mask[spike_locs] = 1

    # Type 3: Missing data (NaN regions)
    n_gaps = rng.poisson(n_outliers * 0.2 / 30)
    for _ in range(n_gaps):
        start = rng.randint(0, n - 60)
        duration = rng.randint(30, 60)
        pupil_orig[start:start+duration] = np.nan
        outlier_mask[start:start+duration] = 1

    # Add baseline noise everywhere
    pupil_orig += rng.normal(0, 0.05, size=n)

    return pupil_orig, outlier_mask
```

### 1.3 Privacy Safeguards

**CRITICAL**: The synthetic data must NOT be derived from or resemble individual patient data.

**Safeguards:**

1. **No Template Reuse**: Do NOT load real PLR curves as templates
2. **DTW Prohibited**: Do NOT use DTW to morph real data (preserves structure)
3. **Statistical Mimicry Only**: Use population-level statistics (mean, variance) only
4. **Unique Generation**: Each synthetic subject is independently generated from parametric models

**Verification Tests:**

```python
def test_no_pii_leakage():
    """Ensure synthetic data is not derived from real subjects."""
    real_db = duckdb.connect("SERI_PLR_GLAUCOMA.db")
    synth_db = duckdb.connect("SYNTH_PLR_DEMO.db")

    real_curves = real_db.execute("SELECT pupil_gt FROM train").fetchnumpy()
    synth_curves = synth_db.execute("SELECT pupil_gt FROM train").fetchnumpy()

    for synth_curve in synth_curves:
        for real_curve in real_curves:
            # Pearson correlation must be below threshold
            corr = np.corrcoef(synth_curve, real_curve)[0, 1]
            assert abs(corr) < 0.85, "Synthetic curve too similar to real data!"

            # DTW distance must be above threshold
            dtw_dist = dtw.distance(synth_curve, real_curve)
            assert dtw_dist > 0.5, "DTW distance too small - possible PII leakage!"

def test_synthetic_subject_codes_distinct():
    """Ensure no overlap with real subject codes."""
    synth_db = duckdb.connect("SYNTH_PLR_DEMO.db")
    codes = synth_db.execute("SELECT DISTINCT subject_code FROM train").fetchall()

    for code in codes:
        assert code.startswith("SYNTH_"), f"Invalid subject code: {code}"
        assert "PLR" not in code, "Real subject code pattern detected!"
```

### 1.4 Module Structure

```
src/synthetic/
├── __init__.py
├── plr_generator.py          # Core PLR curve generation
├── artifact_injection.py     # Noise and outlier injection
├── database_builder.py       # Assemble DuckDB from synthetic subjects
├── privacy_validator.py      # PII leakage prevention tests
└── demo_dataset.py           # Public API for generating demo data
```

**Output Location:**

```
data/synthetic/
├── SYNTH_PLR_DEMO.db         # Main synthetic database (COMMITTED to git)
└── generation_params.yaml    # Parameters used for generation (COMMITTED)
```

---

## Phase 2: End-to-End Pipeline Tests

### 2.1 Test Architecture

```
tests/e2e/
├── test_full_pipeline_synthetic.py   # Full pipeline with synthetic data
├── test_outlier_detection_flow.py    # Outlier detection Prefect flow
├── test_imputation_flow.py           # Imputation Prefect flow
├── test_classification_flow.py       # Classification Prefect flow
├── conftest.py                       # Fixtures for synthetic data
└── markers.py                        # Pytest markers (slow, gpu, etc.)
```

### 2.2 MLflow Backend Architecture

**Design Principle**: The architecture must be flexible to support multiple MLflow backends without code changes.

#### Supported Backends

| Backend | Use Case | Configuration |
|---------|----------|---------------|
| **File-based (`mlruns/`)** | Local development, CI testing | `MLFLOW_TRACKING_URI=mlruns/` |
| **SQLite** | Lightweight local server | `MLFLOW_TRACKING_URI=sqlite:///mlflow.db` |
| **Self-hosted server** | Team collaboration | `MLFLOW_TRACKING_URI=http://mlflow.example.com:5000` |
| **DagsHub** | Academic sharing + arXiv integration | `MLFLOW_TRACKING_URI=https://dagshub.com/<user>/<repo>.mlflow` |
| **Managed (Databricks, AWS)** | Enterprise deployment | Provider-specific URIs |

#### DagsHub Integration (Recommended for Academic Use)

[DagsHub](https://dagshub.com/) provides MLflow hosting with unique benefits for academic research:

1. **arXiv Integration**: Direct linking between experiments and arXiv papers
   - See: [DagsHub + arXiv announcement](https://blog.arxiv.org/2023/03/27/new-arxivlabs-integrations-provide-insights-into-the-academic-influence-of-researchers-and-enable-reproducibility-through-access-to-data-and-code/)
   - Enables reviewers to access experiment data directly from paper

2. **Free for Open Source**: Academic-friendly pricing
3. **Git + DVC + MLflow**: Unified platform for code, data, and experiments
4. **Easy Setup**:
   ```python
   import dagshub
   dagshub.init(repo_owner='<username>', repo_name='foundation-PLR', mlflow=True)

   # Now mlflow.* calls automatically log to DagsHub
   import mlflow
   mlflow.log_metric("auroc", 0.911)
   ```

**Documentation**: https://dagshub.com/docs/integration_guide/mlflow_tracking/

#### Backend Configuration

```yaml
# configs/defaults.yaml - MLflow backend selection
MLFLOW:
  # Backend options: "file", "sqlite", "server", "dagshub"
  backend: "file"  # Default: local file-based

  # Backend-specific URIs (only the active backend is used)
  backends:
    file:
      tracking_uri: "mlruns/"  # Local directory
    sqlite:
      tracking_uri: "sqlite:///mlflow.db"
    server:
      tracking_uri: "${oc.env:MLFLOW_TRACKING_URI,http://localhost:5000}"
    dagshub:
      # Set DAGSHUB_TOKEN environment variable for authentication
      tracking_uri: "https://dagshub.com/${oc.env:DAGSHUB_USER}/${oc.env:DAGSHUB_REPO}.mlflow"

  # Experiment naming
  experiment_prefix: ""  # Empty for production, "debug_" or "synth_" for testing
```

#### Switching Backends

```bash
# Local development (default)
python src/pipeline_PLR.py

# Use DagsHub for shareable experiments
export DAGSHUB_USER=petteriTeikari
export DAGSHUB_REPO=foundation-PLR
python src/pipeline_PLR.py mlflow.backend=dagshub

# Use self-hosted server
export MLFLOW_TRACKING_URI=http://mlflow.mylab.edu:5000
python src/pipeline_PLR.py mlflow.backend=server
```

#### Experiment Naming Convention

| Context | Experiment Prefix | Example |
|---------|------------------|---------|
| Production runs | None | `PLR_Classification` |
| Debug runs | `debug_` | `debug_PLR_Classification` |
| Synthetic CI runs | `synth_` | `synth_PLR_Classification` |

#### Synthetic Run Configuration

```yaml
# configs/synthetic_run.yaml
defaults:
  - defaults
  - override EXPERIMENT: synthetic

EXPERIMENT:
  experiment_prefix: "synth_"
  debug: True
  use_demo_data: True  # Use synthetic data

DEBUG:
  debug_n_subjects: 8

DATA:
  import_from_DuckDB: True
  filename_DuckDB: 'SYNTH_PLR_DEMO.db'  # Use synthetic database
  demo_data_path: 'data/synthetic/'     # Path to synthetic data

MLFLOW:
  backend: "file"  # Use local file backend for CI (no server required)
```

### 2.3 Hyperparameter Combo Verification

**Registry-Driven Testing:**

```python
# tests/e2e/test_all_combos.py

@pytest.fixture
def valid_outlier_methods():
    """Load all 11 valid outlier methods from registry."""
    from src.data_io.registry import get_valid_outlier_methods
    return get_valid_outlier_methods()

@pytest.fixture
def valid_imputation_methods():
    """Load all 8 valid imputation methods from registry."""
    from src.data_io.registry import get_valid_imputation_methods
    return get_valid_imputation_methods()

@pytest.mark.slow
@pytest.mark.parametrize("outlier_method", [
    pytest.param("pupil-gt", id="ground_truth"),
    pytest.param("LOF", id="traditional"),
    pytest.param("MOMENT-gt-finetune", id="foundation_model"),
    # Add more as needed...
])
def test_outlier_method_runs(outlier_method, synthetic_db):
    """Verify each outlier method completes without error."""
    cfg = load_config_with_overrides(
        outlier_models=[outlier_method],
        use_demo_data=True
    )
    result = run_outlier_detection(cfg, synthetic_db)

    assert result.success, f"Outlier method {outlier_method} failed: {result.error}"
    assert result.metrics["outlier_f1"] >= 0.0, "F1 must be non-negative"
```

### 2.4 Test Markers and Categories

```python
# conftest.py

def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "gpu: requires GPU")
    config.addinivalue_line("markers", "synthetic: uses synthetic data")
    config.addinivalue_line("markers", "e2e: end-to-end pipeline test")
    config.addinivalue_line("markers", "integration: integration test")
    config.addinivalue_line("markers", "unit: unit test")
```

**Run Commands:**

```bash
# Quick CI (unit tests only)
pytest tests/unit -v --ignore=tests/e2e

# Full CI with synthetic data
pytest tests/ -v --synthetic -m "not slow"

# Slow tests (hyperparameter sweeps)
pytest tests/e2e -v -m slow --timeout=3600

# GPU tests
pytest tests/ -v -m gpu
```

---

## Phase 3: UniTS External Integration

### 3.1 Documentation Requirements

**CRITICAL**: UniTS cannot be reproduced within this framework.

**External Repository:**
- URL: https://github.com/petteriTeikari/UniTS
- Fork of original UniTS with PLR-specific modifications
- Requires separate environment and execution

**Documentation Location:**
```
docs/reproducibility/
├── README.md                    # Main reproducibility guide
├── units_external_execution.md  # Detailed UniTS instructions
└── experiment_completion.md     # Full experiment checklist
```

### 3.2 UniTS Reproducibility Section

```markdown
# UniTS Experiment Reproduction

## IMPORTANT: External Repository Required

The UniTS outlier detection experiments CANNOT be reproduced within this repository.
You must use the separate UniTS repository.

### Steps

1. Clone the UniTS fork:
   ```bash
   git clone https://github.com/petteriTeikari/UniTS.git
   cd UniTS
   ```

2. Follow UniTS-specific setup instructions in that repository

3. Run PLR-specific experiments as documented there

4. Import results back to this repository's MLflow

### Why Separate?

UniTS requires:
- Different Python environment (conflicting dependencies)
- Specific GPU memory configurations
- Custom data preprocessing pipeline
- Separate training loop

### Artifact Integration

After running UniTS externally, import results:
```bash
python scripts/import_external_results.py \
    --source-mlruns /path/to/UniTS/mlruns \
    --target-experiment PLR_OutlierDetection
```
```

---

## Phase 4: CI/CD Integration

### 4.1 GitHub Actions Workflow

```yaml
# .github/workflows/test-pipeline.yml
name: Test Pipeline

on:
  pull_request:
    branches: [main]
  push:
    branches: [main]
  schedule:
    - cron: '0 2 * * 0'  # Weekly full test

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
      - run: uv sync
      - run: uv run pytest tests/unit -v

  synthetic-integration:
    runs-on: ubuntu-latest
    needs: unit-tests
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
      - run: uv sync
      - name: Generate synthetic data
        run: uv run python -m src.synthetic.demo_dataset
      - name: Run integration tests
        run: uv run pytest tests/e2e -v -m synthetic --timeout=1800

  slow-tests:
    runs-on: ubuntu-latest
    needs: unit-tests
    if: github.event_name == 'schedule'
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
      - run: uv sync
      - run: uv run pytest tests/e2e -v -m slow --timeout=7200
```

### 4.2 Local Testing Commands

```makefile
# Makefile additions

test-unit:
	uv run pytest tests/unit -v

test-synthetic:
	uv run python -m src.synthetic.demo_dataset
	uv run pytest tests/e2e -v -m synthetic

test-slow:
	uv run pytest tests/e2e -v -m slow --timeout=7200

test-coverage:
	uv run pytest tests/ --cov=src --cov-report=term-missing

test-all:
	$(MAKE) test-unit
	$(MAKE) test-synthetic
```

---

## Implementation Timeline

### Week 1: Synthetic Data Module

| Day | Task | Deliverable |
|-----|------|-------------|
| 1-2 | PLR curve generation | `src/synthetic/plr_generator.py` |
| 3 | Artifact injection | `src/synthetic/artifact_injection.py` |
| 4 | Database builder | `src/synthetic/database_builder.py` |
| 5 | Privacy validation | `tests/test_synthetic_privacy.py` |

### Week 2: Pipeline Tests

| Day | Task | Deliverable |
|-----|------|-------------|
| 1-2 | Outlier detection tests | `tests/e2e/test_outlier_detection_flow.py` |
| 3 | Imputation tests | `tests/e2e/test_imputation_flow.py` |
| 4 | Classification tests | `tests/e2e/test_classification_flow.py` |
| 5 | Full pipeline test | `tests/e2e/test_full_pipeline_synthetic.py` |

### Week 3: Integration & Documentation

| Day | Task | Deliverable |
|-----|------|-------------|
| 1-2 | Hyperparameter combo verification | `tests/e2e/test_all_combos.py` |
| 3 | GitHub Actions setup | `.github/workflows/test-pipeline.yml` |
| 4 | UniTS documentation | `docs/reproducibility/units_external_execution.md` |
| 5 | Coverage target verification | Coverage report ≥85% |

---

## Success Criteria

| Criterion | Target | Measurement |
|-----------|--------|-------------|
| Unit test coverage | ≥85% | `pytest --cov` |
| E2E test pass rate | 100% | GitHub Actions |
| Synthetic data privacy | 0 leakage | `test_no_pii_leakage()` |
| All registered methods tested | 11 outlier + 8 imputation | Registry validation |
| CI run time | <30 min | GitHub Actions logs |
| Documentation complete | All sections | Manual review |

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Synthetic data too similar to real | Pearson + DTW thresholds in validation |
| CI timeout on slow tests | Weekly schedule for slow tests |
| GPU tests fail in CI | Mark as optional, run locally |
| UniTS integration fragile | Clear separation, manual import |

---

## Appendix A: Synthetic PLR Physiology Model

The PLR (Pupillary Light Reflex) curve has distinct phases:

1. **Baseline** (0-10s): Steady-state pupil diameter in darkness
2. **Red Light Response** (10-25s): Rod-mediated constriction
3. **Recovery 1** (25-35s): Partial dilation after red light
4. **Blue Light Response** (35-50s): Melanopsin-mediated constriction (PIPR)
5. **Recovery 2** (50-66s): Sustained constriction (pathological in glaucoma)

**Key Differences Between Classes:**

| Feature | Control | Glaucoma |
|---------|---------|----------|
| Baseline diameter | 4.5-6.5 mm | 3.5-5.5 mm |
| Red light amplitude | 0.8-1.5 mm | 0.5-1.0 mm |
| Blue light amplitude | 1.2-2.0 mm | 0.8-1.5 mm |
| PIPR (6-second post-illumination) | 15-25% | 8-15% |
| Recovery time constant | 5-10 s | 8-15 s |

---

## Appendix B: Registry Method Lists

### Outlier Detection Methods (11 total)

| Method | Category | Notes |
|--------|----------|-------|
| `pupil-gt` | Ground truth | Human-annotated |
| `MOMENT-gt-finetune` | Foundation model | Fine-tuned on PLR |
| `MOMENT-gt-zeroshot` | Foundation model | Zero-shot |
| `UniTS-gt-finetune` | Foundation model | Fine-tuned on PLR |
| `TimesNet-gt` | Deep learning | Trained on PLR |
| `LOF` | Traditional | Local Outlier Factor |
| `OneClassSVM` | Traditional | One-class SVM |
| `PROPHET` | Traditional | Facebook Prophet |
| `SubPCA` | Traditional | Subspace PCA |
| `ensemble-LOF-MOMENT-...` | Ensemble | Multi-method voting |
| `ensembleThresholded-...` | Ensemble | Thresholded voting |

### Imputation Methods (8 total)

| Method | Category | Notes |
|--------|----------|-------|
| `pupil-gt` | Ground truth | Human-annotated |
| `SAITS` | Deep learning | Self-attention imputation |
| `CSDI` | Deep learning | Conditional score diffusion |
| `TimesNet` | Deep learning | Time-series net |
| `MOMENT-finetune` | Foundation model | Fine-tuned |
| `MOMENT-zeroshot` | Foundation model | Zero-shot |
| `linear` | Traditional | Linear interpolation |
| `ensemble-CSDI-MOMENT-...` | Ensemble | Multi-method |

---

## Appendix C: Reviewer Insights and Plan Refinements

### C.1 Privacy Review Findings (CRITICAL)

**Original thresholds were TOO PERMISSIVE:**

| Metric | Original | Revised | Rationale |
|--------|----------|---------|-----------|
| Pearson correlation | 0.85 | **0.60** | Limit shared variance to <36% |
| Spearman correlation | Not used | **0.55** | Detect monotonic relationships |
| DTW (normalized) | 0.5 | **0.15** | 15% deviation per timepoint |
| Euclidean (normalized) | Not used | **0.30** | 30% average point-wise deviation |

**Additional safeguards added:**
1. File access monitoring during generation (no real DB access)
2. UUID-based subject codes (no collision risk)
3. Distribution divergence checks (5-15% from published stats)
4. Generation metadata with SHA256 hash for integrity

### C.2 Pipeline Coverage Review Findings

**Critical missing components:**
- Ensemble pipeline tests (`src/ensemble/`)
- Featurization flow tests (`src/featurization/`)
- Data import/wrangling tests (`src/data_io/`)
- Stratification logic tests
- MLflow artifact logging validation
- ALL STRATOS metrics (not just AUROC)

**Edge cases to add:**
- 100% outlier subjects (graceful failure)
- Zero outlier subjects
- Truncated timeseries (<1981 points)
- Extreme class imbalance (1:99)
- Invalid method combinations (registry enforcement)

### C.3 CI/CD Feasibility Review Findings (CRITICAL)

**Original plan was NOT FEASIBLE for 30min CI target.**

**Key insights:**
1. Foundation models (MOMENT, TimesNet) require GPU - NOT available on GitHub Actions free tier
2. TabPFN requires 32GB+ RAM - exceeds 7GB limit
3. Full training pipeline takes 2-4 hours, not 30 minutes

**Decision: DEFER MLflow Server for CI**

Since there's no shared MLflow server online:
- Use file-based backend (`mlruns/` directory) locally
- CI should NOT run training - only unit tests and extraction validation
- Full pipeline testing runs **LOCALLY** with user's localhost MLflow

**Revised Tiered Strategy:**

```
┌─────────────────────────────────────────────────────────────┐
│ TIER 1: Fast CI (10-15 min) - Every PR                     │
│ GitHub Actions                                              │
├─────────────────────────────────────────────────────────────┤
│ • Unit tests (no training, no MLflow)                      │
│ • Synthetic data validation                                │
│ • Registry integrity checks                                │
│ • Figure QA tests (no generation)                          │
│ • Extraction code tests (mock data)                        │
│ • GPU: ❌ No    MLflow: ❌ No (mocked)                     │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ TIER 2: Local Testing (manual) - Developer machine         │
├─────────────────────────────────────────────────────────────┤
│ • LOF + linear + CatBoost (CPU-only methods)               │
│ • Full extraction to DuckDB                                │
│ • STRATOS metric computation                               │
│ • Ground truth AUROC validation                            │
│ • GPU: Optional   MLflow: ✅ localhost                     │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ TIER 3: Full Pipeline (2-4 hours) - GPU workstation        │
├─────────────────────────────────────────────────────────────┤
│ • MOMENT-gt-finetune (GPU required)                        │
│ • TimesNet, SAITS, CSDI                                    │
│ • All 440 hyperparameter combos                            │
│ • pminternal instability analysis (R)                      │
│ • GPU: ✅ Yes   MLflow: ✅ localhost                       │
└─────────────────────────────────────────────────────────────┘
```

**Makefile commands (revised):**

```makefile
# Fast CI (what GitHub Actions runs)
test-ci:
	uv run pytest tests/unit -v -m "not slow and not gpu"
	uv run pytest tests/test_registry.py -v

# Local synthetic testing (developer runs manually)
test-synthetic-local:
	uv run python -m src.synthetic.demo_dataset
	uv run pytest tests/e2e -v -m "synthetic and not gpu"

# Full GPU pipeline (requires GPU workstation)
test-full-gpu:
	uv run pytest tests/e2e -v -m "gpu or slow" --timeout=14400
```

---

## Appendix D: Updated Success Criteria

| Criterion | Target | Measurement | Tier |
|-----------|--------|-------------|------|
| Unit test coverage | ≥85% | `pytest --cov` | CI |
| Synthetic data privacy | 0 leakage | Pearson <0.60, DTW >0.15 | CI |
| Registry compliance | 100% | `test_registry.py` | CI |
| CPU methods tested | 5 outlier + 3 imputation | Local | Local |
| GPU methods tested | All registered | Periodic | GPU |
| STRATOS metrics complete | All 10 metrics | Local | Local |
| CI run time | <15 min | GitHub Actions | CI |

---

## Appendix E: DagsHub + arXiv Integration for Academic Reproducibility

### Why DagsHub for Academic Research?

[DagsHub](https://dagshub.com/) is particularly valuable for academic ML research because it combines:

1. **Git repository hosting** (like GitHub)
2. **DVC for data versioning** (large datasets, models)
3. **MLflow experiment tracking** (hosted, no server setup)
4. **arXiv integration** (direct linking to papers)

### arXiv Labs Integration

In March 2023, arXiv announced official integration with DagsHub:

> "DagsHub integration with arXiv enables reproducibility through access to data and code"
> — [arXiv Blog](https://blog.arxiv.org/2023/03/27/new-arxivlabs-integrations-provide-insights-into-the-academic-influence-of-researchers-and-enable-reproducibility-through-access-to-data-and-code/)

**Benefits:**
- Reviewers can inspect experiment logs directly from the arXiv paper page
- Data lineage and model artifacts are version-controlled
- Readers can reproduce experiments with exact configurations
- Complements "Reproducibility 4 Reviewers" (R4R) initiative

### Setup for Foundation PLR

```bash
# 1. Create DagsHub repo (mirrors GitHub)
# Go to dagshub.com and import from GitHub

# 2. Install dagshub package
uv add dagshub

# 3. Authenticate
export DAGSHUB_TOKEN=<your-token>

# 4. Initialize in your script
import dagshub
dagshub.init(
    repo_owner='petteriTeikari',
    repo_name='foundation-PLR',
    mlflow=True
)

# 5. Run experiments - MLflow automatically logs to DagsHub
python src/pipeline_PLR.py
```

### Linking to arXiv Submission

When submitting to arXiv:
1. Include DagsHub repo URL in the paper
2. Enable arXiv Labs integration in DagsHub settings
3. arXiv paper page will show "View on DagsHub" button
4. Reviewers can click to see:
   - All experiment runs with metrics
   - Hyperparameters used
   - Data versions (via DVC)
   - Model artifacts

### Configuration for Publication

```yaml
# configs/publication_run.yaml
defaults:
  - defaults

EXPERIMENT:
  experiment_prefix: ""  # No prefix for publication runs
  debug: False

MLFLOW:
  backend: "dagshub"
  # Add paper metadata to all runs
  tags:
    paper_arxiv_id: "2401.XXXXX"  # Fill after arXiv submission
    paper_title: "Foundation Models for Pupillary Light Reflex Analysis"
    reproducibility: "full"
```

### References

- [DagsHub MLflow Integration Guide](https://dagshub.com/docs/integration_guide/mlflow_tracking/)
- [arXiv Labs + DagsHub Announcement](https://blog.arxiv.org/2023/03/27/new-arxivlabs-integrations-provide-insights-into-the-academic-influence-of-researchers-and-enable-reproducibility-through-access-to-data-and-code/)
- [DagsHub arXiv Integration Blog](https://dagshub.com/blog/dagshub-integration-with-arxiv/)

---

*Document Version: 2.1*
*Created: 2026-02-01*
*Updated: 2026-02-01 (added MLflow backend flexibility, DagsHub + arXiv integration)*
*Status: Reviewed - Ready for Implementation*
