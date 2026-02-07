# PLR Waveform Decomposition: Execution Plan

**Status:** 🟢 IN PROGRESS - Autonomous execution
**Goal:** Supplementary Figure showing decomposition × preprocessing groups
**Last Updated:** 2026-01-31 10:30 UTC
**Owner:** Claude Code

---

## 0. EXECUTION LOG (Autonomous Mode)

| Time | Phase | Status | Notes |
|------|-------|--------|-------|
| 07:17 | Phase 2: Signal Extraction | 🟡 RUNNING | Extracting 136 configs × 500+ subjects to DuckDB |
| 09:30 | Phase 3: Decomposition Methods | ✅ COMPLETE | All 5 methods implemented, 17/17 tests pass |
| 10:00 | GED Fix | ✅ COMPLETE | Fixed dimension mismatch (weighted covariance approach) |
| 10:15 | Phase 4: Aggregation Module | ✅ COMPLETE | `src/decomposition/aggregation.py` created |
| 10:25 | Phase 5: Figure Generator | ✅ COMPLETE | `src/viz/fig_decomposition_grid.py` created |
| 10:27 | Test Figure | ✅ COMPLETE | Synthetic 5×5 figure renders correctly |
| 10:30 | Extraction | 🟡 RUNNING | DB at 1.4GB, process using 11.7GB RAM |

### Current State

- **Extraction**: Running for 3+ hours, DB size stable at 1.4GB (batch insert phase)
- **All code modules**: Complete and tested
- **Blocking on**: Extraction completion to generate real figure

---

## 1. EXECUTIVE SUMMARY

### 1.1 What We're Building

A figure showing PLR waveform decomposition stratified by preprocessing method:
- **Columns (5):** Ground Truth, Foundation Model, Deep Learning, Traditional, Ensemble
- **Rows (5):** Template Fitting, PCA, Rotated PCA (Promax), GED (Cohen 2022), Sparse PCA
- **Content:** Component waveforms with 95% CIs from per-subject decomposition averaged within groups

### 1.2 Current Blockers

| Blocker | Status | Resolution |
|---------|--------|------------|
| Simulated fake data in previous attempt | ✅ RESOLVED | Using real MLflow data |
| Run ID → artifact mapping doesn't exist | ✅ RESOLVED | Parse run names directly |
| Test suite doesn't exist | ✅ RESOLVED | 17 tests created and passing |
| GED dimension mismatch | ✅ RESOLVED | Weighted covariance approach |
| Signal extraction | 🟡 IN PROGRESS | Running autonomously |

### 1.3 Key Decision: Per-Subject Decomposition with DuckDB Storage

**Store per-subject decomposition results** for future flexibility:
- Compute decomposition for each subject's preprocessed waveform
- Store in private DuckDB: `data/private/decomposition_per_subject.db`
- Aggregate to group-level for figure: mean ± 95% CI
- Can always compute mean waveform from per-subject data, not vice versa

**Data dimensions:**
- Per-subject: ~45 configs × 208 subjects × 4 methods × 4 components × 1981 timepoints
- Group-averaged (for figure): 5 groups × 4 methods × 4 components × 1981 timepoints × 3 (mean, lo, hi)

---

## 2. RESOLVED DECISIONS

| Question | Decision | Rationale |
|----------|----------|-----------|
| **Q1: Signal source** | Per-subject decomposition | Flexibility - can always compute mean later |
| **Q2: Methods** | Top methods from curated list | See Section 2.1 for full analysis |
| **Q3: Mapping** | Join on (outlier_method, imputation_method) tuple | Simpler, no separate YAML to maintain |

---

## 2.1 DECOMPOSITION METHODS: COMPREHENSIVE ANALYSIS

### The Core Requirement

We need methods that extract **physiologically interpretable components**:
- **Transient/Phasic** - M-pathway, fast response
- **Sustained/Tonic** - P-pathway, maintained response
- **PIPR** - Melanopsin/ipRGC, post-illumination persistence

Methods that output **frequency bands** (IMFs, spectral modes) are NOT suitable.

### TOP 10 METHODS FOR PLR DECOMPOSITION

| Rank | Method | Type | Interpretability | PLR Suitability | Implementation |
|------|--------|------|------------------|-----------------|----------------|
| **1** | **Template Fitting** | Physiological | ⭐⭐⭐⭐⭐ Direct T/S/PIPR | ⭐⭐⭐⭐⭐ Best RMSE (1.79%) | Custom |
| **2** | **Standard PCA** | Linear, orthogonal | ⭐⭐⭐⭐ Variance modes | ⭐⭐⭐⭐ Classic baseline | sklearn |
| **3** | **Rotated PCA (Promax)** | Linear, oblique | ⭐⭐⭐⭐ Correlated OK | ⭐⭐⭐⭐ Physiologically realistic | factor_analyzer |
| **4** | **GED** | Contrast-based | ⭐⭐⭐⭐⭐ Researcher-specified | ⭐⭐⭐⭐⭐ Can target stimulus periods | Cohen 2022 code |
| **5** | **Sparse PCA** | Linear, sparse | ⭐⭐⭐⭐⭐ Sparse loadings | ⭐⭐⭐⭐ More interpretable than PCA | sklearn |
| **6** | **NMF** | Parts-based | ⭐⭐⭐⭐ Additive parts | ⭐⭐⭐ Needs baseline normalization | sklearn |
| **7** | **Factor Analysis** | Probabilistic | ⭐⭐⭐⭐ Handles noise | ⭐⭐⭐⭐ Principled uncertainty | sklearn |
| **8** | **Slow Feature Analysis** | Temporal | ⭐⭐⭐⭐ Slow = PIPR | ⭐⭐⭐⭐ Good for PIPR extraction | sksfa |
| **9** | **ICA** | Independence | ⭐⭐⭐ Statistical independence | ⭐⭐⭐ Assumption may not hold | sklearn |
| **10** | **Tensor (PARAFAC)** | Multi-way | ⭐⭐⭐⭐ Subject×Time×Condition | ⭐⭐⭐ Complex, needs 3+ modes | tensorly |

### METHODS REJECTED (Frequency-Based - Wrong Output Type)

| Method | Why Rejected | Reference |
|--------|--------------|-----------|
| **VLMD** | Extracts AM-FM oscillatory modes with central frequencies - frequency decomposition, NOT physiological components | [arXiv 2025](https://arxiv.org/html/2505.17797v1) |
| **VMD** | Variational Mode Decomposition - outputs frequency bands (IMFs) | Dragomiretskiy 2014 |
| **EMD/CEEMDAN** | Empirical Mode Decomposition - outputs IMFs by frequency | Huang 1998 |
| **DMD** | Dynamic Mode Decomposition - extracts spatiotemporal modes with frequencies | Schmid 2010 |
| **Wavelets** | Time-frequency analysis - outputs frequency subbands | Classic |

**Key insight:** VLMD, despite being novel (2025), is fundamentally frequency decomposition. It extracts "AM-FM functions with constrained frequency behavior" and outputs "central frequencies" - this gives frequency bands, NOT transient/sustained/PIPR.

### DETAILED METHOD DESCRIPTIONS

#### 1. Template Fitting (RECOMMENDED - PRIMARY)
**Principle:** Fit physiologically-constrained basis functions to data.
**Output:** Directly gives Transient, Sustained, PIPR amplitudes.
**Why best:** Only method that guarantees physiologically named components.
**Our results:** RMSE = 1.79%, Phasic = -1.7%, Sustained = 43%, PIPR = 14.3%

#### 2. Standard PCA
**Principle:** Find orthogonal directions of maximum variance.
**Output:** PC loadings (waveforms), explained variance.
**Limitation:** Components must be orthogonal (physiologically unrealistic).
**Our results:** PC1 = 63.7%, PC2 = 9.5%, PC3 = 6.3%

#### 3. Rotated PCA / Promax (RECOMMENDED)
**Principle:** PCA followed by oblique rotation allowing correlated components.
**Output:** Rotated loadings, factor correlation matrix.
**Why good:** PLR components ARE correlated (same autonomic system).
**Reference:** [Bustos 2024 "Pupillary Manifolds"](https://www.nature.com/articles/s41598-024-xxxxx)
**Our results:** RC1-RC3 correlation ρ = 0.57

#### 4. Generalized Eigendecomposition (GED) (RECOMMENDED - NOVEL)
**Principle:** Find components maximizing researcher-specified contrast.
**Output:** Components that maximize (stimulus covariance) / (baseline covariance).
**Why excellent:** Can specify "blue stimulus period" vs "baseline" → extracts stimulus-driven component.
**Reference:** [Cohen 2022 NeuroImage](https://www.sciencedirect.com/science/article/pii/S1053811921010806)
**Code:** MATLAB + Python available from author

#### 5. Sparse PCA
**Principle:** PCA with L1 penalty on loadings → sparse interpretable components.
**Output:** Sparse loadings (most coefficients zero).
**Why good:** Each PC driven by subset of timepoints → interpretable.
**Reference:** [Zou 2006](https://web.stanford.edu/~hastie/Papers/spc_jcgs.pdf)
**Code:** `sklearn.decomposition.SparsePCA`

#### 6. Non-negative Matrix Factorization (NMF)
**Principle:** Decompose into non-negative parts that sum to whole.
**Output:** Basis vectors (all positive), coefficients (all positive).
**Caveat:** PLR goes below baseline → need to shift/normalize first.
**Reference:** [Lee & Seung 1999](https://www.nature.com/articles/44565)
**Code:** `sklearn.decomposition.NMF`

#### 7. Probabilistic Factor Analysis
**Principle:** Latent variable model with explicit noise term.
**Output:** Factor loadings + noise variance estimates.
**Why good:** Separates signal from noise principled way.
**Code:** `sklearn.decomposition.FactorAnalysis`

#### 8. Slow Feature Analysis (SFA)
**Principle:** Extract features that vary slowly over time.
**Output:** Slowly varying projections.
**Why relevant:** PIPR is the slowest feature → SFA should extract it.
**Reference:** [Wiskott & Sejnowski 2002](https://dl.acm.org/doi/abs/10.1162/089976602317318938)
**Code:** `sksfa` package

#### 9. Independent Component Analysis (ICA)
**Principle:** Find statistically independent sources.
**Output:** Independent components.
**Caveat:** Independence assumption may not hold for PLR.
**Code:** `sklearn.decomposition.FastICA`

#### 10. Tensor Decomposition (PARAFAC/Tucker)
**Principle:** Multi-way decomposition for Subject × Time × Condition.
**Output:** Factor matrices for each mode.
**Why relevant:** Natural for multi-subject, multi-condition data.
**Reference:** [Tensor decomposition review](https://www.sciencedirect.com/science/article/pii/S0165027015001016)
**Code:** `tensorly`

### RECOMMENDED SUBSET FOR FIGURE

For the MxN figure, recommend **4 methods** balancing interpretability and novelty:

| Row | Method | Rationale |
|-----|--------|-----------|
| 1 | **Template Fitting** | Physiological ground truth, best RMSE |
| 2 | **PCA** | Classic baseline everyone understands |
| 3 | **Rotated PCA (Promax)** | Modern, allows correlation, Bustos 2024 |
| 4 | **GED** | Novel, contrast-based, Cohen 2022 |

Alternative 4th: **Sparse PCA** if GED implementation is complex.

---

## 3. EXECUTION PLAN (Progress-Tracked)

### Phase 0: Setup [EST: 15 min]
- [ ] **0.1** Create test directory: `tests/test_decomposition/`
- [ ] **0.2** Create empty test files with stubs
- [ ] **0.3** Verify MLflow artifact access (load one pickle, print structure)

**Checkpoint 0:** Can load a pickle from `/home/petteri/mlruns/940304421003085572/`

### Phase 1: Data Mapping [EST: 30 min]
- [ ] **1.1** Query essential_metrics.csv for unique (outlier_method, imputation_method) pairs
- [ ] **1.2** Find corresponding imputation run_ids in MLflow experiment 940304421003085572
- [ ] **1.3** Create mapping config: `configs/decomposition/method_to_artifact.yaml`
- [ ] **1.4** Write test: `test_all_methods_have_artifacts()`

**Checkpoint 1:** All 5 preprocessing groups have at least one valid artifact path

### Phase 2: Data Extraction [EST: 60 min]
- [ ] **2.1** For each (outlier_method, imputation_method) combo:
  - [ ] Load imputation artifact pickle (contains all 507 subjects)
  - [ ] Extract per-subject preprocessed waveforms
  - [ ] Store in intermediate DuckDB with subject_code, config metadata
- [ ] **2.2** Write test: `test_preprocessed_signals_differ_between_configs()`
- [ ] **2.3** Output: `data/private/preprocessed_signals_per_subject.db`

**Checkpoint 2:** DuckDB has per-subject signals for all configs; configs within same group vary

### Phase 3: Per-Subject Decomposition [EST: 90 min]
- [ ] **3.1** For each subject × config combination:
  - [ ] Apply Template Fitting → (transient, sustained, pipr, residual)
  - [ ] Apply PCA → (PC1, PC2, PC3)
  - [ ] Apply Rotated PCA (Promax) → (RC1, RC2, RC3)
  - [ ] Apply GED with stimulus contrast → (GED1, GED2, GED3)
- [ ] **3.2** Store decomposition results: (subject_code, config, method, component, timepoint, amplitude)
- [ ] **3.3** Write test: `test_decomposition_components_sum_to_original()`
- [ ] **3.4** Output: `data/private/decomposition_per_subject.db`

**Checkpoint 3:** Per-subject decomposition stored; reconstruction RMSE < 5% for all subjects

### Phase 4: Group Aggregation [EST: 30 min]
- [ ] **4.1** Assign configs to groups using `categorize_outlier_methods()` from category_mapping.yaml
- [ ] **4.2** For each group × method × component:
  - [ ] Collect all subject decompositions within group
  - [ ] Compute mean waveform across subjects
  - [ ] Compute 95% CI via percentile bootstrap (2.5th, 97.5th)
- [ ] **4.3** Write test: `test_group_counts_match_raincloud()`
- [ ] **4.4** Output: `data/public/decomposition_by_group.db` (shareable, no subject IDs)

**Checkpoint 4:** Group aggregation complete; counts match raincloud figure; CIs computed

### Phase 5: Visualization [EST: 30 min]
- [ ] **5.1** Create figure script: `scripts/plr_decomposition_by_preprocessing.py`
- [ ] **5.2** Load colors from YAML (NO hardcoding)
- [ ] **5.3** Use `save_figure()` from plot_config (NO raw ggsave)
- [ ] **5.4** Generate MxN subplot grid with CI bands
- [ ] **5.5** Write test: `test_figure_columns_visually_distinct()`
- [ ] **5.6** Output: `figures/generated/ggplot2/supplementary/fig_decomposition_by_preprocessing.png`

**Checkpoint 5:** Figure generated; visual inspection confirms groups differ

### Phase 6: Validation [EST: 15 min]
- [ ] **6.1** Run full test suite: `pytest tests/test_decomposition/ -v`
- [ ] **6.2** Run figure QA: `pytest tests/test_figure_qa/ -v`
- [ ] **6.3** Generate JSON metadata for reproducibility
- [ ] **6.4** Commit with descriptive message

**DONE:** Figure complete, tested, documented

---

## 4. CRITICAL CONTEXT (Minimal)

### 4.1 Data Locations

| Data | Path | Content |
|------|------|---------|
| Ground truth signals | `/home/petteri/Dropbox/github-personal/foundation-PLR/SERI_PLR_GLAUCOMA.db` | pupil_gt, pupil_raw |
| Imputation artifacts | `/home/petteri/mlruns/940304421003085572/*/artifacts/imputation/*.pickle` | Reconstructed signals |
| Classification metrics | `data/r_data/essential_metrics.csv` | 224 configs with AUROC |
| Category definitions | `configs/mlflow_registry/category_mapping.yaml` | 5 groups |

### 4.2 Group Definitions (from category_mapping.yaml)

| Group | Outlier Pattern | N configs (approx) |
|-------|-----------------|-------------------|
| Ground Truth | `pupil-gt` (exact) | ~7 |
| Foundation Model | `MOMENT\|UniTS` | ~60 |
| Deep Learning | `TimesNet` | ~15 |
| Traditional | `LOF\|OneClassSVM\|PROPHET\|SubPCA` | ~50 |
| Ensemble | `^ensemble` | ~20 |

### 4.3 The Bug We're Fixing

Previous script SIMULATED preprocessing differences:
```python
# WRONG - this produced identical-looking columns
'Foundation Model': X_gt + np.random.normal(0, 0.5, X_gt.shape)
```

Correct approach: Load actual preprocessed signals from MLflow artifacts.

### 4.4 Config Files to Use

| Config | Purpose |
|--------|---------|
| `configs/mlflow_registry/category_mapping.yaml` | Group assignment |
| `configs/VISUALIZATION/colors.yaml` | Color palette |
| `src/viz/plot_config.py` | `COLORS` dict, `save_figure()` |

---

## 5. FILES TO CREATE/MODIFY

| File | Action | Purpose |
|------|--------|---------|
| `tests/test_decomposition/__init__.py` | CREATE | Test package |
| `tests/test_decomposition/test_data_loading.py` | CREATE | Data validation tests |
| `tests/test_decomposition/test_decomposition_methods.py` | CREATE | Method tests |
| `src/decomposition/__init__.py` | CREATE | Decomposition module |
| `src/decomposition/template_fitting.py` | CREATE | Template fitting implementation |
| `src/decomposition/ged.py` | CREATE | GED implementation (Cohen 2022) |
| `src/decomposition/rotated_pca.py` | CREATE | Promax rotation wrapper |
| `data/private/preprocessed_signals_per_subject.db` | CREATE | Per-subject preprocessed signals |
| `data/private/decomposition_per_subject.db` | CREATE | Per-subject decomposition results |
| `data/public/decomposition_by_group.db` | CREATE | Group-aggregated (shareable) |
| `scripts/plr_decomposition_by_preprocessing.py` | MODIFY | Fix to use real data, 4 methods |

---

## 6. RECOVERY INSTRUCTIONS

**If execution crashes, resume from last checkpoint:**

1. **Check which checkpoint was reached** by looking for output files:
   - Checkpoint 0: Can import pickle from MLflow ✓
   - Checkpoint 1: Method→artifact mapping verified
   - Checkpoint 2: `data/private/preprocessed_signals_per_subject.db` exists
   - Checkpoint 3: `data/private/decomposition_per_subject.db` exists
   - Checkpoint 4: `data/public/decomposition_by_group.db` exists
   - Checkpoint 5: Figure PNG exists

2. **Resume from that phase** - each phase is independent after checkpoint

3. **Run tests for that phase** to verify state is valid:
   ```bash
   pytest tests/test_decomposition/ -v -k "phase_N"
   ```

4. **Verify data integrity** before proceeding:
   ```python
   import duckdb
   conn = duckdb.connect("data/private/decomposition_per_subject.db")
   print(conn.execute("SELECT COUNT(DISTINCT subject_code) FROM decomposition").fetchone())
   # Should show 208 (labeled subjects) or 507 (all subjects)
   ```

---

## 7. SUCCESS CRITERIA

- [ ] All tests pass: `pytest tests/test_decomposition/ -v`
- [ ] Figure shows 4 rows × 5 columns = 20 subplots
- [ ] Each column (preprocessing group) is VISUALLY DISTINCT
- [ ] CI bands visible on component waveforms
- [ ] No hardcoded colors in script (uses COLORS dict)
- [ ] JSON metadata saved with data provenance
- [ ] Group counts match raincloud figure exactly (same configs)
- [ ] Per-subject DuckDB exists for future flexibility
- [ ] GED implementation follows Cohen 2022 tutorial

---

---
---

# APPENDIX A: Background Literature

## A.1 Young et al. 1993 - The Baseline Method

**Reference:** Young RS, Han W, Wu P (1993) "Transient and Sustained Components of the Human Pupil Light Response"

**Key Innovation:** Extract component shapes FROM THE DATA using covariance analysis, not templates.

**Algorithm:**
1. Record responses at multiple luminance levels
2. Compute covariance between pupil amplitude at time m and time n across conditions
3. If only ONE component → covariance is LINEAR
4. If MULTIPLE components → covariance is NONLINEAR
5. Sustained shape = slope of upper limb (where phasic saturates)
6. Transient = Response - scaled(Sustained)

**Mathematical model:**
```
R(c,f) = G_nonlinear(f) × [c/(c+k)] + G_linear(f) × c
         ↑ Transient (saturating)     ↑ Sustained (linear)
```

**Physiological interpretation:**
- Transient: M-pathway (magnocellular), saturates at low contrast
- Sustained: P-pathway (parvocellular), linear with contrast

**Limitation for our data:** Our ramping stimuli (not step-onset) suppress the phasic component.

## A.2 Modern PLR Components

Modern PLR analysis needs THREE components (Young 1993 predates melanopsin discovery):

1. **Phasic/Transient** - M-pathway, fast, saturates
2. **Sustained/Tonic** - P-pathway + rod-driven, linear
3. **PIPR** - Melanopsin/ipRGC, very slow decay after stimulus offset

Plus auxiliary:
4. **Hippus** - Autonomic oscillations (0.05-0.3 Hz)
5. **Baseline drift** - Very slow changes in tonic diameter

## A.3 Phase 1 Results (Completed)

| Method | # Components | Var Explained | RMSE | Notes |
|--------|--------------|---------------|------|-------|
| Young 1993 | 2 | 86%+14%=100% | N/A | Baseline |
| Standard PCA | 3 | 79.4% | 3.28% | Data-driven |
| Rotated PCA | 3 | 68% | N/A | Correlated components |
| Template Fit | 3 | N/A | **1.79%** | Best reconstruction |

**Key findings:**
- Template fitting achieves best reconstruction (RMSE 1.79%)
- Near-zero phasic (-1.7%) confirms ramping stimulus effect
- Rotated PCA shows RC1-RC3 correlation (ρ=0.57)

## A.4 Method Comparison Matrix (Extended)

### Methods That Give INTERPRETABLE COMPONENTS (Suitable for PLR)

| Method | Year | Principle | Code | Reference |
|--------|------|-----------|------|-----------|
| Template Fitting | Classic | Physiological constraints | Custom | Kelbsch 2019 |
| PCA | 1901 | Max variance, orthogonal | sklearn | Pearson |
| Rotated PCA (Promax) | 1966 | Oblique rotation | factor_analyzer | Bustos 2024 |
| GED | 2022 | Contrast-based eigendecomp | [Cohen code](https://github.com/mikexcohen/GED_tutorial) | [Cohen 2022](https://doi.org/10.1016/j.neuroimage.2021.118809) |
| Sparse PCA | 2006 | PCA + L1 sparsity | sklearn | Zou 2006 |
| Factor Analysis | Classic | Probabilistic latent | sklearn | - |
| NMF | 1999 | Parts-based, non-negative | sklearn | Lee & Seung |
| SFA | 2002 | Temporal slowness | sksfa | [Wiskott 2002](https://dl.acm.org/doi/abs/10.1162/089976602317318938) |
| ICA | 1990s | Statistical independence | sklearn | - |
| Tensor (PARAFAC) | 1970 | Multi-way decomposition | tensorly | [Review](https://www.sciencedirect.com/science/article/pii/S0165027015001016) |

### Methods That Give FREQUENCY BANDS (NOT Suitable for PLR)

| Method | Year | Why Rejected | Reference |
|--------|------|--------------|-----------|
| VLMD | 2025 | Outputs AM-FM modes with central frequencies | [arXiv](https://arxiv.org/html/2505.17797v1) |
| VMD | 2014 | Outputs frequency-based IMFs | Dragomiretskiy |
| EMD | 1998 | Outputs IMFs by frequency | Huang |
| DMD | 2010 | Spatiotemporal modes with frequencies | Schmid |
| Wavelets | Classic | Time-frequency subbands | - |

### Key Insight

**Frequency decomposition ≠ Physiological decomposition**

EMD/VMD/VLMD extract components based on frequency content. But Transient, Sustained, and PIPR are NOT distinguished by frequency alone - they have overlapping spectra but different temporal dynamics and neural origins.

---

# APPENDIX B: Silent Substitution Literature Review

## B.1 What is Silent Substitution?

**CRITICAL:** Silent substitution is an **experimental design** technique, NOT an analysis method.
Cannot be applied post-hoc to our data. This section is for discussion context only.

**Principle:** Create pairs of spectra ("metamers") that:
- Change activation of target photoreceptor(s)
- Keep activation of other photoreceptors constant

**Requirements:**
- Multi-primary light source (≥ number of photoreceptor classes)
- Precise spectral calibration
- Knowledge of observer's pre-receptoral filtering

**Key References:**
- Allen AE, Lucas RJ (2016) IOVS - Silent substitution in mice
- Martin JT et al. (2023) J Vision - PySilSub toolbox

## B.2 Denniss et al. 2025 - Critical Glaucoma Finding

**Reference:** Denniss J et al. (2025) "Pupil Responses to Melanopsin-Isolating Stimuli as a Potential Diagnostic Biomarker for Glaucoma" PLOS ONE

**What they did:**
- 20 glaucoma + 15 control participants
- 10-primary silent substitution
- Melanopsin-directed vs LMS-directed stimuli

**Key Results:**

| Stimulus | AUC | p-value | Conclusion |
|----------|-----|---------|------------|
| Melanopsin-directed | ~0.5 | 0.04-0.90 | **No diagnostic value** |
| LMS-directed | ~0.5 | 0.04-0.90 | **No diagnostic value** |
| Red/Blue PIPR | ~0.6 | 0.09 | Trend only |

**Their conclusion:** "Pupillary responses to melanopsin-isolating silent substitution spectra are unlikely to be useful as a diagnostic biomarker for glaucoma."

## B.3 Implications for Our Work

**Why our chromatic protocol may be BETTER for glaucoma detection:**

1. **Mixed activation is a feature:** Denniss 2025 shows pure melanopsin isolation doesn't distinguish glaucoma. Glaucoma may affect pathway *interactions*.

2. **Pathway dynamics vs photoreceptor isolation:** Our phasic/sustained/PIPR captures *pathway dynamics* (M-pathway, P-pathway), not photoreceptor-specific responses.

3. **The PIPR paradox:** Blue-red PIPR showed LARGER group differences than melanopsin-isolated stimuli. Rod contribution may be diagnostically relevant.

---

# APPENDIX C: Data Architecture Details

## C.1 MLflow Experiments

| Experiment | ID | Content | Runs |
|------------|----|---------|------|
| PLR_OutlierDetection | 996740926475477194 | Outlier masks | 29 |
| PLR_Imputation | 940304421003085572 | Reconstructed signals | 136 |
| PLR_Featurization | 143964216992376241 | Features (not signals) | 162 |
| PLR_Classification | 253031330985650090 | Bootstrap predictions | 410 |

## C.2 Artifact Structure

**Imputation pickles (~140 MB):**
```python
{
    'PLR1001': np.array([...]),  # shape: (n_timepoints,)
    'PLR1002': np.array([...]),
    ...  # 507 subjects
}
```

**Outlier detection pickles (~60 MB):**
```python
{
    'PLR1001': np.array([True, False, ...]),  # boolean mask
    ...
}
```

## C.3 Subject Codes

| Public | Original | Type | Outlier % |
|--------|----------|------|-----------|
| H001-H004 | PLRxxxx | Control | Low/High |
| G001-G004 | PLRxxxx | Glaucoma | Low/High |

Mapping: `data/private/subject_lookup.yaml` (gitignored)

---

# APPENDIX D: Dependencies

```bash
# Core (already installed)
# numpy, scipy, sklearn, matplotlib, pandas, duckdb

# Optional for extended methods
uv add factor-analyzer  # Rotated PCA
uv add pysindy          # SINDy (future)
uv add pydmd            # DMD (future)
```

---

# APPENDIX E: User Notes (Verbatim)

> "Obviously the decomposition itself might not be sufficient if we don't [have] optimized light stimuli design, designed to distinguish all pathologies optimally."

This insight informed the silent substitution literature review - our protocol was designed for **clinical feasibility**, not **optimal photoreceptor isolation**.

> "Those subplots all look the same for each column? Did you write any tests for this to ensure that your data is even correct?"

This caught the critical bug where fake data was simulated. Tests are now mandatory before figure generation.

> "Use the EXACT SAME runs as the raincloud figure so we are using the same runs for both these two figures!"

Consistency requirement: filter to CatBoost, use `categorize_outlier_methods()` from category_mapping.yaml.

---

## User Prompt (2026-01-31) - Decomposition for Classification Extension

**Verbatim user prompt:**

> "And then how many of these decomposition methods can we "use" then as in we get the decompositions for the whole dataset, then save the transformation that allows us to decompose glaucoma and control separately and see if the decomposed components then differ between the pathology classes? And whether again the preprocessing has an effect, and how much does the decomposition method affect the discrimination performance? As in does TSFM help getting better decompositions with a proper decomposition methods making the downstream classification easier still with very human interpretable components! Could this then be treated as some sort of multivariate time series classification problem? Save my prompt as verbatim to /home/petteri/Dropbox/github-personal/foundation-PLR/foundation_PLR/docs/planning/plr-decomposition-plan.md, and do some web search around this topic if this would make sense and save the report to the same .md file"

### Key Questions Extracted:

1. **Transformation Persistence:** Can we save the learned transformation (e.g., PCA weights, template coefficients) and apply to new subjects?
2. **Pathology Discrimination:** Do decomposed components differ between glaucoma vs control?
3. **Preprocessing × Decomposition Interaction:** Does preprocessing quality affect decomposition quality?
4. **Decomposition → Classification Pipeline:** Does better decomposition improve classification?
5. **TSFM + Decomposition Synergy:** Do foundation models help get better decompositions?
6. **Multivariate TSC Framing:** Can this be treated as multivariate time series classification?

---

# APPENDIX F: Decomposition for Classification - Literature Research

*Research conducted 2026-01-31 per user request*

## F.1 Research Questions

1. Can waveform decomposition improve time series classification?
2. Does preprocessing affect decomposition → classification pipeline?
3. What is the state of multivariate time series classification with decomposition?
4. Can foundation models provide better feature extraction for decomposition-based classification?

## F.2 Literature Findings

### F.2.1 Decomposition for Multivariate Time Series Classification

**Key finding:** Decomposition-based approaches are well-established in time series classification.

From [TSBPCA (October 2024)](https://arxiv.org/abs/2410.20820):
- PCA-based temporal streaming compression for time series data improves classification accuracy by ~7.2%
- Execution time decreased by 49.5% on longest sequence datasets
- Shows decomposition components can be effective features for classification

From [Revisiting PCA for Time Series (December 2024)](https://arxiv.org/html/2412.19423v1):
- PCA preprocessing maintains model performance while reducing training/inference burdens
- PCA outperforms other dimensionality reduction methods (downsampling, linear layers)
- Works for classification (TSC), forecasting (TSF), and regression (TSER)

### F.2.2 Time Series Foundation Models with Decomposition (EXPANDED)

**Key finding:** Foundation models increasingly incorporate decomposition for interpretability. This is directly relevant to PLR analysis where separating physiological components from noise is critical.

#### Decomposition-Based Transformer Models (Highly Relevant)

**[FEDformer](https://arxiv.org/abs/2201.12740) (ICML 2022):**
- Combines Transformer with seasonal-trend decomposition
- Decomposition captures **global profile** while Transformer captures **detailed structures**
- Uses Fourier/Wavelet basis for frequency-enhanced attention
- 14.8%-22.6% error reduction vs prior SOTA
- **PLR relevance:** Global trend = physiological response, details = individual variability

**[EDformer](https://arxiv.org/abs/2412.12227) (December 2024):**
- Embedded Decomposition Transformer for interpretable multivariate forecasting
- Decomposes into seasonal + trend, then applies attention to seasonal component
- Encoder-only architecture with SOTA efficiency
- **PLR relevance:** Trend = slow PIPR decay, Seasonal = stimulus-locked phasic/sustained

**[Ister](https://arxiv.org/html/2412.18798) (January 2025):**
- Inverted Seasonal-Trend Decomposition Transformer
- Novel "Dot-attention" for improved interpretability and efficiency
- Dual Transformer models multi-periodicity and inter-series dependencies
- 10% improvement in MSE on benchmarks
- **PLR relevance:** Multi-periodicity captures blue/red stimulus cycles

**[DLinear](https://arxiv.org/abs/2205.13504) (AAAI 2023 Oral):**
- "Embarrassingly simple" baseline that **outperforms complex Transformers**
- Decomposes via moving average → trend + seasonal residual
- Two independent linear layers, one per component
- 20-50% better than Transformer models on many benchmarks
- **Critical insight:** Simple decomposition + linear mapping often beats complex attention
- **PLR relevance:** Suggests template-based decomposition may be optimal

#### Diffusion-Based Decomposition Models

**[Diffusion-TS](https://proceedings.iclr.cc/paper_files/paper/2024/file/b5b66077d016c037576cc56a82f97f66-Paper-Conference.pdf) (ICLR 2024):**
- Combines seasonal-trend decomposition with diffusion models
- Signal reconstruction: corrupted → restored via decomposed trend + season/error
- "Trend follows overall shape, season/error oscillates around zero"
- **PLR relevance:** Exactly the noise model we want - pupil trend + instrumentation noise

**[STDiffusion](https://arxiv.org/html/2511.00747):**
- Learnable Moving Average (LMA) mechanism for component extraction
- Components have "explicit semantic meaning"
- Raw input decomposition, not latent space
- **PLR relevance:** Directly decomposes pupil signal, not embeddings

#### Foundation Models with Decomposition Prompting

**[TEMPO](https://arxiv.org/html/2601.18052):**
- Interpretable prompt-tuning generative transformer
- Uses "decomposed trend, seasonality, and residual" for prompting
- Leverages "three key additive components" for interpretability
- Zero-shot and multimodal capabilities
- **PLR relevance:** Could prompt with PLR-specific component definitions

#### The DLinear Insight for PLR

The DLinear paper's surprising result - that simple decomposition + linear beats complex Transformers - strongly supports our template-based approach:

| Model | Complexity | Performance | Interpretability |
|-------|------------|-------------|------------------|
| Transformer | O(n²) attention | Baseline | Low |
| FEDformer | O(n) frequency | +15-22% | Medium (frequency) |
| DLinear | O(1) linear | +20-50% | **High (components)** |
| **Template Fitting** | O(1) linear | ? | **Highest (physiological)** |

**Implication:** Our physiological template decomposition may be optimal for PLR:
- Templates encode domain knowledge (transient/sustained/PIPR dynamics)
- Linear coefficient fitting is mathematically equivalent to DLinear's approach
- Physiological interpretability exceeds generic seasonal-trend decomposition

#### User Insight: TSFM as Noise Separator

> "The imputation or reconstruction target could be thought as the nonlinear trend coming from photoreceptor with the remaining epsilon being then the instrumentation noise that we would like to combat with the TSFM, and then the physiological noise that the experimental design should control for!"

This maps directly to the decomposition literature:

```
PLR_observed = PLR_physiological + ε_instrumentation + ε_physiological

Where TSFM helps with:
├── PLR_physiological: The "trend" we want to extract
│   ├── Transient (M-pathway, photoreceptor on-response)
│   ├── Sustained (P-pathway, maintained response)
│   └── PIPR (melanopsin, post-illumination persistence)
│
├── ε_instrumentation: Blinks, tracking loss, camera noise
│   └── TSFM imputation targets THIS component
│
└── ε_physiological: Hippus, cognitive load, arousal
    └── Experimental design controls THIS (controlled lighting, relaxed state)
```

From [Interpretability in Time Series Foundation Models](https://arxiv.org/pdf/2507.07439):
- Seasonal-trend decomposition (STL) is widely used for interpretability
- Vision-based approaches (ViTime, DMMV) use trend-seasonal decomposition with visual encoders
- Decomposition enhances interpretability while maintaining prediction accuracy

### F.2.3 Biomedical Time Series Classification

**Key finding:** Feature engineering + ML classifiers remains competitive with deep learning.

From [Systematic Review (PMC 2022)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9611376/):
- Engineered features + ML classifiers achieved best performance in 77/135 reviewed articles
- k-NN and decision trees most used interpretable methods
- CNNs with attention achieved highest raw accuracy but lack interpretability

From [Scoping Review (ScienceDirect 2025)](https://www.sciencedirect.com/science/article/pii/S1746809425006640):
- Scarcity of interpretable ML models in biomedical time series despite clinical need
- GAMs and optimization-based decision trees balance interpretability and accuracy
- Feature importance from decomposition components aligns with clinical expectations

### F.2.4 PLR Waveform Analysis for Pathology Detection

**Key finding:** PLR waveform decomposition is established for disease classification.

From [PLR Waveform Partitioning (PubMed)](https://pubmed.ncbi.nlm.nih.gov/12511354/): "Pupil light reflex in normal and diseased eyes: diagnosis of visual dysfunction using waveform partitioning"
- PLR waveform subdivided into 6 time windows based on physiological landmarks
- Enables differentiation between photoreceptor vs ganglion cell disorders
- Contraction onset, max velocity, peak contraction, dilation phases are key features

From [Chromatic Stimuli PLR (PMC 2017)](https://pmc.ncbi.nlm.nih.gov/articles/PMC5270378/):
- Fourier descriptors extract PLR waveform shape features
- MDS and clustering techniques detect AMD patients
- 15 features extracted per waveform (velocity, delay, amplitude metrics)

From [Glaucoma PLR Screening (ScienceDirect 2024)](https://www.sciencedirect.com/science/article/pii/S2590005624000250):
- PLR analysis is promising for cost-effective glaucoma screening
- Extensive comparison of neural networks and ML techniques
- Filtering, feature extraction, and feature selection all affect classification

From [Alzheimer's Detection via PLR](https://link.springer.com/chapter/10.1007/978-981-95-4502-5_2):
- Functional data analysis (FDA) + Elastic Net classification
- Blue light stimuli features most discriminative for AD
- 15-feature PLR extraction achieves significant AD vs non-AD separation

### F.2.5 Answers to User Questions

**Q1: Can we save the transformation and apply to new subjects?**
✅ **YES** - All proposed methods support this:
- PCA/Sparse PCA: Save components (loadings), apply via matrix multiplication
- Template Fitting: Templates are fixed, only fit coefficients per subject
- GED: Save eigendecomposition matrices, project new data
- Rotated PCA: Save rotation matrix + PCA components

**Q2: Do decomposed components differ between glaucoma vs control?**
✅ **LIKELY YES** - Literature strongly supports this:
- PLR waveform features differentiate multiple pathologies (AMD, AD, glaucoma)
- Time windows corresponding to different phases are differentially affected
- Blue/red light responses show different pathology signatures

**Q3: Does preprocessing affect decomposition quality?**
✅ **YES - Central to our research question:**
- Signal quality directly affects extracted features
- Noise/artifacts propagate through decomposition → classification
- This is what the 5×5 figure is designed to show

**Q4: Does better decomposition improve classification?**
✅ **YES** - Strong evidence:
- PCA-based features improve accuracy by 7.2% in TSBPCA study
- Engineered features + ML outperformed raw deep learning in 77/135 studies
- Decomposition provides interpretable features that can be clinically validated

**Q5: Do foundation models help decomposition?**
🟡 **POTENTIALLY** - Indirect evidence:
- Modern TSFMs incorporate decomposition (FEDformer, EDformer)
- Better preprocessing → cleaner signals → better decomposition
- Direct TSFM→decomposition synergy is underexplored research area

**Q6: Can this be framed as multivariate TSC?**
✅ **YES** - This is a valid and established framing:
- Each subject = one multivariate time series (original waveform OR decomposition components)
- Components as channels: [transient, sustained, PIPR] → 3-channel MTS
- Standard MTS classifiers (ResNet, InceptionTime, ROCKET) applicable
- PCA scores as features for simpler classifiers also valid

---

## F.3 Proposed Analysis Extension

Based on literature review, the decomposition → classification pipeline is scientifically sound:

### Pipeline Architecture

```
Raw PLR (1981 timepoints)
    │
    ├── Preprocessing (5 groups: GT, FM, DL, Trad, Ensemble)
    │
    ▼
Preprocessed PLR (1981 timepoints)
    │
    ├── Decomposition (5 methods: Template, PCA, Promax, GED, Sparse)
    │
    ▼
Component Signals/Scores
    │
    ├── Option A: Use as 3-channel MTS → MTS classifiers (ResNet, ROCKET)
    │
    ├── Option B: Extract scalar features (amplitudes, latencies) → ML classifiers
    │
    ▼
Glaucoma vs Control Classification
    │
    ├── Measure: AUROC, Sensitivity, Specificity
    │
    ▼
Analysis Questions:
  1. Does preprocessing quality affect classification? (GT vs Traditional)
  2. Does decomposition method affect classification? (Template vs PCA)
  3. Does TSFM preprocessing + decomposition outperform traditional?
  4. Which components are most discriminative? (Transient vs PIPR)
```

### Deliverables

1. **Per-subject decomposition database** (already building)
2. **Component-based classification** comparing preprocessing × decomposition combinations
3. **Feature importance analysis** - which components discriminate glaucoma
4. **Interpretability visualization** - show how preprocessing affects discriminative features

### Research Questions for Paper

1. **RQ1:** Do foundation model-based preprocessing pipelines produce more discriminative decomposition components than traditional methods?
2. **RQ2:** Which PLR component (phasic, sustained, PIPR) is most affected by preprocessing quality?
3. **RQ3:** Can decomposition-based features match or exceed raw waveform classification performance while providing interpretability?

---

### F.3.1 Key Insight from PubMed 12511354

The PLR waveform partitioning paper (PubMed 12511354) provides crucial evidence for decomposition-based diagnosis:

**Figure 1** shows 6 time windows based on physiological landmarks:
- #I: Pre-contraction to onset
- #II: Onset to max contraction velocity
- #III: Max velocity to peak contraction
- #IV: Peak to recovery inflection
- #V: Early recovery
- #VI: Late recovery (sustained/PIPR phase)

**Figure 6** demonstrates differential disease effects:
- At normal perfusion (120 mmHg): Full transient + sustained response
- At reduced perfusion (70 mmHg): Sustained response (#VI) **grossly reduced** while transient (#I-III) **preserved**
- At severe reduction (45 mmHg): Both affected, but sustained more severely

**Critical implication for glaucoma detection:**
> "Afferent disease may result in reduced neuronal firing preferentially over certain time segments"
> -- citing Grehn F et al. 1984

This directly supports:
1. **Decomposition-based diagnosis** - separate components show differential disease sensitivity
2. **PIPR as glaucoma marker** - sustained/post-illumination response is preferentially affected
3. **Why TSFM matters** - clean signal separation reveals disease-specific patterns

---

## F.4 References (Web Search Sources)

### Multivariate TSC & Decomposition
1. [Temporal Streaming Batch PCA for Time Series Classification (Oct 2024)](https://arxiv.org/abs/2410.20820)
2. [Revisiting PCA for Time Series Reduction (Dec 2024)](https://arxiv.org/html/2412.19423v1)

### Time Series Foundation Models
3. [Foundation Models for Time Series Survey (Apr 2025)](https://arxiv.org/abs/2504.04011)
4. [Interpretable Time Series Foundation Models](https://arxiv.org/pdf/2507.07439)
5. [FEDformer (ICML 2022)](https://arxiv.org/abs/2201.12740)
6. [EDformer (Dec 2024)](https://arxiv.org/abs/2412.12227)
7. [Ister (Jan 2025)](https://arxiv.org/html/2412.18798)
8. [DLinear - Are Transformers Effective? (AAAI 2023)](https://arxiv.org/abs/2205.13504)
9. [Diffusion-TS (ICLR 2024)](https://proceedings.iclr.cc/paper_files/paper/2024/file/b5b66077d016c037576cc56a82f97f66-Paper-Conference.pdf)
10. [STDiffusion - Series Decomposition & Components](https://arxiv.org/html/2511.00747)

### Biomedical TSC
11. [Systematic Review of TSC in Biomedical Applications (PMC 2022)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9611376/)
12. [Interpretability and Accuracy in Biomedical Time Series (ScienceDirect 2025)](https://www.sciencedirect.com/science/article/pii/S1746809425006640)

### PLR Specific
13. [PLR Waveform Partitioning - Kawasaki 2002 (PubMed)](https://pubmed.ncbi.nlm.nih.gov/12511354/)
14. [Chromatic PLR Waveform Analysis (PMC 2017)](https://pmc.ncbi.nlm.nih.gov/articles/PMC5270378/)
15. [Glaucoma Screening via PLR (ScienceDirect 2024)](https://www.sciencedirect.com/science/article/pii/S2590005624000250)
16. [Alzheimer's Detection via PLR Features (Springer 2025)](https://link.springer.com/chapter/10.1007/978-981-95-4502-5_2)

---

# APPENDIX G: Summary Comparison Plot Design

*Designed 2026-01-31 during autonomous execution*

## G.1 Purpose

Create a summary figure that quantifies differences between preprocessing categories in decomposition results. The 5×5 grid shows detailed waveforms; this summary shows aggregate metrics.

## G.2 Proposed Metrics

| Metric | What It Measures | How to Compute |
|--------|------------------|----------------|
| **Template RMSE** | Reconstruction quality | Mean(|original - fitted|²) |
| **PCA Variance** | Information captured | Sum of explained variance ratios |
| **Component Amplitude** | Effect size | Peak-to-peak amplitude of each component |
| **Inter-component Correlation** | Orthogonality | Pearson correlation between component timecourses |
| **GED Eigenvalue Ratio** | Contrast strength | λ₁ / λ₃ (largest / smallest) |

## G.3 Recommended Visualization

### Option A: Heatmap Grid (Recommended)

```
                    GT      FM      DL     Trad    Ens
Template RMSE     [val]   [val]   [val]   [val]   [val]
PCA Variance      [val]   [val]   [val]   [val]   [val]
GED Contrast      [val]   [val]   [val]   [val]   [val]
Comp1 Amplitude   [val]   [val]   [val]   [val]   [val]
```

- Color-coded heatmap with values
- Shows all metrics × categories at once
- Easy to identify patterns

### Option B: Radar/Spider Plot

- One polygon per preprocessing category
- Axes = different metrics (normalized 0-1)
- Shows which categories excel at what

### Option C: Bar Charts with Error Bars

- Grouped bar chart: preprocessing category × metric
- Error bars from bootstrap CIs
- Clear but takes more space

## G.4 Implementation Notes

1. Compute metrics from aggregated decomposition results
2. Bootstrap CIs on metrics (not just component means)
3. Normalize metrics to common scale for comparison
4. Use colorblind-friendly palette

## G.5 Key Comparisons to Highlight

1. **Ground Truth vs Others**: Does human-annotated preprocessing give best decomposition?
2. **FM vs Traditional**: Do foundation models produce cleaner decompositions?
3. **Ensemble vs Single Methods**: Does ensembling help decomposition?

