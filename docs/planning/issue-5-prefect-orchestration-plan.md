# Issue #5: Prefect Orchestration Pipeline - Implementation Plan

**Issue**: Implement reproducibility pipeline with Prefect orchestration
**Status**: ✅ Core implementation complete, refinements remaining
**Date**: 2026-01-25

## Executive Summary

This issue implements a two-block Prefect orchestration pipeline that separates data extraction (with privacy controls) from analysis, enabling reproducible results generation while protecting sensitive data.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│ BLOCK 1: EXTRACTION FLOW (requires mlruns + SERI DB access)            │
│                                                                         │
│  Inputs:                                                                │
│    - /home/petteri/mlruns (83GB MLflow experiments)                    │
│    - SERI_PLR_GLAUCOMA.db (source database)                            │
│                                                                         │
│  Outputs:                                                               │
│    - data/public/foundation_plr_results.db  [SHAREABLE]                │
│    - data/private/subject_lookup.yaml       [GITIGNORED]               │
│    - data/private/demo_subjects_traces.pkl  [GITIGNORED]               │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ BLOCK 2: ANALYSIS FLOW (works from public DB alone)                    │
│                                                                         │
│  Inputs:                                                                │
│    - data/public/foundation_plr_results.db                             │
│    - (optional) data/private/* for demo trace figures                  │
│                                                                         │
│  Outputs:                                                               │
│    - figures/generated/*.png, *.pdf                                    │
│    - figures/generated/data/*.json                                     │
│    - outputs/tables/*.tex                                              │
│    - outputs/latex/numbers.tex                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Completed Tasks ✅

### 1. Two-Block Architecture Design
- [x] Defined separation of concerns (extraction vs analysis)
- [x] Privacy model: re-anonymization scheme (PLRxxxx → Hxxx/Gxxx)
- [x] Graceful degradation when private data unavailable

### 2. Configuration Files
- [x] `configs/demo_subjects.yaml` - 8 demo subjects for trace visualization
- [x] Updated `.gitignore` for privacy protection
- [x] Updated `.claude/CLAUDE.md` with pipeline documentation

### 3. Block 1: Extraction Flow
- [x] `src/orchestration/flows/extraction_flow.py`
  - [x] `generate_subject_mapping()` - Creates Hxxx/Gxxx codes for all 208 subjects
  - [x] `save_private_lookup()` - Saves re-identification table (gitignored)
  - [x] `extract_mlflow_runs()` - Extracts 25,452 predictions with anonymized codes
  - [x] `export_to_duckdb()` - Creates shareable public database
  - [x] `extract_demo_traces()` - Extracts 8 demo PLR traces (gitignored)

### 4. Block 2: Analysis Flow
- [x] `src/orchestration/flows/analysis_flow.py`
  - [x] `load_public_data()` - Loads from public DuckDB
  - [x] `check_private_data()` - Warns if private data missing
  - [x] `generate_demo_trace_figure()` - Gracefully skips if no private data
  - [x] Placeholder tasks for stats/figures (TODO: connect to existing viz modules)

### 5. Master Orchestration
- [x] `scripts/reproduce_all_results.py` - CLI for running pipeline
- [x] Makefile targets: `reproduce`, `reproduce-from-checkpoint`, `extract`, `analyze`

### 6. Test Suite (TDD)
- [x] `tests/test_orchestration_flows.py` - 28 tests, all passing
  - [x] Demo subjects config validation
  - [x] Private lookup table tests
  - [x] Subject re-anonymization tests
  - [x] DuckDB export tests
  - [x] Demo trace extraction tests
  - [x] Graceful degradation tests
  - [x] Privacy guarantee tests
  - [x] Prediction extraction tests
  - [x] End-to-end integration test

### 7. Generated Artifacts
- [x] `data/public/foundation_plr_results.db` (1.51 MB, 25,452 predictions, 407 metrics)
- [x] `data/private/subject_lookup.yaml` (208 subject mappings)
- [x] `data/private/demo_subjects_traces.pkl` (8 PLR traces)

---

## Remaining Tasks 🔲

### Priority 1: Connect Analysis Flow to Existing Visualization

The analysis flow has placeholder tasks. Need to connect to existing modules:

- [ ] **Connect forest plot generation** → `src/viz/forest_plot.py`
- [ ] **Connect CD diagram generation** → `src/viz/cd_diagram.py`
- [ ] **Connect calibration plots** → `src/viz/calibration.py`
- [ ] **Connect demo trace figure** → `src/viz/light_protocol_plot.py`
- [ ] **Connect variance decomposition** → `src/stats/variance_decomposition.py`
- [ ] **Connect LaTeX table generation** → existing table generators

### Priority 2: Verify Full Figure Pipeline

- [ ] Run `make reproduce` and verify all figures generate correctly
- [ ] Verify JSON data files are created for each figure
- [ ] Verify numbers.tex contains all manuscript statistics

### Priority 3: Documentation

- [ ] Update `docs/api-reference/orchestration.md` with new flows
- [ ] Add usage examples to quickstart guide
- [ ] Document the privacy model in a dedicated page

### Priority 4: CI/CD Integration (Optional)

- [ ] Add GitHub Action to run `make reproduce-from-checkpoint`
- [ ] Verify reproducibility on clean environment
- [ ] Consider adding pre-commit hook for test suite

---

## Commands Reference

```bash
# Full pipeline (requires mlruns access)
make reproduce

# From checkpoint (public DB exists)
make reproduce-from-checkpoint

# Individual blocks
make extract    # Block 1 only
make analyze    # Block 2 only

# Run tests
make test-orchestration
PREFECT_DISABLED=1 pytest tests/test_orchestration_flows.py -v
```

---

## Data Privacy Summary

| Data Type | Code Format | Location | Git Status |
|-----------|-------------|----------|------------|
| Bootstrap predictions | Hxxx/Gxxx | `data/public/` | ✅ Tracked |
| Aggregate metrics | N/A | `data/public/` | ✅ Tracked |
| Subject lookup | Hxxx → PLRxxxx | `data/private/` | ❌ Gitignored |
| Demo PLR traces | N/A | `data/private/` | ❌ Gitignored |
| Demographics | N/A | Not extracted | ❌ Not stored |

---

## Test Coverage

```
tests/test_orchestration_flows.py
├── TestDemoSubjectsConfig (5 tests)
├── TestPrivateLookupTable (5 tests)
├── TestSubjectReanonymization (3 tests)
├── TestDuckDBExport (2 tests)
├── TestDemoTracesExtraction (2 tests)
├── TestAnalysisFlowGracefulDegradation (3 tests)
├── TestPrivacyGuarantees (3 tests)
├── TestPredictionExtraction (2 tests)
├── TestDemoTracesComplete (2 tests)
└── TestEndToEndIntegration (1 test)

Total: 28 tests, 28 passed
```

---

## Acceptance Criteria for Issue Closure

- [x] Two-block architecture implemented and tested
- [x] Re-anonymization working (no PLRxxxx codes in public data)
- [x] 28 tests passing
- [x] Extraction produces correct outputs (25,452 predictions, 8 traces)
- [x] Analysis flow has graceful degradation
- [x] Makefile targets working
- [ ] **Analysis flow connected to visualization modules**
- [ ] **Full `make reproduce` generates all manuscript figures**

---

## Files Modified/Created

### New Files
- `configs/demo_subjects.yaml`
- `src/orchestration/flows/__init__.py`
- `src/orchestration/flows/extraction_flow.py`
- `src/orchestration/flows/analysis_flow.py`
- `scripts/reproduce_all_results.py`
- `tests/test_orchestration_flows.py`
- `docs/planning/issue-5-prefect-orchestration-plan.md`

### Modified Files
- `.gitignore` (added privacy patterns)
- `.claude/CLAUDE.md` (added orchestration docs)
- `Makefile` (added reproduce targets)
- `tests/conftest.py` (added PREFECT_DISABLED)

### Generated Files (gitignored where applicable)
- `data/public/foundation_plr_results.db`
- `data/private/subject_lookup.yaml`
- `data/private/demo_subjects_traces.pkl`
