# PR Rebasing Plan: Splitting `chore/publication-polish` into Semantic PRs

## Context

- **Source branch**: `chore/publication-polish` (325 commits, 1,625 files changed)
- **Target branch**: `main`
- **Merge strategy**: Squash-and-merge (each PR = 1 final commit)
- **Method**: Cherry-pick FILES from HEAD, not commits. Each file appears in exactly one PR.
- **Docs principle**: "Hebbian documentation" -- docs ship with the code they describe.

## Branch Naming Convention

```
pr/NN-short-description    (e.g., pr/01-core-stats-modules)
```

## Execution Strategy

For each PR branch:
```bash
git checkout main
git checkout -b pr/NN-description
# Copy files from chore/publication-polish HEAD
git checkout chore/publication-polish -- path/to/files
git commit -m "descriptive message"
git push -u origin pr/NN-description
gh pr create --title "..." --body "..."
```

After squash-merge of PR N, the next branch (N+1) is created from updated main.

---

## PR Plan (27 PRs)

### Phase 1: Foundation Infrastructure (PRs 01-05)

#### PR 01: Core Stats Modules
**Theme**: STRATOS-compliant metrics computation library
**Files** (~35):
- `src/stats/` (22 files: `__init__.py`, `_defaults.py`, `calibration_extended.py`, `classifier_metrics.py`, `clinical_utility.py`, `effect_sizes.py`, `fdr_correction.py`, `nemenyi_test.py`, `pminternal_wrapper.py`, `scaled_brier.py`, `uncertainty_propagation.py`, `decision_uncertainty.py`, `ci_aggregation.py`, etc.)
- `tests/unit/test_calibration_extended.py`, `test_clinical_utility.py`, `test_scaled_brier.py`, `test_effect_sizes.py`, `test_fdr_correction.py`, `test_nemenyi.py`, `test_pminternal_wrapper.py`, `test_pminternal_analysis.py`, `test_uncertainty_propagation.py`, `test_decision_uncertainty.py`, `test_ci_aggregation.py`, `test_bootstrap.py`
- `docs/explanation/stratos-metrics.md`, `docs/tutorials/stratos-metrics.md`

#### PR 02: Data I/O and Registry Infrastructure
**Theme**: MLflow registry, data loading, streaming export
**Files** (~50):
- `src/data_io/` (22 files: `registry.py`, `display_names.py`, `data_loader.py`, `data_utils.py`, `streaming_duckdb_export.py`, etc.)
- `configs/mlflow_registry/` (12 files: parameters, methods, frozen registry)
- `tests/unit/test_display_names.py`, `test_data_loader.py`, `test_data_utils.py`, `test_duckdb_export.py`, etc.
- `tests/test_registry.py`
- `tests/test_data_io/test_streaming_duckdb_export.py`

#### PR 03: Configuration System Overhaul
**Theme**: Hydra configs, visualization configs, experiment configs
**Files** (~60):
- `configs/VISUALIZATION/` (10 files: `figure_registry.yaml`, `plot_hyperparam_combos.yaml`, `figure_layouts.yaml`, `metrics.yaml`, `colors.yaml`, `display_names.yaml`, etc.)
- `configs/CLS_HYPERPARAMS/` (5 files)
- `configs/CLS_MODELS/` (6 files)
- `configs/OUTLIER_MODELS/` (12 files)
- `configs/MODELS/` (7 files)
- `configs/PLR_FEATURIZATION/` (3 files)
- `configs/_version_manifest.yaml`
- `configs/combos/`, `configs/experiment/`, `configs/demo_subjects.yaml`
- `configs/defaults.yaml` (if modified)
- `src/data_io/config_loader.py`, `src/data_io/config_utils.py`
- `tests/unit/test_config_loader.py`, `test_config_utils.py`, `test_experiment_config.py`, `test_config_versioning.py`
- `tests/integration/test_config_integration.py`
- `docs/getting-started/configuration.md`

#### PR 04: Classification Pipeline
**Theme**: CatBoost, XGBoost, ensemble, weighing utils
**Files** (~50):
- `src/classification/` (75 files but many may exist on main -- include only modified/new)
- `src/ensemble/` (9 files)
- `tests/unit/test_ensemble_utils.py`, `test_balanced_subset.py`, `test_classifier_log_utils.py`
- `tests/integration/test_classification.py`
- `tests/unit/test_weighing_utils.py`
- `docs/user-guide/classification.md`

#### PR 05: Feature Extraction and Preprocessing
**Theme**: Featurization, imputation, anomaly detection source code
**Files** (~95):
- `src/featurization/` (11 files)
- `src/imputation/` (48 files)
- `src/anomaly_detection/` (36 files)
- `src/preprocess/` (if modified)
- `tests/integration/test_anomaly_detection.py`, `test_imputation.py`
- `tests/unit/test_feature_utils.py`, `test_hyperparam_utils.py`, `test_hyperparam_validation.py`
- `docs/user-guide/outlier-detection.md`, `imputation.md`, `featurization.md`

---

### Phase 2: Visualization System (PRs 06-11)

#### PR 06: Python Visualization Core
**Theme**: plot_config, setup_style, COLORS, save_figure, metric_registry
**Files** (~25):
- `src/viz/__init__.py`, `plot_config.py`, `metric_registry.py`, `generate_all_figures.py`
- `src/viz/` shared utilities (non-figure-specific modules)
- `tests/unit/test_metric_registry.py`

#### PR 07: Python Visualization Figures (STRATOS)
**Theme**: Calibration, DCA, probability distribution, ROC, forest plots
**Files** (~40):
- `src/viz/calibration_plot.py`, `dca_plot.py`, `prob_distribution.py`, `retained_metric.py`, `uncertainty_scatter.py`, `forest_plot.py`, `stratos_figures.py`, `metric_vs_cohort.py`
- `tests/test_figure_generation/` (8 files)
- `tests/unit/test_calibration_plot.py`, `test_dca_plot.py`, `test_prob_distribution.py`, `test_retained_metric.py`, `test_uncertainty_scatter.py`, `test_stratos_figures.py`, `test_metric_vs_cohort.py`

#### PR 08: Python Visualization Figures (Analysis)
**Theme**: CD diagrams, specification curve, variance decomposition, SHAP, instability
**Files** (~35):
- `src/viz/cd_diagram.py`, `cd_diagram_preprocessing.py`, `specification_curve.py`, `variance_decomposition.py`, `featurization_comparison.py`, `foundation_model_dashboard.py`, `factorial_matrix.py`, `utility_matrix.py`, `fig_instability_plots.py`
- `tests/unit/test_variance_decomposition.py`, `test_fig_instability_plots.py`, `test_generate_instability_figures.py`
- `src/viz/decomposition_grid.py`
- `tests/test_decomposition/` (3 files)

#### PR 09: R Figure System
**Theme**: R/ggplot2 infrastructure, figure scripts, config loaders
**Files** (~55):
- `src/r/` (44 files: `theme_foundation_plr.R`, `config_loader.R`, `category_loader.R`, figure scripts, `helpers/`)
- `renv.lock`, `renv/` (R environment)
- `tests/test_r_figures/` (2 files)
- `tests/test_r_environment.py`
- `tests/test_no_hardcoding/test_r_*.py` (4 files)

#### PR 10: SHAP Analysis
**Theme**: SHAP extraction, figure generation, VIF analysis
**Files** (~15):
- `src/viz/shap_figures.py` (if exists)
- `scripts/extraction/generate_shap_figures.py` (or wherever SHAP scripts live)
- `tests/test_shap_extraction.py`
- `tests/test_vif_analysis.py`
- Related config files for SHAP combos

#### PR 11: Generated Figures and Data Exports
**Theme**: All generated PNG/SVG/PDF figures, JSON data sidecars, R data exports
**Files** (~100+):
- `figures/generated/` (87 files: PNGs, JSONs, data/)
- `data/r_data/` (export CSVs/JSONs for R)
- `scripts/extraction/` (data export scripts for R figures)

---

### Phase 3: Testing Infrastructure (PRs 12-15)

#### PR 12: Unit and Integration Test Expansion
**Theme**: Tests not tied to a specific PR above
**Files** (~40):
- `tests/unit/` remaining files not covered by PRs 01-10
- `tests/integration/` remaining (artifact_consistency, data_import, extraction_*, synthetic_pipeline, etc.)
- `tests/e2e/` (3 files)
- `tests/smoke/` (1 file)
- `tests/conftest.py`

#### PR 13: Figure QA Test Suite
**Theme**: Zero-tolerance figure quality assurance
**Files** (~12):
- `tests/test_figure_qa/` (9 files: data_provenance, statistical_validity, publication_standards, rendering_artifacts, accessibility, visual_rendering, no_nan_ci)
- Figure QA golden images if any

#### PR 14: Guardrail Tests (Code Quality)
**Theme**: Hardcoding prevention, computation decoupling enforcement
**Files** (~25):
- `tests/test_guardrails/test_no_hardcoded_values.py`, `test_computation_decoupling.py`, `test_data_location_policy.py`, `test_yaml_consistency.py`, `test_json_provenance.py`, `test_deliverables_verification.py`
- `tests/test_no_hardcoding/` (12 files: absolute_paths, computation_decoupling, method_abbreviations, display_names, hex_colors, registry_compliance, yaml_single_source)
- `tests/test_guardrails/conftest.py`, `__init__.py`

#### PR 15: Test Hardening and Remaining Test Files
**Theme**: Legacy R tool tests, documentation tests, subject trace tests, data quality tests
**Files** (~30):
- `tests/test_legacy_r_tools/`
- `tests/test_documentation/`
- `tests/test_subject_traces/`
- `tests/test_data_quality/`
- `tests/test_data_integrity.py`
- `tests/test_extraction/`, `test_extraction_double_check.py`, `test_extraction_verification.py`
- `tests/test_pminternal_*.py`
- `tests/test_report_metrics.py`
- `tests/test_foundation_plr.py`
- `tests/test_synthetic/`

---

### Phase 4: Project Infrastructure (PRs 16-20)

#### PR 16: Repository Reorganization
**Theme**: Script restructuring, directory moves, root cleanup
**Files** (~30):
- `scripts/` (52 files: `validation/`, `extraction/`, `infra/`, `misc/`)
- `Makefile`
- `.gitignore`
- `.pre-commit-config.yaml`

#### PR 17: Docker Infrastructure
**Theme**: Dockerfiles, docker-compose, Docker CI
**Files** (~10):
- `Dockerfile`, `Dockerfile.r`, `Dockerfile.dev` (if exist)
- `docker-compose.yml`
- `apps/visualization/` (React/TypeScript app, 16 files)

#### PR 18: Orchestration and Pipeline Flows
**Theme**: Prefect flows, extraction flow, analysis flow
**Files** (~15):
- `src/orchestration/` (12 files)
- `tests/test_orchestration_flows.py`
- `docs/user-guide/prefect-orchestration.md`, `pipeline-overview.md`

#### PR 19: Synthetic Data and Isolation Architecture
**Theme**: Synthetic demo database, isolation gates, data mode
**Files** (~15):
- `data/synthetic/SYNTH_PLR_DEMO.db`
- `src/data_io/data_mode.py` (if exists)
- `tests/unit/test_data_mode.py`, `test_data_validators.py`
- `tests/e2e/test_synthetic_isolation_e2e.py`
- `tests/integration/test_extraction_isolation.py`, `test_mlflow_isolation.py`

#### PR 20: Root Documentation (README, ARCHITECTURE, AGENTS)
**Theme**: Top-level project docs
**Files** (~10):
- `README.md`
- `ARCHITECTURE.md`
- `AGENTS.md`
- `CITATION.cff`
- `LICENSE`
- `pyproject.toml` (if changed)
- `uv.lock` (dependency lockfile)

---

### Phase 5: Documentation Expansion (PRs 21-25)

#### PR 21: MkDocs Documentation Site
**Theme**: MkDocs config, docs site structure, tutorials, getting-started, user-guide, API reference
**Files** (~40):
- `mkdocs.yml`
- `docs/index.md`
- `docs/getting-started/` (installation, quickstart, index)
- `docs/user-guide/index.md`
- `docs/api-reference/` (15 files)
- `docs/explanation/` (research-question, stratos-metrics, index)
- `docs/research/` (documentation-as-code-analysis)
- `docs/tutorials/index.md`, `running-experiments.md`, `adding-new-methods.md`, `reading-plots.md`, `reproducibility.md`, `dependencies.md`, `translational-insights.md`
- `docs/concepts-for-researchers.md`, `docs/mlflow-naming-convention.md`
- `docs/KNOWLEDGE_GRAPH.md`, `docs/API_ARCHITECTURE.md`
- `docs/STRATOS_CALIBRATION_IMPLEMENTATION.md`
- `docs/stylesheets/extra.css`, `docs/javascripts/mathjax.js`

#### PR 22: Repository Figures (Assets + Plans, Batch 1)
**Theme**: fig-repo-01 to fig-repo-40 (documentation infographics)
**Files** (~120):
- `docs/repo-figures/assets/fig-repo-01*` through `fig-repo-40*` (JPG images)
- `docs/repo-figures/figure-plans/fig-repo-01*` through `fig-repo-40*` (markdown plans)
- `docs/repo-figures/README.md`, `CLAUDE.md`, `STYLE-GUIDE.md`, `CONTENT-TEMPLATE.md`, `PROMPTING-INSTRUCTIONS.md`
- `docs/repo-figures/figure-alt-text-catalog.md`

#### PR 23: Repository Figures (Assets + Plans, Batch 2)
**Theme**: fig-repo-41 to fig-repo-98, fig-repro, fig-trans series
**Files** (~150):
- `docs/repo-figures/assets/fig-repo-41*` through `fig-repo-98*`
- `docs/repo-figures/figure-plans/fig-repo-41*` through `fig-repo-98*`
- `docs/repo-figures/figure-plans/fig-repro-*` (24 files)
- `docs/repo-figures/figure-plans/fig-trans-*` (20 files)
- `docs/repo-figures/generated/archived/` (4 files)
- `docs/repo-figures/*.md` (coverage plans, deeper plan, layprofessional)

#### PR 24: Planning Documents
**Theme**: All planning and figure-report markdown
**Files** (~80):
- `docs/planning/` (64 files)
- `docs/figure-reports/` (5 files)
- `docs/repo-documentation-plan.md`, `test-coverage-improvement-plan.md`

#### PR 25: Claude Code Context (`.claude/`)
**Theme**: AI assistant context, rules, skills, meta-learnings
**Files** (~86):
- `.claude/CLAUDE.md`
- `.claude/rules/` (6 files)
- `.claude/skills/` (25 files)
- `.claude/docs/` (17 files: meta-learnings, code-reviews, etc.)
- `.claude/domains/` (4 files)
- `.claude/planning/` (15 files)
- `.claude/auto-context.yaml`
- `CLAUDE.md` (root)

---

### Phase 6: CI/CD (PR 26, LAST)

#### PR 26: CI/CD Workflows + Guardrail Tests (LAST, AUTONOMOUS)
**Theme**: All GitHub Actions workflows, CI guardrail tests, docs link guardrail
**Auto-merge**: YES, via `gh pr merge --squash` once ALL CI jobs are green.
Claude Code monitors, fixes failures autonomously, and merges when ready.
**Files** (~7):
- `.github/workflows/ci.yml` (enhanced with `workflow_dispatch` inputs)
- `.github/workflows/config-integrity.yml`
- `.github/workflows/deploy-docs.yml`
- `.github/workflows/docker.yml`
- `tests/test_guardrails/test_ci_workflow_structure.py`
- `tests/test_guardrails/test_docs_links.py`

**Why combined**: No CI runs on PRs before this one (no workflow files on main yet).
Link guardrail test has no value being merged separately without CI to validate it.

---

## Merge Order

```
PR 01 → 02 → 03 → 04 → 05 (Foundation: independent, logical order)
PR 06 → 07 → 08 (Python viz: 06 first as core, then figures)
PR 09 → 10 → 11 (R/SHAP/generated: 09 first for R system)
PR 12 → 13 → 14 → 15 (Tests: any order within phase)
PR 16 → 17 → 18 → 19 → 20 (Infra: 16 first for directory structure)
PR 21 → 22 → 23 → 24 → 25 (Docs: 21 first for MkDocs config)
PR 26 (CI/CD — LAST, autonomous monitor + fix + auto-merge)
```

PRs 01-25 have NO CI to run (workflow files don't exist on main yet).
They are auto-merged without waiting for any checks.

## Autonomous CI Monitoring & Fixing Protocol for PR 26

### Overview

PR 26 introduces CI to a codebase that has never run on GitHub Actions.
This protocol allows Claude Code to **autonomously monitor, diagnose, fix,
and re-verify** CI failures without human intervention.

**Trigger**: After PR 26 is opened / first CI run starts.
**Goal**: All CI jobs green → auto-merge via `gh pr merge --squash`.

### Prerequisites

```bash
# Verify gh CLI is authenticated and can access the repo
gh auth status
gh repo view petteriTeikari/foundation_PLR --json name
```

### `workflow_dispatch` Inputs (already in ci.yml)

```yaml
on:
  workflow_dispatch:
    inputs:
      test_filter:          # e.g., "tests/test_guardrails/ -v" or "-k test_name"
        type: string
        default: ''
      skip_integration:     # checkbox to skip integration tier
        type: boolean
        default: false
```

---

### Phase 1: Monitor Initial CI Run

```bash
# 1. Get the latest workflow run for the PR branch
gh run list --branch pr/26-ci-cd-workflows --limit 3

# 2. Watch the run in real-time (blocks until complete)
gh run watch <RUN_ID>

# 3. If run finished, check status
gh run view <RUN_ID> --json status,conclusion,jobs

# 4. Get failed job details
gh run view <RUN_ID> --json jobs --jq '.jobs[] | select(.conclusion=="failure") | {name, conclusion}'

# 5. Download full logs for failed jobs
gh run view <RUN_ID> --log-failed
```

### Phase 2: Diagnose Failures (Root Cause, Not Whac-a-Mole)

For each failed job, follow this **mandatory diagnosis checklist** before writing any fix:

```
DIAGNOSIS CHECKLIST (do ALL before coding):
[ ] 1. Read the FULL error output (not just the last line)
[ ] 2. Identify the error CATEGORY (see table below)
[ ] 3. Check: Does this test pass locally? (run it)
[ ] 4. Check: Is the failure environment-specific or a real bug?
[ ] 5. Determine the CORRECT fix approach (see decision tree)
[ ] 6. Write the fix
[ ] 7. Verify fix locally before pushing
```

**Error Category Decision Tree:**

```
Is the test failing because of missing LOCAL-ONLY resources?
├── YES (mlruns/, private data, local R packages)
│   → Add @pytest.mark.skipif with descriptive reason
│   → NEVER delete the test. It must run when resources exist.
│
├── NO → Is it an environment difference (Ubuntu vs local)?
│   ├── Missing system package → Add to workflow setup steps
│   ├── Missing Python dep → Check uv.lock is committed
│   ├── Path difference → Use PROJECT_ROOT / relative paths
│   ├── Timeout too short → Increase timeout in workflow
│   └── Font/display issue → Ensure MPLBACKEND=Agg
│
└── NO → Is it a real bug exposed by CI?
    → Fix the actual code. This is a genuine issue.
```

### Common CI-vs-Local Issues

| Issue | Local | GitHub Actions | Fix Approach |
|-------|-------|---------------|-------------|
| MLflow artifacts | `/home/petteri/mlruns` | Not available | `@pytest.mark.skipif(not Path(mlruns).exists())` |
| Private DuckDB | `data/private/` | Not in repo | Skip tests requiring private data |
| Public DuckDB | `data/public/*.db` | In repo (if committed) | Verify file is in git |
| R packages | System R + renv | `r-lib/actions/setup-r` | Already configured in r-lint job |
| Rscript timeout | Fast | Slow CI runner | Increase timeout or add `timeout` param |
| Arial font | Installed | Not on Ubuntu | `MPLBACKEND=Agg` (already set) |
| CUDA/GPU | Available | No GPU | Tests already mock `torch.cuda.is_available` |
| Synthetic DB | `data/synthetic/` | Must be committed | Ensure `.db` file is in git |

### Phase 3: Fix, Push, Re-run (Targeted)

For EACH failure, follow this loop:

```bash
# 1. Make the fix locally
#    (edit test file, workflow, or source code)

# 2. Verify fix passes locally
uv run python -m pytest <EXACT_TEST_PATH> -v

# 3. Commit and push
git add <changed_files>
git commit -m "fix(ci): <description of root cause and fix>"
git push

# 4. Trigger TARGETED re-run (only the failing test, saves credits)
gh workflow run ci.yml \
  --ref pr/26-ci-cd-workflows \
  -f test_filter="<EXACT_TEST_PATH> -v" \
  -f skip_integration=true

# 5. Monitor the targeted run
gh run list --branch pr/26-ci-cd-workflows --limit 1
gh run watch <NEW_RUN_ID>

# 6. Verify it passed
gh run view <NEW_RUN_ID> --json conclusion --jq '.conclusion'
# Expected: "success"
```

**Repeat for each distinct failure.** Do NOT batch unrelated fixes.

### Phase 4: Full Validation Run

Once all individual failures are fixed and verified:

```bash
# 1. Trigger full run (no filter)
gh workflow run ci.yml --ref pr/26-ci-cd-workflows

# 2. Watch until completion
RUN_ID=$(gh run list --branch pr/26-ci-cd-workflows --limit 1 --json databaseId --jq '.[0].databaseId')
gh run watch $RUN_ID

# 3. Verify ALL jobs passed
gh run view $RUN_ID --json jobs --jq '.jobs[] | {name: .name, status: .conclusion}'
# ALL must show "success"
```

### Phase 5: Auto-Merge

Once full CI is green:

```bash
# 1. Verify PR is ready
gh pr view <PR_NUMBER> --json mergeable,reviewDecision,statusCheckRollup

# 2. Auto-merge with squash
gh pr merge <PR_NUMBER> --squash --auto \
  --subject "ci: Add GitHub Actions CI/CD with smart path filtering and test inputs"

# 3. Verify merge completed
gh pr view <PR_NUMBER> --json state --jq '.state'
# Expected: "MERGED"
```

### Pre-Flight: Known Local-Only Dependencies (11 test files)

These files reference `/home/petteri/mlruns` or `/home/petteri/` paths and WILL
fail on CI. Add proper `skipif` markers BEFORE opening PR 26 to minimize
first-run failures:

```
tests/test_extraction_double_check.py
tests/test_extraction_verification.py
tests/test_orchestration_flows.py
tests/test_no_hardcoding/test_absolute_paths.py
tests/test_decomposition/test_data_loading.py
tests/integration/test_artifact_consistency.py
tests/test_shap_extraction.py
tests/test_pminternal_extraction.py
tests/test_data_io/test_streaming_duckdb_export.py
tests/test_data_quality/test_normalization_consistency.py
tests/unit/test_streaming_duckdb_export.py
```

**Action**: Before creating PR 26, grep each file for hardcoded local paths.
Add `@pytest.mark.skipif` with reason for genuinely local-only resource access.
Tests that use local paths for VALIDATION (checking no hardcoded paths exist)
should remain as-is -- they test string patterns, not access the paths.

### Anti-Patterns (BANNED)

| BANNED | DO INSTEAD |
|--------|------------|
| Deleting a failing test | Add `@pytest.mark.skipif` with reason |
| `# noqa` / `# type: ignore` to silence | Fix the actual issue |
| Disabling an entire CI job | Fix the job or mark specific tests |
| Re-running without any fix ("maybe it'll pass") | Diagnose first |
| Fixing 5 things at once | One fix per commit, verify each |
| Pushing to main directly | Always push to PR branch |
| Weakening test assertions | Fix environment or add proper skip |

### Skip Marker Convention

When a test must be skipped on CI due to genuinely unavailable resources:

```python
import os
from pathlib import Path

# For MLflow-dependent tests
MLRUNS_DIR = Path("/home/petteri/mlruns")
pytestmark = pytest.mark.skipif(
    not MLRUNS_DIR.exists(),
    reason="MLflow artifacts not available (local-only resource)"
)

# For private data tests
PRIVATE_DATA = Path(__file__).parent.parent.parent / "data" / "private"
@pytest.mark.skipif(
    not PRIVATE_DATA.exists(),
    reason="Private patient data not available on CI"
)

# For CI environment detection (generic)
ON_CI = os.environ.get("CI") == "true" or os.environ.get("GITHUB_ACTIONS") == "true"
```

### Credit Budget

| Job | Duration | Cost (approx) |
|-----|----------|---------------|
| Lint | ~30s | negligible |
| Tier 1 (filtered) | ~1-2 min | minimal |
| Tier 1 (full) | ~5 min | low |
| Quality gates | ~2 min | low |
| Integration | ~5 min | low |
| R lint | ~2 min | low |
| **Full run** | **~10-15 min** | **moderate** |

**Budget target**: Fix all failures in 3-5 targeted runs + 1 full validation = ~20-25 min total.
Avoid re-running full suite more than twice.

## Conflict Mitigation

- **uv.lock**: Include in PR 20 (with pyproject.toml). All subsequent PRs rebase on updated main.
- **Each file committed exactly once**: No cross-PR file conflicts by design.
- If a file logically belongs to multiple PRs, include it in the FIRST relevant PR.
- After each merge, subsequent PR branches rebase from updated main (may need trivial conflict resolution in files added by earlier PRs).

## Estimated Time

- Creating 26 branches with cherry-picked files: ~2-3 hours (automated)
- Creating 26 PRs: ~30 minutes
- Merging 25 auto-merge PRs: ~1-2 hours (sequential)
- PR 26 autonomous CI monitoring + fixing + auto-merge: ~1-2 hours

## Notes

- `src/tools/` (226 files, all Added) contains legacy R-PLR wiki HTML pages with Unicode paths. Include in PR 15 (legacy R tools) or a dedicated "legacy tools" PR. Consider whether these should even be committed (they're saved HTML + assets -- quite large).
- `.coverage` file should NOT be committed (add to .gitignore if not already).
- Binary assets (JPG images, .db files) will make PRs 11, 19, 22, 23 large.
- The `figures/generated/fig_decomposition_grid_TEST.*` files are now gitignored; exclude from PRs.
