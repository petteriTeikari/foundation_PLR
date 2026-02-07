# New Repository Figures Plan: 38 Figures in 7 Groups

**Date**: 2026-02-06
**Status**: PROPOSAL - awaiting approval
**Numbering**: fig-repo-61 through fig-repo-98

## Gap Analysis

### What EXISTS (100+ plans across 3 series)
- **fig-repo-01 to 60**: Pipeline, MLflow, Hydra, STRATOS, DuckDB, methods, calibration/DCA, combos, logging, uv, Polars, "how to read" guides, pre-commit (basic), test pyramid (basic)
- **fig-repro-01 to 24**: Reproducibility crisis, notebooks, 5 horsemen, dependency hell, Docker, lockfiles, R4R
- **fig-trans-01 to 20**: Transferability, domain fit, sparse vs dense, domain-specific vs generic

### What's MISSING (identified gaps)
1. **Test skip groups A-H** (user explicitly requested)
2. **Docker testing flow** (local vs Docker vs CI decision tree)
3. **GitHub Actions CI pipeline** (job dependency graph)
4. **Test marker system** (how pytest markers partition the suite)
5. **Docker multi-image architecture** (4 Dockerfiles, their purposes)
6. **Registry anti-cheat system** (5-layer verification)
7. **Computation decoupling enforcement** (import ban mechanics)
8. **Color system architecture** (YAML → dict → resolve_color flow)
9. **Data path resolution** (how conftest.py finds databases)
10. **DuckDB entity-relationship diagram**
11. **Configuration space anatomy** (11 x 8 x 5 grid)
12. **CLAUDE.md instruction hierarchy**
13. **Developer onboarding flow**
14. **R figure system architecture**
15. **Extraction guardrails** (memory/disk/stall)
16. **Ensemble construction** (how 7-method ensemble is built)

---

## GROUP 1: Testing Architecture (7 figures)
*The user explicitly requested test documentation. This is the highest-priority group.*

### fig-repo-61: Test Skip Groups A-H
| Field | Value |
|-------|-------|
| **Title** | Test Skip Groups: What's Skipped and Why |
| **Complexity** | L3 |
| **Persona** | ML Engineer / Research Scientist |
| **Priority** | P1 (user-requested) |

**Key Message**: "181 test skips are organized into 7 groups (A-H), each with a distinct root cause and fix strategy."

**Content**:
```
┌─────────────────────────────────────────────────────────────────────┐
│              TEST SKIP GROUPS: A DIAGNOSTIC MAP                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Group A: DB Paths (43 skips)        Group B: R_DATA_DIR (34 skips)│
│  ┌──────────────────────┐            ┌──────────────────────┐      │
│  │ 5 test files          │            │ test_figure_qa/      │      │
│  │ Root: DB not at       │            │ Root: R output data  │      │
│  │ expected path         │            │ not generated yet    │      │
│  │ Fix: make extract     │            │ Fix: make r-figures  │      │
│  └──────────────────────┘            └──────────────────────┘      │
│                                                                     │
│  Group D: Figure Filenames (24)      Group E: Demo Data (11)       │
│  ┌──────────────────────┐            ┌──────────────────────┐      │
│  │ Root: Generated fig   │            │ Root: Demo subjects  │      │
│  │ names don't match     │            │ not yet created      │      │
│  │ expected patterns     │            │ Fix: make analyze    │      │
│  │ Fix: Update registry  │            └──────────────────────┘      │
│  └──────────────────────┘                                           │
│                                                                     │
│  Group F: Manuscript (10)   Group G: TDD Stubs (6)  Group H: (9)  │
│  ┌───────────────┐          ┌───────────────┐       ┌───────────┐  │
│  │ LaTeX/artifact │          │ Placeholder    │       │ Vendored  │  │
│  │ tests need     │          │ tests for      │       │ exception │  │
│  │ make analyze   │          │ future work    │       │ skips     │  │
│  └───────────────┘          │ Delete when    │       └───────────┘  │
│                              │ implemented    │                      │
│                              └───────────────┘                      │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ RESOLUTION FLOWCHART                                         │   │
│  │ make extract → make analyze → make r-figures → 0 skips      │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

### fig-repo-62: Test Tier to CI Job Mapping
| Field | Value |
|-------|-------|
| **Title** | From Test Tiers to CI Jobs: How Tests Run in GitHub Actions |
| **Complexity** | L3 |
| **Persona** | ML Engineer |
| **Priority** | P1 (user-requested) |

**Key Message**: "4 test tiers map to 5 CI jobs with explicit dependency chains and parallelism."

**Content**:
```
┌─────────────────────────────────────────────────────────────────────┐
│           TEST TIERS → CI JOBS MAPPING                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  LOCAL TIERS              GITHUB ACTIONS JOBS                      │
│  ┌─────────────┐         ┌─────────────────────┐                  │
│  │ Tier 0:     │─────────│ lint (5 min)        │──┐               │
│  │ ruff lint   │         │ ruff check + format  │  │               │
│  └─────────────┘         └─────────────────────┘  │ PARALLEL      │
│  ┌─────────────┐         ┌─────────────────────┐  │               │
│  │ Tier 1:     │─────────│ test-fast (10 min)  │──┤               │
│  │ unit +      │         │ -m "unit or guard"   │  │               │
│  │ guardrail   │         │ pytest-xdist -n auto │  │               │
│  └─────────────┘         └──────────┬──────────┘  │               │
│  ┌─────────────┐         ┌──────────▼──────────┐  │               │
│  │ Quality     │─────────│ quality-gates       │──┤               │
│  │ Gates       │         │ verify_registry      │  │               │
│  └─────────────┘         │ check_decoupling     │  │               │
│                          │ check_parallel       │  │               │
│                          └──────────┬──────────┘  │               │
│  ┌─────────────┐         ┌──────────▼──────────┐  │               │
│  │ Tier 3:     │─────────│ test-integration    │  │               │
│  │ integration │         │ -m "integration or  │  │               │
│  │ + e2e       │         │  e2e" with xdist    │  │               │
│  └─────────────┘         └─────────────────────┘  │               │
│                          ┌─────────────────────┐  │               │
│                          │ r-lint (10 min)     │──┘               │
│                          │ R script syntax     │                   │
│                          └─────────────────────┘                   │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────┐     │
│  │ KEY: ── dependency │ ═══ parallel │ ~35 min total wall   │     │
│  └───────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────┘
```

---

### fig-repo-63: Test Marker System
| Field | Value |
|-------|-------|
| **Title** | pytest Markers: How 2000+ Tests Are Partitioned |
| **Complexity** | L3 |
| **Persona** | ML Engineer / Research Scientist |
| **Priority** | P1 |

**Key Message**: "7 pytest markers partition 2000+ tests into fast/slow, data-dependent/independent, and R-required groups for selective execution."

**Content**: Venn-diagram-style showing marker overlaps:
- **unit** (239): Pure functions, no I/O
- **integration** (96): Demo data, no patient data
- **e2e** (18): Full pipeline, slow
- **slow** (31): >30s runtime (overlaps with integration/e2e)
- **guardrail**: Code quality (overlaps unit)
- **data**: Needs data/r_data or data/public
- **r_required**: Needs Rscript binary

Command cheatsheet: `pytest -m unit`, `pytest -m "not slow"`, `pytest -m "integration and not r_required"`

---

### fig-repo-64: Test Fixture Architecture
| Field | Value |
|-------|-------|
| **Title** | conftest.py Hierarchy: Fixtures from Root to Leaf |
| **Complexity** | L4 |
| **Persona** | ML Engineer |
| **Priority** | P2 |

**Key Message**: "Test fixtures follow a tree hierarchy from root conftest.py (session-scoped DB connections) down to per-directory conftest.py files (module-scoped mocks)."

**Content**:
```
tests/conftest.py (ROOT)
├── Session fixtures: results_db_path, cd_diagram_db_path, r_data_dir
├── Data paths: RESULTS_DB, CD_DIAGRAM_DB, FIGURES_DIR
├── Skip logic: pytest.mark.skipif(not path.exists())
│
├── test_figure_qa/conftest.py
│   ├── figure_dir, json_files, png_files
│   └── Scans figures/generated/ for QA
│
├── test_figure_generation/conftest.py
│   ├── db_connection (function-scoped)
│   └── Mock data for figure unit tests
│
├── unit/conftest.py
│   └── Pure function fixtures, no I/O
│
└── integration/conftest.py
    └── synthetic_db, demo_subjects
```

---

### fig-repo-65: Figure QA Test Categories
| Field | Value |
|-------|-------|
| **Title** | Figure QA: 7 Test Files, Zero Tolerance |
| **Complexity** | L3 |
| **Persona** | Research Scientist / ML Engineer |
| **Priority** | P1 |

**Key Message**: "Figure QA has 7 specialized test files organized by priority (P0=synthetic fraud, P1=statistical, P2=rendering, P3=accessibility). ALL must pass before any figure commit."

**Content**: Table + dependency tree:

| Priority | Test File | What It Catches | Example Failure |
|----------|-----------|-----------------|-----------------|
| P0 | test_data_provenance.py | Synthetic data in figures | CRITICAL-FAILURE-001 |
| P1 | test_statistical_validity.py | Invalid metrics | Overlapping CIs |
| P1 | test_no_nan_ci.py | NaN confidence intervals | Bootstrap failures |
| P2 | test_publication_standards.py | DPI/dimensions/fonts | <100 DPI figure |
| P2 | test_rendering_artifacts.py | Visual defects | Missing axis labels |
| P2 | test_visual_rendering.py | Visual inspection | Clipped content |
| P3 | test_accessibility.py | Color blindness | Red-green only palette |

---

### fig-repo-66: Test Data Flow
| Field | Value |
|-------|-------|
| **Title** | Where Test Data Comes From |
| **Complexity** | L3 |
| **Persona** | ML Engineer |
| **Priority** | P2 |

**Key Message**: "Test data flows through 4 channels: synthetic DB (isolated), demo subjects (8 curated), real DB (skipped if absent), and in-memory fixtures."

**Content**:
```
┌──────────────┐    ┌───────────────┐    ┌──────────────────┐
│ SYNTH_PLR_   │    │ demo_subjects │    │ foundation_plr_  │
│ DEMO.db      │    │ .yaml (8)     │    │ results.db       │
│ (synthetic)  │    │ H001-H004     │    │ (real, optional) │
│              │    │ G001-G004     │    │                  │
│ For: unit,   │    │ For: viz      │    │ For: integration │
│ guardrail    │    │ tests, demos  │    │ e2e tests        │
└──────┬───────┘    └──────┬────────┘    └────────┬─────────┘
       │                   │                      │
       │        ┌──────────▼──────────┐           │
       │        │ In-memory fixtures  │           │
       │        │ (conftest.py)       │           │
       │        │ np.random arrays,   │           │
       │        │ pd.DataFrame mocks  │           │
       └────────┤                     ├───────────┘
                │ ISOLATION RULE:     │
                │ Synthetic NEVER     │
                │ touches production  │
                └─────────────────────┘
```

---

### fig-repo-67: Local vs Docker vs CI Test Execution
| Field | Value |
|-------|-------|
| **Title** | Running Tests: Pick Your Path |
| **Complexity** | L2 |
| **Persona** | All (new developer onboarding) |
| **Priority** | P1 (user-requested) |

**Key Message**: "Three test execution paths serve different needs: local (fastest feedback), Docker (CI parity), GitHub Actions (PR gate)."

**Content**: Decision tree:
```
"I want to run tests"
  │
  ├── Quick check while coding?
  │   └── make test-local          # ~90s, Tier 1 only, needs Python
  │
  ├── Full suite, CI parity?
  │   └── make test-all            # Docker, all tiers, ~10 min
  │
  ├── Just figure QA?
  │   └── make test-figures        # ZERO TOLERANCE, ~30s
  │
  ├── Need R tests?
  │   └── make r-docker-test       # Docker with R, ~5 min
  │
  ├── Specific test file?
  │   └── pytest tests/path/file.py -v   # Direct, instant
  │
  └── PR submitted?
      └── GitHub Actions auto-runs:
          lint → test-fast → quality-gates → test-integration
```

---

## GROUP 2: CI/CD & Docker (5 figures)

### fig-repo-68: GitHub Actions CI Pipeline
| Field | Value |
|-------|-------|
| **Title** | CI Pipeline: From Push to Green Check |
| **Complexity** | L3 |
| **Persona** | ML Engineer |
| **Priority** | P1 (user-requested) |

**Key Message**: "The CI pipeline runs 5 jobs: 3 in parallel (lint, test-fast, r-lint), quality-gates after test-fast, then integration tests. Total wall time ~35 minutes."

**Content**: DAG visualization of ci.yml showing:
- Trigger conditions (push to main, PR)
- Job boxes with steps inside
- Dependency arrows
- Parallelism indicators
- Estimated durations
- Failure modes (which job fails most often, what to check)

---

### fig-repo-69: Docker Multi-Image Architecture
| Field | Value |
|-------|-------|
| **Title** | Four Docker Images, Four Purposes |
| **Complexity** | L3 |
| **Persona** | ML Engineer |
| **Priority** | P1 (user-requested) |

**Key Message**: "4 specialized Docker images serve different needs: full dev (2GB, Python+R+Node), R-only (1GB, renv), test-only (400MB, Python), and Shiny (interactive tools)."

**Content**:
```
┌────────────────────────────────────────────────────────────────────┐
│  Dockerfile (FULL)           Dockerfile.r (R FIGURES)             │
│  ┌──────────────────────┐    ┌──────────────────────┐             │
│  │ rocker/tidyverse:4.5 │    │ rocker/tidyverse:4.5 │             │
│  │ + Python 3.11 (uv)   │    │ + renv 1.1.6         │             │
│  │ + Node.js 20 LTS     │    │ + renv.lock pinned   │             │
│  │ ~2GB                 │    │ ~1GB                 │             │
│  │ USE: full dev env    │    │ USE: R figure gen    │             │
│  └──────────────────────┘    └──────────────────────┘             │
│                                                                    │
│  Dockerfile.test (LEAN)      Shiny (INTERACTIVE)                  │
│  ┌──────────────────────┐    ┌──────────────────────┐             │
│  │ python:3.11-slim     │    │ rocker/shiny         │             │
│  │ Multi-stage build    │    │ Ground truth tools   │             │
│  │ No R, No Node.js     │    │ Port 3838            │             │
│  │ ~400MB              │    │ USE: GT creation     │             │
│  │ USE: fast CI tests   │    │                      │             │
│  └──────────────────────┘    └──────────────────────┘             │
└────────────────────────────────────────────────────────────────────┘
```

---

### fig-repo-70: Docker Compose Service Map
| Field | Value |
|-------|-------|
| **Title** | Docker Compose: 7 Services, 2 Profiles |
| **Complexity** | L3 |
| **Persona** | ML Engineer |
| **Priority** | P2 |

**Key Message**: "docker-compose.yml defines 7 services: 3 always-available (dev, r-figures, test) and 2 profile-gated (viz, shiny). Volume mounts ensure code changes are live."

**Content**: Service topology diagram showing:
- Each service box with image, command, volumes, ports
- Volume mount arrows (src/ → container, read-only vs read-write)
- Profile grouping (default vs viz vs shiny)
- Port mappings (3000 for viz, 3838 for shiny)

---

### fig-repo-71: Docker Build Pipeline (CI)
| Field | Value |
|-------|-------|
| **Title** | Docker Workflow: Build, Test, Push |
| **Complexity** | L3 |
| **Persona** | ML Engineer |
| **Priority** | P2 |

**Key Message**: "docker.yml builds 3 images in parallel, runs tests in each, then pushes to GHCR on main branch only."

**Content**: DAG of docker.yml jobs:
```
build-r (45m)  ──┐
build-test (30m) ├── test-python (30m) ── push-images (main only)
build-full (60m) ┘
```

---

### fig-repo-72: Pre-commit Hook Execution Chain
| Field | Value |
|-------|-------|
| **Title** | Pre-commit: 7 Hooks That Guard Every Commit |
| **Complexity** | L3 |
| **Persona** | ML Engineer / Research Scientist |
| **Priority** | P1 |

**Key Message**: "7 pre-commit hooks run in sequence on every `git commit`. Each catches a specific class of violation. Bypass with `SKIP=hook-name` only when documented."

**Content**: Sequential chain with what each hook catches:
```
git commit
  │
  ├─ 1. ruff (format + lint)
  │     Catches: style violations, unused imports
  │
  ├─ 2. registry-integrity
  │     Catches: method count tampering (11/8/5)
  │
  ├─ 3. registry-validation (pytest)
  │     Catches: registry YAML vs code mismatch
  │
  ├─ 4. r-hardcoding-check
  │     Catches: hex colors, ggsave() in R code
  │
  ├─ 5. computation-decoupling
  │     Catches: sklearn imports in src/viz/
  │
  ├─ 6. extraction-isolation-check
  │     Catches: synthetic data in production paths
  │
  └─ 7. figure-isolation-check
        Catches: synthetic data in figure outputs
```

Known bypass: `SKIP=renv-sync-check` (pre-existing renv failure).

---

## GROUP 3: Code Quality Enforcement (5 figures)

### fig-repo-73: Registry Anti-Cheat System
| Field | Value |
|-------|-------|
| **Title** | 5-Layer Registry Verification: How Method Counts Stay Honest |
| **Complexity** | L4 |
| **Persona** | ML Engineer |
| **Priority** | P2 |

**Key Message**: "Method counts (11/8/5) are verified by 5 independent systems that must ALL agree. Tampering with one triggers failures in the others."

**Content**: 5 concentric rings:
```
Layer 1 (outer): registry_canary.yaml         → Reference values
Layer 2:         mlflow_registry/parameters/   → YAML definitions
Layer 3:         src/data_io/registry.py        → EXPECTED_*_COUNT constants
Layer 4:         tests/test_registry.py         → pytest assertions
Layer 5 (inner): .pre-commit-config.yaml        → Pre-commit hook

ALL 5 MUST AGREE → Change one, update all 5
```

---

### fig-repo-74: Computation Decoupling Enforcement
| Field | Value |
|-------|-------|
| **Title** | The Import Ban: What src/viz/ Cannot Touch |
| **Complexity** | L3 |
| **Persona** | ML Engineer / Research Scientist |
| **Priority** | P1 |

**Key Message**: "src/viz/ files are BANNED from importing sklearn.metrics, scipy.stats, or any computation module. The pre-commit hook scans every .py file in src/viz/ for banned patterns."

**Content**:
```
┌─────────────────────────────┐     ┌─────────────────────────────┐
│  EXTRACTION (Block 1)       │     │  VISUALIZATION (Block 2)    │
│  ┌─────────────────┐        │     │  ┌─────────────────┐        │
│  │ sklearn.metrics  │        │     │  │ ✗ BANNED imports │        │
│  │ scipy.stats      │        │     │  │                  │        │
│  │ src/stats/*      │        │     │  │ ✓ DuckDB SELECT  │        │
│  │ Bootstrap loops  │        │     │  │ ✓ pd.read_sql    │        │
│  └────────┬────────┘        │     │  └────────┬────────┘        │
│           │                  │     │           │                  │
│           ▼                  │     │           ▼                  │
│  ┌─────────────────┐        │     │  ┌─────────────────┐        │
│  │ DuckDB WRITE    │────────│─────│──│ DuckDB READ     │        │
│  │ essential_metrics│        │     │  │ SELECT auroc,   │        │
│  │ calibration_*    │        │     │  │ calibration_*   │        │
│  └─────────────────┘        │     │  └─────────────────┘        │
└─────────────────────────────┘     └─────────────────────────────┘

ENFORCEMENT:
├── Pre-commit: scripts/check_computation_decoupling.py
├── Tests: tests/test_no_hardcoding/test_computation_decoupling.py
└── Rules: .claude/rules/ (for AI agents)
```

---

### fig-repo-75: Hardcoding Prevention Matrix
| Field | Value |
|-------|-------|
| **Title** | Four Types of Hardcoding and How They're Caught |
| **Complexity** | L3 |
| **Persona** | ML Engineer / Research Scientist |
| **Priority** | P2 |

**Key Message**: "4 hardcoding types (colors, paths, methods, dimensions) each have a specific detection mechanism and a correct pattern to follow."

**Content**: 4-column matrix:

| Hardcoding Type | Example Violation | Detection | Correct Pattern |
|-----------------|-------------------|-----------|-----------------|
| Hex colors | `color="#006BA2"` | r-hardcoding-check, test_r_no_hex_colors.py | `COLORS["name"]` / `resolve_color()` |
| Literal paths | `"figures/generated/..."` | test_absolute_paths.py | `save_figure()` / config |
| Method names | `"CatBoost"` in SQL | test_method_abbreviations.py | Load from YAML combos |
| Dimensions | `width=14, height=6` | test_no_hardcoded_values.py | `fig_config.dimensions.*` |

---

### fig-repo-76: Color System Architecture
| Field | Value |
|-------|-------|
| **Title** | From YAML to Pixel: How Colors Are Resolved |
| **Complexity** | L4 |
| **Persona** | ML Engineer |
| **Priority** | P3 |

**Key Message**: "Colors flow from configs/VISUALIZATION/colors.yaml → color_definitions + combo_colors → Python COLORS dict → resolve_color() → matplotlib."

**Content**: Data flow diagram:
```
colors.yaml
├── color_definitions:
│   ├── economist_palette: [#006BA2, #3EBCD2, ...]
│   └── metric_colors: {auroc: "#...", brier: "#..."}
├── combo_colors:
│   ├── ground_truth: {color_ref: "--color-gt"}
│   ├── best_ensemble: {color_ref: "--color-ensemble"}
│   └── ...
│
    ▼
plot_config.py
├── COLORS dict (canonical Python source)
├── resolve_color("--color-ref") → hex
├── get_combo_color("ground_truth") → via YAML color_ref
└── ECONOMIST_PALETTE (fallback)
│
    ▼
matplotlib / R (via colors.yaml cross-reference)
```

---

### fig-repo-77: Data Isolation Gates
| Field | Value |
|-------|-------|
| **Title** | Synthetic vs Production: The Isolation Boundary |
| **Complexity** | L3 |
| **Persona** | ML Engineer / Research Scientist |
| **Priority** | P2 |

**Key Message**: "Synthetic data (for testing) and production data (from MLflow) are isolated by directory, config, pre-commit hooks, and tests. Cross-contamination triggers test failures."

**Content**:
```
┌─────────── PRODUCTION ──────────┐   ┌────────── SYNTHETIC ──────────┐
│ data/public/                     │   │ data/synthetic/                │
│   foundation_plr_results.db     │   │   SYNTH_PLR_DEMO.db           │
│   cd_diagram_data.duckdb        │   │                                │
│                                  │   │ src/synthetic/                 │
│ /home/petteri/mlruns/            │   │   Generators (isolated)       │
│   (410 real MLflow runs)        │   │                                │
│                                  │   │ tests/ use synthetic only     │
│ figures/generated/               │   │ (unless real DB exists)       │
│   (from real data only)         │   │                                │
└──────────────────────────────────┘   └────────────────────────────────┘
                ▲                                    ▲
                │ GATES                              │
                ├── extraction-isolation-check (pre-commit)
                ├── figure-isolation-check (pre-commit)
                ├── data_isolation.yaml (config)
                └── test_synthetic_data.py (pytest)
```

---

## GROUP 4: Data & Configuration Deep Dives (5 figures)

### fig-repo-78: Data Path Resolution
| Field | Value |
|-------|-------|
| **Title** | How Tests Find Their Data: Path Resolution Chain |
| **Complexity** | L4 |
| **Persona** | ML Engineer |
| **Priority** | P2 |

**Key Message**: "Test data paths are resolved through a chain: PROJECT_ROOT → canonical path constants → conftest.py fixtures → skipif decorators. If the file doesn't exist, the test skips gracefully."

**Content**: Flowchart showing:
```
Path(__file__).parent.parent → PROJECT_ROOT
  │
  ├── RESULTS_DB = PROJECT_ROOT / "data/public/foundation_plr_results.db"
  ├── CD_DIAGRAM_DB = PROJECT_ROOT / "data/public/cd_diagram_data.duckdb"
  ├── R_DATA_DIR = PROJECT_ROOT / "data/r_data"
  ├── FIGURES_DIR = PROJECT_ROOT / "figures/generated"
  └── DEMO_DB = PROJECT_ROOT / "data/synthetic/SYNTH_PLR_DEMO.db"
       │
       ▼
  conftest.py: @pytest.fixture(scope="session")
  def results_db_path():
      path = RESULTS_DB
      if not path.exists():
          pytest.skip("Production DB not available")
      return path
```

---

### fig-repo-79: DuckDB Table Relationships
| Field | Value |
|-------|-------|
| **Title** | DuckDB Schema: Tables That Power Every Figure |
| **Complexity** | L3 |
| **Persona** | Research Scientist / ML Engineer |
| **Priority** | P1 |

**Key Message**: "foundation_plr_results.db contains 10+ tables centered around essential_metrics (one row per config). calibration_curves, dca_curves, and predictions link via (outlier_method, imputation_method, classifier)."

**Content**: ER diagram:
```
essential_metrics (316 rows)
├── outlier_method, imputation_method, classifier (PK)
├── auroc, auroc_ci_lower, auroc_ci_upper
├── calibration_slope, calibration_intercept, o_e_ratio
├── brier, scaled_brier
├── net_benefit_5pct, 10pct, 15pct, 20pct
│
├── 1:N → predictions (316 × 208 rows)
│         ├── subject_id, y_true, y_prob
│
├── 1:N → calibration_curves
│         ├── bin_midpoint, observed_freq, predicted_freq
│
├── 1:N → dca_curves
│         ├── threshold, net_benefit, treat_all, treat_none
│
├── 1:N → retention_metrics
│         ├── threshold, metric_value
│
└── 1:N → distribution_stats
          ├── class_label, mean, std, quantiles
```

---

### fig-repo-80: Configuration Space Anatomy
| Field | Value |
|-------|-------|
| **Title** | 11 x 8 x 5 = 440: The Full Configuration Space |
| **Complexity** | L2 |
| **Persona** | Biostatistician / Research Scientist |
| **Priority** | P2 |

**Key Message**: "The full experiment grid is 11 outlier x 8 imputation x 5 classifiers = 440 configs. CatBoost is FIXED for the main analysis (88 configs). The remaining 352 configs exist for validation."

**Content**: 3D grid visualization:
```
          Imputation (8)
         ┌─────────────────────┐
        /│ CSDI SAITS TimesNet │/│
       / │ NuwaTS MissForest  │ │
      /  │ Linear Ground-Truth │ │
     /   └─────────────────────┘ │ Classifier (5)
    /    │                       │ CatBoost ← FIXED
   /     │                       │ XGBoost
  /      │                       │ LogReg
 /       │                       │ TabPFN
/        │                       │ TabM
┌────────────────┐               │
│ Outlier (11)   │               │
│ pupil-gt       │───────────────┘
│ MOMENT-gt-ft   │
│ MOMENT-gt-zs   │
│ UniTS-gt-ft    │
│ TimesNet-gt    │
│ LOF            │
│ OneClassSVM    │
│ PROPHET        │
│ SubPCA         │
│ Ensemble       │
│ EnsThreshold   │
└────────────────┘

Main analysis: 11 × 8 × 1 (CatBoost) = 88 configs
Full grid:     11 × 8 × 5           = 440 configs
In DuckDB:     316 configs (available runs)
```

---

### fig-repo-81: Figure Registry & Combo System
| Field | Value |
|-------|-------|
| **Title** | Figure Registry: From YAML to Generated PNG |
| **Complexity** | L3 |
| **Persona** | ML Engineer |
| **Priority** | P2 |

**Key Message**: "Every figure is defined in figure_registry.yaml with its combo source, privacy level, and output path. plot_hyperparam_combos.yaml provides the standard 4 (or extended 9) combos."

**Content**:
```
figure_registry.yaml                    plot_hyperparam_combos.yaml
┌─────────────────────────┐             ┌─────────────────────────┐
│ R7_calibration:         │             │ standard_combos:        │
│   script: calibration_  │             │   ground_truth:         │
│           plot.py       │──references─│     outlier: pupil-gt   │
│   combos: standard      │             │     imputation: pupil-gt│
│   json_privacy: public  │             │   best_ensemble:        │
│   output: fig_R7_*.png  │             │     outlier: Ensemble   │
└────────┬────────────────┘             │     imputation: CSDI    │
         │                              │   ...                   │
         ▼                              └─────────────────────────┘
generate_all_figures.py
  --figure R7 → loads registry → loads combos → calls script → save_figure()
  --list     → prints all available figure IDs
```

---

### fig-repo-82: Metric Registry API
| Field | Value |
|-------|-------|
| **Title** | MetricRegistry: The Code That Knows Every Metric |
| **Complexity** | L4 |
| **Persona** | ML Engineer |
| **Priority** | P3 |

**Key Message**: "MetricRegistry is a Python class that groups all STRATOS metrics by domain (discrimination, calibration, clinical utility, overall) and provides display names, DuckDB column mappings, and validation."

**Content**: Class diagram:
```
MetricRegistry
├── STRATOS_CORE = ["auroc", "calibration_slope", "brier", ...]
├── DISCRIMINATION = ["auroc"]
├── CALIBRATION = ["calibration_slope", "calibration_intercept", "o_e_ratio"]
├── CLINICAL_UTILITY = ["net_benefit_5pct", ..., "net_benefit_20pct"]
├── OVERALL = ["brier", "scaled_brier"]
│
├── has(name) → bool
├── get_display_name(name) → str
├── get_domain(name) → str
├── get_duckdb_column(name) → str
└── all_metrics() → list
```

---

## GROUP 5: Developer Onboarding (6 figures)

### fig-repo-83: Repository Directory Map
| Field | Value |
|-------|-------|
| **Title** | Repository at a Glance: What's Where |
| **Complexity** | L1 |
| **Persona** | All (first thing a new developer sees) |
| **Priority** | P1 |

**Key Message**: "The repo has 6 main directories: src/ (code), configs/ (settings), tests/ (validation), data/ (databases), figures/ (outputs), and docs/ (documentation)."

**Content**: Annotated tree with color-coded purpose:
```
foundation_PLR/
├── src/                    ← ALL Python source code
│   ├── anomaly_detection/  ← 11 outlier methods
│   ├── classification/     ← 5 classifiers (CatBoost fixed)
│   ├── extraction/         ← MLflow → DuckDB (Block 1)
│   ├── stats/              ← STRATOS metric computation
│   ├── viz/                ← Figure generation (Block 2, READ-ONLY)
│   ├── r/                  ← R figure scripts (ggplot2)
│   ├── orchestration/      ← Prefect flows
│   └── data_io/            ← Registry, data loading
├── configs/                ← ALL configuration (Hydra)
│   ├── mlflow_registry/    ← SINGLE SOURCE OF TRUTH (11/8/5)
│   ├── VISUALIZATION/      ← Figure configs, combos, colors
│   ├── CLS_MODELS/         ← Classifier configs
│   └── OUTLIER_MODELS/     ← Outlier detector configs
├── tests/                  ← 2000+ tests (unit/integration/e2e)
├── data/                   ← DuckDB databases
│   ├── public/             ← Extracted results (checked in)
│   ├── private/            ← Subject lookup (gitignored)
│   └── synthetic/          ← Test-only synthetic data
├── figures/generated/      ← Output figures + JSON sidecars
├── docs/                   ← Documentation
└── .claude/                ← AI agent instructions & rules
```

---

### fig-repo-84: CLAUDE.md Instruction Hierarchy
| Field | Value |
|-------|-------|
| **Title** | CLAUDE.md: How AI Agent Instructions Compose |
| **Complexity** | L3 |
| **Persona** | ML Engineer (who uses Claude Code) |
| **Priority** | P3 |

**Key Message**: "AI instructions follow a 4-level hierarchy: root CLAUDE.md (project overview) → .claude/CLAUDE.md (behavior contract) → .claude/rules/*.md (specific rules) → .claude/domains/*.md (context files loaded on demand)."

**Content**:
```
ALWAYS LOADED (30K chars total)
┌────────────────────────────────┐
│ CLAUDE.md (root, 9K)          │  ← Project overview, data sources, findings
│ .claude/CLAUDE.md (9K)        │  ← Behavior contract, quick reference
│ .claude/rules/*.md (6 files)  │  ← Specific rules (registry, STRATOS, etc.)
│ .claude/auto-context.yaml (2K)│  ← Auto-loaded context selector
└────────────────────────────────┘

LOADED ON DEMAND
┌────────────────────────────────┐
│ .claude/domains/*.md           │  ← MLflow, visualization, manuscript
│ .claude/docs/meta-learnings/   │  ← CRITICAL-FAILURE-001 through 006
│ .claude/planning/              │  ← Active planning docs
└────────────────────────────────┘

PER-DIRECTORY OVERRIDES
┌────────────────────────────────┐
│ docs/repo-figures/CLAUDE.md    │  ← "Repo figures show CODE not RESULTS"
│ (other dirs may have CLAUDE.md)│
└────────────────────────────────┘
```

---

### fig-repo-85: New Developer Quick Start
| Field | Value |
|-------|-------|
| **Title** | From Clone to First Test: 5-Minute Setup |
| **Complexity** | L1 |
| **Persona** | All (new developer) |
| **Priority** | P1 |

**Key Message**: "Clone → run setup script → make test-local → you're contributing. No manual package installation, no configuration, no copy-paste."

**Content**: Linear flowchart:
```
1. git clone → 2. sudo ./scripts/setup-dev-environment.sh
   │                  │
   │                  ├── Installs uv, Python 3.11
   │                  ├── Creates .venv
   │                  ├── Installs all deps (uv sync)
   │                  ├── Installs pre-commit hooks
   │                  └── Verifies R (optional)
   │
   3. make test-local
   │     ├── 2042 passed
   │     ├── 0 failed
   │     └── 181 skipped (need `make extract` + `make analyze`)
   │
   4. (Optional) make extract → make analyze → 0 skips
   │
   5. Ready to contribute!
```

---

### fig-repo-86: How to Add a New Method
| Field | Value |
|-------|-------|
| **Title** | Adding a New Outlier/Imputation Method: Step-by-Step |
| **Complexity** | L3 |
| **Persona** | Research Scientist / ML Engineer |
| **Priority** | P2 |

**Key Message**: "Adding a new method requires changes in 5 places: registry YAML, config file, source module, tests, and re-extraction."

**Content**: Checklist flowchart:
```
Step 1: Registry YAML
  configs/mlflow_registry/parameters/outlier_detection.yaml
  Add entry, increment count (11 → 12)

Step 2: Config File
  configs/OUTLIER_MODELS/NewMethod.yaml
  Define hyperparameters

Step 3: Source Module
  src/anomaly_detection/new_method/
  Implement detect_outliers() interface

Step 4: Tests
  tests/unit/test_new_method.py
  tests/integration/test_anomaly_detection.py

Step 5: Re-extract
  make extract  (Block 1: runs new method through MLflow)
  make analyze  (Block 2: generates updated figures)

Step 6: Update Anti-Cheat
  ├── registry_canary.yaml (count: 12)
  ├── src/data_io/registry.py (EXPECTED_OUTLIER_COUNT = 12)
  ├── tests/test_registry.py (assert count == 12)
  └── .pre-commit-config.yaml (hook args: --expected 12)
```

---

### fig-repo-87: How to Add a New Figure
| Field | Value |
|-------|-------|
| **Title** | Creating a New Figure: Config → Code → QA → Commit |
| **Complexity** | L3 |
| **Persona** | Research Scientist / ML Engineer |
| **Priority** | P2 |

**Key Message**: "Every figure follows the same lifecycle: register in YAML → write plot script → save JSON sidecar → pass QA tests → commit."

**Content**:
```
1. Register in figure_registry.yaml
   ├── ID, script, combos, privacy level
   │
2. Write plot script (src/viz/new_figure.py)
   ├── setup_style()                    ← FIRST call
   ├── combos = load_combos("standard") ← From YAML
   ├── data = db.execute("SELECT...")   ← DuckDB READ ONLY
   ├── COLORS[combo["id"]]             ← Semantic colors
   └── save_figure(fig, "name", data=d) ← JSON sidecar auto-created
   │
3. Generate: python src/viz/generate_all_figures.py --figure NEW
   │
4. QA: pytest tests/test_figure_qa/ -v  ← ZERO TOLERANCE
   ├── P0: Data provenance (no synthetic)
   ├── P1: Statistical validity
   ├── P2: Rendering standards
   └── P3: Accessibility
   │
5. Commit (pre-commit hooks run automatically)
```

---

### fig-repo-88: Makefile Target Map
| Field | Value |
|-------|-------|
| **Title** | 40+ Make Targets: Organized by Purpose |
| **Complexity** | L2 |
| **Persona** | All |
| **Priority** | P2 |

**Key Message**: "Make targets are organized into 7 categories. Start with `make test-local` for development, `make reproduce` for full pipeline, `make figures` for publication."

**Content**: Radial/sunburst map:
```
                    make
                     │
    ┌────────┬───────┼───────┬─────────┐
    │        │       │       │         │
 Pipeline  Figures Testing Docker  Registry
    │        │       │       │         │
 reproduce  figures test-local docker-  verify-
 extract    figure  test-all  build    registry-
 analyze    r-fig   test-fig  docker-  integrity
 verify-    fig-    test-viz  run
 extract    list    test-reg  docker-
            r-val   test-int  test
                    type-chk
```

---

## GROUP 6: R Ecosystem & Advanced Analysis (4 figures)

### fig-repo-89: R Figure System Architecture
| Field | Value |
|-------|-------|
| **Title** | R Figure Pipeline: renv → rocker → ggplot2 |
| **Complexity** | L3 |
| **Persona** | Research Scientist / Biostatistician |
| **Priority** | P2 |

**Key Message**: "R figures use a reproducible pipeline: renv.lock pins exact package versions, rocker Docker images provide the runtime, and src/r/figure_system/ provides helper functions (theme, colors, save)."

**Content**:
```
┌─────────────────────────────────────────────────────────────────┐
│                    R FIGURE SYSTEM                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  renv.lock (pinned)           src/r/figure_system/              │
│  ├── ggplot2 3.5.1            ├── theme_foundation_plr()        │
│  ├── pminternal 0.2.0         ├── load_color_definitions()      │
│  ├── dcurves 0.4.0            ├── save_publication_figure()     │
│  ├── pROC 1.18.5              └── (shared utilities)            │
│  └── data.table 1.15.2                                          │
│                                                                  │
│  ┌───────────────────────────────────────────────────────┐      │
│  │ EXECUTION PATH                                         │      │
│  │ DuckDB → outputs/r_data/*.csv → src/r/figures/*.R     │      │
│  │          (Python exports)        (R reads & plots)     │      │
│  │                                       │                │      │
│  │                              figures/generated/ggplot2/ │      │
│  └───────────────────────────────────────────────────────┘      │
│                                                                  │
│  ENFORCEMENT:                                                    │
│  ├── Pre-commit: r-hardcoding-check (no hex, no ggsave)        │
│  ├── Docker: Dockerfile.r (isolated R environment)              │
│  └── Tests: test_r_figures/ (method validation, hardcoding)     │
└─────────────────────────────────────────────────────────────────┘
```

---

### fig-repo-90: Model Stability Analysis Pipeline
| Field | Value |
|-------|-------|
| **Title** | pminternal: Measuring Model Instability (Riley 2023) |
| **Complexity** | L4 |
| **Persona** | Biostatistician |
| **Priority** | P3 |

**Key Message**: "Model instability is assessed via the pminternal R package (Rhodes 2025), wrapped in Python via subprocess. It produces Optimism-corrected Performance Estimates (OPE) and instability indices."

**Content**:
```
Python wrapper                    R (pminternal)
┌──────────────────┐             ┌──────────────────┐
│ pminternal_      │  subprocess │ library(pminternal)│
│ wrapper.py       │────────────│                    │
│                  │  CSV in/out │ validate()         │
│ Inputs:          │             │ ├── OPE            │
│  y_true, y_prob  │             │ ├── Instability    │
│  B=200 bootstrap │             │ │   index          │
│                  │             │ └── Forest plots    │
│ Outputs:         │             │                    │
│  instability.json│◄───────────│ Results as CSV     │
└──────────────────┘             └──────────────────────┘
│
▼
Figures: R1-R6 (model stability figures)
```

---

### fig-repo-91: Ensemble Method Construction
| Field | Value |
|-------|-------|
| **Title** | How 7 Methods Become One Ensemble |
| **Complexity** | L3 |
| **Persona** | Research Scientist / ML Engineer |
| **Priority** | P3 |

**Key Message**: "The ensemble outlier detector combines 7 base methods (LOF, MOMENT, OneClassSVM, PROPHET, SubPCA, TimesNet, UniTS) via majority voting. The thresholded variant uses only the 4 best foundation models."

**Content**:
```
BASE METHODS                         ENSEMBLE
┌───────────┐                       ┌─────────────────────────┐
│ LOF       │──┐                    │ ensemble-LOF-MOMENT-    │
│ MOMENT    │──┤                    │ OneClassSVM-PROPHET-    │
│ OneClassSVM│──┤  Majority         │ SubPCA-TimesNet-UniTS-  │
│ PROPHET   │──├──Voting ──────────│ gt-finetune             │
│ SubPCA    │──┤                    │                         │
│ TimesNet  │──┤                    │ (7 methods, all)        │
│ UniTS-ft  │──┘                    └─────────────────────────┘

FM-ONLY SUBSET                      THRESHOLDED ENSEMBLE
┌───────────┐                       ┌─────────────────────────┐
│ MOMENT    │──┐                    │ ensembleThresholded-    │
│ TimesNet  │──┤  Threshold         │ MOMENT-TimesNet-UniTS-  │
│ UniTS-ft  │──├──Voting ──────────│ gt-finetune             │
│           │──┘  (≥2 agree)        │                         │
└───────────┘                       │ (4 FM methods only)     │
                                    └─────────────────────────┘
```

---

### fig-repo-92: Feature Definition YAML
| Field | Value |
|-------|-------|
| **Title** | From YAML to Signal: How PLR Features Are Defined |
| **Complexity** | L3 |
| **Persona** | Research Scientist |
| **Priority** | P3 |

**Key Message**: "Each handcrafted feature is defined in featuresBaseline.yaml with time windows (time_from, time_start, time_end) and statistics (mean, min, slope). This maps directly to PLR signal segments."

**Content**: Show a PLR signal with time windows overlaid:
```
PLR Signal
│        ╱╲
│       ╱  ╲         ╱────────────────────
│      ╱    ╲       ╱
│     ╱      ╲     ╱
│    ╱        ╲   ╱
│───╱──────────╲─╱──────────────────────── time
│   │ BASELINE │ │  RECOVERY              │
│   0    0.5   1.0  1.5   2.0        5.0s │

featuresBaseline.yaml:
  constriction_amplitude:
    time_from: stimulus_onset    ← config, not hardcoded
    time_start: 0.0
    time_end: 1.5
    stat: min                    ← minimum value in window

  recovery_75:
    time_from: minimum_point
    time_start: 0.0
    time_end: 5.0
    stat: percentile_75          ← 75th percentile recovery
```

---

## GROUP 7: Debugging & Troubleshooting (4 figures)

### fig-repo-93: Common Test Failures & Fixes
| Field | Value |
|-------|-------|
| **Title** | My Tests Failed: A Diagnostic Flowchart |
| **Complexity** | L2 |
| **Persona** | All |
| **Priority** | P1 |

**Key Message**: "Most test failures fall into 5 categories: missing data (run make extract), import errors (run uv sync), pre-commit failures (fix and re-commit), R not installed, or genuine bugs."

**Content**: Decision tree:
```
Tests Failed
  │
  ├── "FileNotFoundError" or "pytest.skip"?
  │   └── Missing data → make extract && make analyze
  │
  ├── "ModuleNotFoundError"?
  │   └── Missing package → uv sync
  │
  ├── "Pre-commit hook failed"?
  │   ├── ruff → fix style, re-stage
  │   ├── registry-integrity → update all 5 layers
  │   ├── computation-decoupling → remove banned import
  │   └── r-hardcoding → use load_color_definitions()
  │
  ├── "R not found" or "Rscript error"?
  │   └── R not installed → install R ≥ 4.4 from CRAN
  │       or use: make r-docker-test
  │
  └── Genuine test failure?
      └── Read assertion message, fix code, re-run
```

---

### fig-repo-94: Extraction Guardrails
| Field | Value |
|-------|-------|
| **Title** | Guardrails: Memory, Disk, and Stall Protection |
| **Complexity** | L4 |
| **Persona** | ML Engineer |
| **Priority** | P3 |

**Key Message**: "The extraction pipeline (MLflow → DuckDB) processes 410 runs with 1000 bootstraps each. Three guardrails prevent resource exhaustion: memory monitor, disk monitor, and stall detector."

**Content**:
```
ExtractionGuardrails
├── MemoryMonitor
│   ├── Threshold: 85% system RAM
│   ├── Action: GC + reduce batch size
│   └── Fallback: Abort with checkpoint
│
├── DiskMonitor
│   ├── Threshold: 90% disk usage
│   ├── Action: Warn + compact DB
│   └── Fallback: Abort with checkpoint
│
└── StallDetector
    ├── Heartbeat: Every 60 seconds per run
    ├── Timeout: 300 seconds no progress
    └── Action: Skip run + log warning

CheckpointManager
├── Saves progress every N runs
├── Resumes from last checkpoint
└── make reproduce-from-checkpoint (skip extraction)
```

---

### fig-repo-95: Quality Gate Decision Flow
| Field | Value |
|-------|-------|
| **Title** | When a Pre-commit Hook Fails: Fix, Stage, Commit (Never Amend) |
| **Complexity** | L2 |
| **Persona** | All |
| **Priority** | P2 |

**Key Message**: "When a pre-commit hook fails, the commit did NOT happen. Fix the issue, re-stage with git add, then create a NEW commit. Never use --amend (it modifies the previous commit)."

**Content**:
```
git commit -m "my changes"
  │
  ├── Hook passes → Commit created ✓
  │
  └── Hook FAILS → Commit NOT created ✗
      │
      ├── Read error message
      │   ├── "ruff" → auto-fixed, just re-stage
      │   ├── "registry" → update count in 5 places
      │   ├── "decoupling" → remove banned import
      │   └── "r-hardcoding" → use YAML colors
      │
      ├── Fix the issue
      │
      ├── git add <fixed files>     ← RE-STAGE
      │
      └── git commit -m "my changes" ← NEW COMMIT (not --amend!)
          │
          └── Why not --amend?
              └── The failed commit doesn't exist.
                  --amend would modify the PREVIOUS commit,
                  potentially destroying unrelated work.
```

---

### fig-repo-96: Selective Classification & Uncertainty
| Field | Value |
|-------|-------|
| **Title** | Abstain When Uncertain: Risk-Coverage Analysis |
| **Complexity** | L3 |
| **Persona** | Biostatistician / Research Scientist |
| **Priority** | P3 |

**Key Message**: "Selective classification allows the model to abstain on uncertain predictions. AURC (Area Under Risk-Coverage) and Risk-Coverage plots show how error rate drops as the model rejects more uncertain cases."

**Content**: Conceptual diagram (no performance numbers):
```
┌─────────────────────────────────────────────────────────────────┐
│  SELECTIVE CLASSIFICATION CONCEPT                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  All 208 subjects                                                │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● │   │
│  │ Predictions sorted by confidence (highest → lowest)       │   │
│  └──────┬─────────────────────────────────────────┬──────────┘   │
│         │ ACCEPT (high confidence)                 │ REJECT       │
│         │                                          │ (uncertain)  │
│         ▼                                          ▼              │
│  ┌──────────────────┐                    ┌──────────────────┐   │
│  │ Risk decreases   │                    │ "I don't know"   │   │
│  │ as coverage      │                    │ → refer to       │   │
│  │ decreases        │                    │   specialist     │   │
│  └──────────────────┘                    └──────────────────┘   │
│                                                                  │
│  AURC = Area Under Risk-Coverage curve                          │
│  Lower AURC = better selective classification                    │
│                                                                  │
│  Code: src/stats/decision_uncertainty.py                        │
│  Config: Threshold sweep from 0.0 to 1.0                        │
│  References: Barrenada 2025, Geifman & El-Yaniv 2017            │
└─────────────────────────────────────────────────────────────────┘
```

---

### fig-repo-97: Critical Failure Meta-Learnings Timeline
| Field | Value |
|-------|-------|
| **Title** | 6 Critical Failures That Shaped Our Guardrails |
| **Complexity** | L2 |
| **Persona** | All |
| **Priority** | P2 |

**Key Message**: "Each CRITICAL-FAILURE (001-006) led to a specific guardrail being added. The codebase's strictness comes from hard-won lessons."

**Content**: Timeline/waterfall:
```
CRITICAL-FAILURE-001: Synthetic Data in Figures
  └── GUARDRAIL: test_data_provenance.py, figure-isolation-check

CRITICAL-FAILURE-002: Mixed Featurization in Extraction
  └── GUARDRAIL: Registry validation, extraction-isolation-check

CRITICAL-FAILURE-003: Computation in Visualization
  └── GUARDRAIL: computation-decoupling hook, import ban

CRITICAL-FAILURE-004: Hardcoded Values Everywhere
  └── GUARDRAIL: r-hardcoding-check, test_no_hardcoded_values.py

CRITICAL-FAILURE-005: (Visual Bug Priority)
  └── GUARDRAIL: Bug-First rule, verify output with Read tool

CRITICAL-FAILURE-006: Shortcuts in Academic Code
  └── GUARDRAIL: Pre-implementation checklist, code review
```

---

### fig-repo-98: JSON Sidecar Pattern
| Field | Value |
|-------|-------|
| **Title** | JSON Sidecars: Every Figure's Reproducibility Passport |
| **Complexity** | L3 |
| **Persona** | Research Scientist / ML Engineer |
| **Priority** | P2 |

**Key Message**: "Every generated figure has a companion JSON file containing all numeric data, source hashes, and generation metadata. This enables figure reproduction without re-running the full pipeline."

**Content**:
```
figures/generated/
├── fig_R7_calibration.png         ← The figure
├── fig_R7_calibration.json        ← The sidecar
│   {
│     "figure_id": "R7",
│     "generated_at": "2026-02-06T12:00:00",
│     "source_db": "data/public/foundation_plr_results.db",
│     "source_hash": "sha256:abc123...",
│     "combos_used": ["ground_truth", "best_ensemble", ...],
│     "data": {
│       "ground_truth": {
│         "auroc": 0.911,
│         "calibration_slope": 1.02,
│         "calibration_intercept": -0.03,
│         "curve_points": [[0.0, 0.0], [0.1, 0.08], ...]
│       },
│       ...
│     },
│     "summary_statistics": {
│       "n_subjects": 208,
│       "n_bootstrap": 1000
│     }
│   }
│
└── data/fig_R7_calibration_TEST.json  ← Subject-level (PRIVATE, gitignored)
```

---

## Summary: 38 New Figures by Group

| Group | Topic | Figures | IDs | Priority |
|-------|-------|---------|-----|----------|
| 1 | **Testing Architecture** | 7 | 61-67 | P1 (user-requested) |
| 2 | **CI/CD & Docker** | 5 | 68-72 | P1 (user-requested) |
| 3 | **Code Quality Enforcement** | 5 | 73-77 | P1-P2 |
| 4 | **Data & Config Deep Dives** | 5 | 78-82 | P1-P3 |
| 5 | **Developer Onboarding** | 6 | 83-88 | P1-P2 |
| 6 | **R & Advanced Analysis** | 4 | 89-92 | P2-P3 |
| 7 | **Debugging & Troubleshooting** | 6 | 93-98 | P1-P3 |
| **TOTAL** | | **38** | 61-98 | |

### Suggested Generation Order (by impact)

**Phase 1 - Core Developer Needs (12 figures)**:
fig-repo-61 (Test Skip Groups), fig-repo-62 (Test→CI Mapping), fig-repo-67 (Local/Docker/CI), fig-repo-68 (CI Pipeline), fig-repo-69 (Docker Images), fig-repo-72 (Pre-commit Chain), fig-repo-74 (Computation Decoupling), fig-repo-79 (DuckDB Schema), fig-repo-83 (Directory Map), fig-repo-85 (Quick Start), fig-repo-93 (Test Failures), fig-repo-95 (Hook Failure Flow)

**Phase 2 - Deep Understanding (14 figures)**:
fig-repo-63 (Markers), fig-repo-65 (Figure QA), fig-repo-66 (Test Data), fig-repo-70 (Docker Compose), fig-repo-73 (Anti-Cheat), fig-repo-75 (Hardcoding), fig-repo-77 (Data Isolation), fig-repo-78 (Data Paths), fig-repo-80 (Config Space), fig-repo-81 (Figure Registry), fig-repo-86 (Add Method), fig-repo-87 (Add Figure), fig-repo-88 (Makefile Map), fig-repo-97 (Critical Failures)

**Phase 3 - Specialist Topics (12 figures)**:
fig-repo-64 (Fixtures), fig-repo-71 (Docker Build), fig-repo-76 (Colors), fig-repo-82 (Metric Registry), fig-repo-84 (CLAUDE.md), fig-repo-89 (R System), fig-repo-90 (pminternal), fig-repo-91 (Ensemble), fig-repo-92 (Features YAML), fig-repo-94 (Guardrails), fig-repo-96 (Selective Classification), fig-repo-98 (JSON Sidecars)
