# fig-repro-09: The R Package Ecosystem Challenge

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repro-09 |
| **Title** | The R Package Ecosystem Challenge |
| **Complexity Level** | L2 (Intermediate) |
| **Target Persona** | Biostatistician, R User |
| **Location** | docs/reproducibility-guide.md |
| **Priority** | P1 |
| **Aspect Ratio** | 16:10 |

## Purpose

Present R4R study findings on R package reproducibility and explain why Foundation PLR uses renv for R dependencies.

## Key Message

"Only 26% of R replication packages run successfully. R4R tool achieves 97.5% by automatically fixing dependencies. Foundation PLR uses renv for R lockfiles alongside uv for Python."

## Literature Sources

| Source | Finding | DOI/URL |
|--------|---------|---------|
| R4R (Donat-Bouillud et al. 2025) | 26% success rate, 97.5% with r4r | [10.1145/3736731.3746156](https://doi.org/10.1145/3736731.3746156) |

## Visual Concept

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    THE R PACKAGE ECOSYSTEM CHALLENGE                            │
│                    R4R 2025 (ACM REP '25)                                       │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  R REPLICATION PACKAGES: THE STATE OF THE ART                                   │
│  ════════════════════════════════════════════                                   │
│                                                                                 │
│  Study: 2,000 R replication packages from published papers                      │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                         │   │
│  │  WITHOUT R4R TOOL                      WITH R4R TOOL                    │   │
│  │  ─────────────────                     ─────────────────                │   │
│  │                                                                         │   │
│  │  74% FAIL  ████████████████████████    2.5% FAIL  █                     │   │
│  │  26% RUN   ████████                    97.5% RUN  ████████████████████  │   │
│  │                                                                         │   │
│  │  Only 520 of 2,000 packages            1,950 of 2,000 packages          │   │
│  │  completed execution                   completed execution               │   │
│  │                                                                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  WHY R PACKAGES FAIL                                                            │
│  ═══════════════════                                                            │
│                                                                                 │
│  │ Failure Type                    │ % of Failures │ R4R Fix         │        │
│  │ ─────────────────────────────── │ ───────────── │ ─────────────── │        │
│  │ Missing dependencies            │     45%       │ Auto-install    │        │
│  │ Version incompatibility         │     25%       │ Snapshot        │        │
│  │ Missing system libraries        │     15%       │ Docker image    │        │
│  │ Path issues                     │     10%       │ Containerize    │        │
│  │ Other (syntax, data)            │      5%       │ Manual          │        │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  R's UNIQUE CHALLENGES                                                          │
│  ═════════════════════                                                          │
│                                                                                 │
│  1. NO NATIVE LOCKFILE (historically)                                           │
│     Python: requirements.txt / pyproject.toml                                   │
│     R: install.packages("X") → whatever is current                              │
│                                                                                 │
│  2. BINARY vs SOURCE COMPLEXITY                                                 │
│     Windows: binary packages                                                    │
│     Linux: compile from source (needs system libs!)                             │
│                                                                                 │
│  3. CRAN SNAPSHOTS EXPIRE                                                       │
│     Old package versions removed from CRAN                                      │
│     install.packages("X", version="1.0") → not found                            │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  FOUNDATION PLR: DUAL LOCKFILE APPROACH                                         │
│  ══════════════════════════════════════                                         │
│                                                                                 │
│  Python:  pyproject.toml  →  uv.lock  →  uv sync                                │
│  R:       DESCRIPTION     →  renv.lock  →  renv::restore()                      │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  renv.lock (example):                                                   │   │
│  │  {                                                                      │   │
│  │    "R": { "Version": "4.4.0" },                                         │   │
│  │    "Packages": {                                                        │   │
│  │      "pminternal": {                                                    │   │
│  │        "Package": "pminternal",                                         │   │
│  │        "Version": "0.1.0",                                              │   │
│  │        "Source": "Repository",                                          │   │
│  │        "Repository": "CRAN"                                             │   │
│  │      },                                                                 │   │
│  │      "ggplot2": { ... }                                                 │   │
│  │    }                                                                    │   │
│  │  }                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  Why renv: Captures exact R + package versions in a lockfile                    │
│  Used for: pminternal (Riley 2023 instability analysis)                         │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Content Elements

1. **Before/after bar charts**: 26% vs 97.5% success rates
2. **Failure taxonomy table**: Why packages fail and how r4r fixes
3. **R-specific challenges**: Three unique problems
4. **Dual lockfile diagram**: uv.lock for Python, renv.lock for R
5. **renv.lock example**: Annotated JSON structure

## Text Content

### Title Text
"The R Package Ecosystem: Only 26% Run Successfully"

### Caption
R replication packages have a 74% failure rate—only 520 of 2,000 packages run successfully (R4R 2025, [DOI](https://doi.org/10.1145/3736731.3746156)). The R4R tool achieves 97.5% success by auto-installing dependencies and using snapshots. Foundation PLR uses a dual lockfile approach: uv.lock for Python (200+ packages), renv.lock for R (pminternal and visualization packages).

## Prompts for Nano Banana Pro

### Style Prompt
Split comparison: before r4r (red, 26%) vs after r4r (green, 97.5%). Failure taxonomy table. Dual lockfile flow diagram showing Python and R parallel paths. renv.lock code block. Professional, technical style.

### Content Prompt
Create "R Package Ecosystem" infographic:

**TOP - Before/After**:
- Two bars: 26% success without tool, 97.5% with r4r
- Sample sizes: 520/2000 vs 1950/2000

**MIDDLE - Failure Table**:
- Missing deps (45%), version (25%), system libs (15%), paths (10%), other (5%)

**BOTTOM - Dual Lockfile**:
- Python path: pyproject.toml → uv.lock → uv sync
- R path: DESCRIPTION → renv.lock → renv::restore()
- renv.lock JSON example

## Alt Text

R package ecosystem reproducibility infographic. Before r4r: only 26% (520/2000) packages run successfully. After r4r: 97.5% (1950/2000) succeed. Failure breakdown: missing dependencies 45%, version incompatibility 25%, system libraries 15%, path issues 10%, other 5%. Foundation PLR uses dual lockfile approach: uv.lock for Python, renv.lock for R. renv.lock example shows R version and package specifications.

## Status

- [x] Draft created
- [ ] Generated
- [ ] Placed in docs/reproducibility-guide.md

