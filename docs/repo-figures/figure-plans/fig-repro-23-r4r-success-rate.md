# fig-repro-23: The 97.5% R4R Success Rate

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repro-23 |
| **Title** | The 97.5% R4R Success Rate |
| **Complexity Level** | L2 (Intermediate) |
| **Target Persona** | Biostatistician, R User |
| **Location** | docs/reproducibility-guide.md |
| **Priority** | P2 |
| **Aspect Ratio** | 16:10 |

## Purpose

Celebrate the R4R achievement as a model for what's possible with proper tooling, and connect to Foundation PLR's approach.

## Key Message

"R4R improved R package reproducibility from 26% to 97.5%—a 3.75x improvement. This proves reproducibility is a tooling problem with tooling solutions."

## Literature Sources

| Source | Finding | DOI/URL |
|--------|---------|---------|
| R4R (Donat-Bouillud et al. 2025) | 26% → 97.5% success rate | [10.1145/3736731.3746156](https://doi.org/10.1145/3736731.3746156) |

## Visual Concept

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    THE 97.5% R4R SUCCESS RATE                                   │
│                    From 26% to 97.5%: A tooling triumph                         │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  THE TRANSFORMATION                                                             │
│  ═══════════════════                                                            │
│                                                                                 │
│  WITHOUT R4R                              WITH R4R                              │
│  ──────────────                           ──────────                            │
│                                                                                 │
│  2,000 R packages tested                  Same 2,000 packages                   │
│                                                                                 │
│  ┌────────────────────────┐               ┌────────────────────────┐            │
│  │████                    │ 26%           │████████████████████████│ 97.5%     │
│  │████ SUCCEED            │               │██████████████ SUCCEED  │            │
│  └────────────────────────┘               └────────────────────────┘            │
│  ┌────────────────────────┐               ┌────────────────────────┐            │
│  │████████████████████████│ 74%           │█                       │ 2.5%      │
│  │████████████████ FAIL   │               │█ FAIL                  │            │
│  └────────────────────────┘               └────────────────────────┘            │
│                                                                                 │
│  520 packages ran                         1,950 packages ran                    │
│                                           +1,430 rescued! (3.75x improvement)   │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  HOW R4R ACHIEVES THIS                                                          │
│  ═════════════════════                                                          │
│                                                                                 │
│  R4R automatically:                                                             │
│                                                                                 │
│  1. TRACES DEPENDENCIES                                                         │
│     Analyzes R code to find all packages used                                   │
│     (not just what's in DESCRIPTION)                                            │
│                                                                                 │
│  2. CAPTURES VERSIONS                                                           │
│     Records exact versions from CRAN snapshot                                   │
│     → renv.lock with all transitive deps                                        │
│                                                                                 │
│  3. CREATES CONTAINER                                                           │
│     Generates Dockerfile with R + packages                                      │
│     → Isolated, reproducible environment                                        │
│                                                                                 │
│  4. TESTS EXECUTION                                                             │
│     Runs code in container to verify                                            │
│     → Catches failures before publication                                       │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  LESSON FOR ALL SCIENTIFIC SOFTWARE                                             │
│  ═══════════════════════════════════                                            │
│                                                                                 │
│  R4R proves:                                                                    │
│                                                                                 │
│  "Reproducibility is not a researcher discipline problem.                       │
│   It is a tooling problem with tooling solutions."                              │
│                                                                                 │
│  The same researchers, the same code—just better tools.                         │
│  → 3.75x improvement in reproducibility.                                        │
│                                                                                 │
│  Foundation PLR applies this philosophy:                                        │
│  • Automatic dependency locking (uv.lock, not requirements.txt)                 │
│  • Single-command reproduction (make reproduce, not 10-step README)             │
│  • Validated artifacts (JSON sidecars, not trust-me figures)                    │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  FOUNDATION PLR R SCRIPTS                                                       │
│  ═════════════════════════                                                      │
│                                                                                 │
│  We use R for specific statistical packages:                                    │
│  • pminternal (Riley 2023) - Model instability analysis                         │
│  • ggplot2 - Publication-quality figures                                        │
│                                                                                 │
│  Our R reproducibility:                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  renv.lock  →  renv::restore()  →  Exact R packages                     │   │
│  │  DuckDB     →  R reads same .db  →  Identical data                      │   │
│  │  make r-figures  →  Regenerate all R outputs                            │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Content Elements

1. **Before/after comparison**: 26% vs 97.5% with bar charts
2. **Improvement calculation**: +1,430 rescued, 3.75x factor
3. **How R4R works**: Four steps explanation
4. **Key lesson**: Tooling problem, tooling solution quote
5. **Foundation PLR R usage**: Our approach to R reproducibility

## Text Content

### Title Text
"R4R: From 26% to 97.5% Reproducibility"

### Caption
R4R (Donat-Bouillud et al. 2025) transformed R package reproducibility from 26% to 97.5%—rescuing 1,430 of 2,000 packages (3.75x improvement). Same researchers, same code, better tools. Key insight: reproducibility is a tooling problem with tooling solutions. Foundation PLR applies this: uv.lock for Python, renv.lock for R, make reproduce for one-command verification.

## Prompts for Nano Banana Pro

### Style Prompt
Before/after bars with dramatic improvement. Four-step process diagram. Quote callout box. Foundation PLR R workflow. Celebratory but data-driven style.

### Content Prompt
Create "R4R 97.5% Success" infographic:

**TOP - Before/After**:
- Two bar comparisons: 26% → 97.5%
- "+1,430 rescued, 3.75x improvement"

**MIDDLE - How**:
- Four steps: trace → capture → containerize → test

**BOTTOM - Lesson**:
- Quote: "Tooling problem with tooling solutions"
- Foundation PLR R workflow

## Alt Text

R4R success rate infographic. Before R4R: 26% (520/2000) R packages succeed. After R4R: 97.5% (1950/2000) succeed—1430 packages rescued, 3.75x improvement. How R4R works: traces dependencies, captures exact versions in renv.lock, creates Docker container, tests execution. Key lesson: "Reproducibility is not a researcher discipline problem. It is a tooling problem with tooling solutions." Foundation PLR R approach: renv.lock for packages, DuckDB for data, make r-figures for regeneration.

## Status

- [x] Draft created
- [ ] Generated
- [ ] Placed in docs/reproducibility-guide.md

