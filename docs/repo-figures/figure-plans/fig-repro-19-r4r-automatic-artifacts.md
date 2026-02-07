# fig-repro-19: R4R: Automatic Artifact Creation

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repro-19 |
| **Title** | R4R: Automatic Artifact Creation |
| **Complexity Level** | L3 (Expert) |
| **Target Persona** | Biostatistician, R User |
| **Location** | docs/reproducibility-guide.md |
| **Priority** | P3 |
| **Aspect Ratio** | 16:10 |

## Purpose

Present the R4R tool's approach to automatic reproducibility artifact generation and discuss potential future integration.

## Key Message

"R4R achieves 97.5% reproducibility by automatically generating containers, lockfiles, and data snapshots. This is the future of scientific computing—automate reproducibility, don't bolt it on."

## Literature Sources

| Source | Finding | DOI/URL |
|--------|---------|---------|
| R4R (Donat-Bouillud et al. 2025) | 97.5% success rate with automatic artifact generation | [10.1145/3736731.3746156](https://doi.org/10.1145/3736731.3746156) |

## Visual Concept

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    R4R: AUTOMATIC ARTIFACT CREATION                             │
│                    From 26% to 97.5% reproducibility                            │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  THE R4R APPROACH                                                               │
│  ════════════════                                                               │
│                                                                                 │
│  Traditional workflow:                                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                         │   │
│  │  Write R Code → Forget to save dependencies → Submit paper              │   │
│  │        │                                              │                 │   │
│  │        └──────────────────────────────────────────────┘                 │   │
│  │                                                                         │   │
│  │  2 years later: "I need to reproduce this..."                           │   │
│  │        │                                                                │   │
│  │        ▼                                                                │   │
│  │  ❌ install.packages("X") fails                                         │   │
│  │  ❌ Package removed from CRAN                                           │   │
│  │  ❌ Version conflict                                                    │   │
│  │                                                                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  R4R workflow:                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                         │   │
│  │  Write R Code → R4R analyzes → Generates artifacts → Submit paper       │   │
│  │        │              │                   │                 │           │   │
│  │        │              ▼                   ▼                 │           │   │
│  │        │        ┌──────────┐      ┌─────────────┐           │           │   │
│  │        │        │ Deps     │      │ renv.lock   │           │           │   │
│  │        │        │ traced   │      │ Dockerfile  │           │           │   │
│  │        │        └──────────┘      │ Data hash   │           │           │   │
│  │        │                          └─────────────┘           │           │   │
│  │        │                                                    │           │   │
│  │        └──────────────────── AUTOMATICALLY ─────────────────┘           │   │
│  │                                                                         │   │
│  │  2 years later: "I need to reproduce this..."                           │   │
│  │        │                                                                │   │
│  │        ▼                                                                │   │
│  │  ✅ Artifacts recreate exact environment                                │   │
│  │  ✅ 97.5% success rate!                                                 │   │
│  │                                                                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  R4R GENERATED ARTIFACTS                                                        │
│  ═══════════════════════                                                        │
│                                                                                 │
│  │ Artifact             │ Contents                        │ Purpose         │  │
│  │ ──────────────────── │ ─────────────────────────────── │ ─────────────── │  │
│  │ renv.lock            │ All R packages + versions       │ R dependencies  │  │
│  │ Dockerfile           │ Container with R, packages      │ Full isolation  │  │
│  │ .Rprofile            │ Library paths, options          │ Configuration   │  │
│  │ MANIFEST.csv         │ Data file hashes                │ Data integrity  │  │
│  │ run.sh               │ Exact execution command         │ Invocation      │  │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  FOUNDATION PLR: SIMILAR PHILOSOPHY FOR PYTHON                                  │
│  ═════════════════════════════════════════════                                  │
│                                                                                 │
│  We apply R4R's principles to Python:                                           │
│                                                                                 │
│  │ R4R Artifact         │ Foundation PLR Equivalent      │                     │
│  │ ──────────────────── │ ────────────────────────────── │                     │
│  │ renv.lock            │ uv.lock                        │                     │
│  │ Dockerfile           │ Dockerfile (pinned digests)    │                     │
│  │ MANIFEST.csv         │ JSON sidecars with data hashes │                     │
│  │ run.sh               │ make reproduce                 │                     │
│                                                                                 │
│  Future: Automatic artifact generation via pre-commit hooks                     │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  git commit → pre-commit hook → Update uv.lock                          │   │
│  │                              → Update JSON sidecars                     │   │
│  │                              → Verify reproducibility                   │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Content Elements

1. **Traditional vs R4R workflow**: Two flow diagrams
2. **Artifact table**: What R4R generates and why
3. **Foundation PLR mapping**: R4R artifacts → our equivalents
4. **Future vision**: Pre-commit hooks for automatic updates

## Text Content

### Title Text
"R4R: Automatic Reproducibility Artifacts (97.5% Success)"

### Caption
R4R achieves 97.5% reproducibility (vs 26% baseline) by automatically generating renv.lock, Dockerfile, data manifests, and run scripts at publication time (Donat-Bouillud et al. 2025, [DOI](https://doi.org/10.1145/3736731.3746156)). Foundation PLR applies similar principles: uv.lock for Python, JSON sidecars for data hashes, and make reproduce for invocation. Future: pre-commit hooks for automatic artifact updates.

## Prompts for Nano Banana Pro

### Style Prompt
Two flow diagrams comparing traditional (linear, failing) vs R4R (branching, generating artifacts). Artifact table. Mapping table showing R4R → Foundation PLR equivalents. Future vision in a callout box. Technical, aspirational style.

### Content Prompt
Create "R4R Automatic Artifacts" infographic:

**TOP - Traditional vs R4R**:
- Traditional: linear path to failure (2 years later)
- R4R: automatic artifact generation during development

**MIDDLE - Artifacts Table**:
- 5 artifacts: renv.lock, Dockerfile, .Rprofile, MANIFEST.csv, run.sh

**BOTTOM - Foundation PLR**:
- Mapping table: R4R artifact → our equivalent
- Future vision: pre-commit hook automation

## Alt Text

R4R automatic artifact creation infographic. Traditional workflow: write R code, forget dependencies, submit paper, 2 years later fails with missing packages (26% success). R4R workflow: write code, R4R traces dependencies, generates renv.lock + Dockerfile + data manifest + run.sh automatically, 97.5% success. Artifacts: renv.lock (R packages), Dockerfile (isolation), .Rprofile (config), MANIFEST.csv (data hashes), run.sh (invocation). Foundation PLR equivalents: uv.lock, Dockerfile, JSON sidecars, make reproduce. Future: pre-commit hooks for automatic artifact updates.

## Status

- [x] Draft created
- [ ] Generated
- [ ] Placed in docs/reproducibility-guide.md

