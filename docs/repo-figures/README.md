# Repository Infographics Suite

This directory contains figure plans for repository documentation infographics, designed for generation with **Nano Banana Pro** (or fallback tools).

## Quick Start

1. Choose a figure from `figure-plans/`
2. Copy the **Style Prompt** and **Content Prompt** to Nano Banana Pro
3. Generate the infographic
4. Save to `generated/fig-repo-{NN}-{name}.png`
5. Embed in the target README/docs location

## Figure Catalog

| ID | Title | Persona | Priority | Status |
|----|-------|---------|----------|--------|
| **fig-repo-01** | What Does This Repo Do? | All | P1 | Draft |
| **fig-repo-02** | Preprocessing Pipeline | All | P1 | Draft |
| **fig-repo-03** | Why Foundation Models? | PI/Stats | P2 | Draft |
| **fig-repo-04** | MLflow = Lab Notebook | PI/Stats/Sci | P2 | Draft |
| **fig-repo-05** | Hydra Config System | Sci/ML | P2 | Draft |
| **fig-repo-06** | Add New Classifier | Sci/ML | P3 | Draft |
| **fig-repo-07** | Create Figure Workflow | Sci/ML/Agent | P1 | Draft |
| **fig-repo-08** | Pre-commit Quality Gates | Sci/ML | P3 | Draft |
| **fig-repo-09** | TDD Workflow | Sci/ML | P3 | Draft |
| **fig-repo-10** | Prefect Orchestration | ML | P3 | Draft |
| **fig-repo-11** | STRATOS Metrics | Stats/Sci | P2 | Draft |
| **fig-repo-12** | Future Experiments | All | P3 | Draft |
| **fig-repo-51** | Classifier Config Architecture | ML | P2 | Plan |
| **fig-repo-52** | Classifier Paradigms | Stats/ML | P2 | Plan |
| **fig-repo-53** | Outlier Detection Methods | Sci/ML | P2 | Plan |
| **fig-repo-54** | Imputation Model Landscape | Sci/ML | P2 | Plan |
| **fig-repo-55** | Registry Single Source of Truth | ML/Agent | P1 | Generated |
| **fig-repo-56** | Experiment Config Hierarchy | ML | P2 | Generated |
| **fig-repo-57** | DuckDB Schema Diagram | ML/Sci | P1 | Plan |
| **fig-repo-58** | Pre-Commit Enforcement Matrix | ML/Sci | P1 | Plan |
| **fig-repo-59** | STRATOS Computation Flow | Stats/Sci | P1 | Plan |
| **fig-repo-60** | Test Pyramid Architecture | ML/Sci | P1 | Plan |
| **fig-repo-61** | Test Skip Groups A-H | ML/Sci | P1 | Generated |
| **fig-repo-62** | Test Tier → CI Job Mapping | ML | P1 | Generated |
| **fig-repo-63** | pytest Marker System | ML/Sci | P1 | Generated |
| **fig-repo-64** | Test Fixture Architecture | ML | P2 | Generated |
| **fig-repo-65** | Figure QA Test Categories | Sci/ML | P1 | Generated |
| **fig-repo-66** | Test Data Flow | ML | P2 | Generated |
| **fig-repo-67** | Local vs Docker vs CI Tests | All | P1 | Generated |
| **fig-repo-68** | GitHub Actions CI Pipeline | ML | P1 | Generated |
| **fig-repo-69** | Docker Multi-Image Architecture | ML | P1 | Generated |
| **fig-repo-70** | Docker Compose Service Map | ML | P2 | Generated |
| **fig-repo-71** | Docker Build Pipeline (CI) | ML | P2 | Generated |
| **fig-repo-72** | Pre-commit Hook Chain | ML/Sci | P1 | Generated |
| **fig-repo-73** | Registry Anti-Cheat System | ML | P2 | Generated |
| **fig-repo-74** | Computation Decoupling Enforcement | ML/Sci | P1 | Generated |
| **fig-repo-75** | Hardcoding Prevention Matrix | ML/Sci | P2 | Generated |
| **fig-repo-76** | Color System Architecture | ML | P3 | Generated |
| **fig-repo-77** | Data Isolation Gates | ML/Sci | P2 | Generated |
| **fig-repo-78** | Data Path Resolution | ML | P2 | Generated |
| **fig-repo-79** | DuckDB Table Relationships | Sci/ML | P1 | Generated |
| **fig-repo-80** | Configuration Space Anatomy | Stats/Sci | P2 | Generated |
| **fig-repo-81** | Figure Registry & Combo System | ML | P2 | Generated |
| **fig-repo-82** | Metric Registry API | ML | P3 | Generated |
| **fig-repo-83** | Repository Directory Map | All | P1 | Generated |
| **fig-repo-84** | CLAUDE.md Instruction Hierarchy | ML/Agent | P3 | Generated |
| **fig-repo-85** | New Developer Quick Start | All | P1 | Generated |
| **fig-repo-86** | How to Add a New Method | Sci/ML | P2 | Generated |
| **fig-repo-87** | How to Add a New Figure | Sci/ML | P2 | Generated |
| **fig-repo-88** | Makefile Target Map | All | P2 | Generated |
| **fig-repo-89** | R Figure System Architecture | Sci/Stats | P2 | Generated |
| **fig-repo-90** | Model Stability Analysis (pminternal) | Stats | P3 | Generated |
| **fig-repo-91** | Ensemble Construction | Sci/ML | P3 | Generated |
| **fig-repo-92** | Feature Definition YAML | Sci | P3 | Generated |
| **fig-repo-93** | Common Test Failures & Fixes | All | P1 | Generated |
| **fig-repo-94** | Extraction Guardrails | ML | P3 | Generated |
| **fig-repo-95** | Quality Gate Decision Flow | All | P2 | Generated |
| **fig-repo-96** | Selective Classification & Uncertainty | Stats/Sci | P3 | Generated |
| **fig-repo-97** | Critical Failure Meta-Learnings | All | P2 | Generated |
| **fig-repo-98** | JSON Sidecar Pattern | Sci/ML | P2 | Generated |

## Personas

| Abbreviation | Full Name | Technical Level |
|--------------|-----------|-----------------|
| **PI** | Ophthalmology PI | Non-technical (Excel user) |
| **Stats** | Biostatistician | R, statistics |
| **Sci** | Research Scientist | Python basics |
| **ML** | ML Engineer | Full stack |
| **Agent** | LLM Agent | N/A (reads structured rules) |

## Directory Structure

```
docs/repo-figures/
├── README.md                 ← This file
├── STYLE-GUIDE.md           ← Visual consistency rules
├── CONTENT-TEMPLATE.md      ← Figure plan template
├── figure-plans/            ← Individual figure plans
│   ├── fig-repo-01-what-this-repo-does.md
│   ├── fig-repo-02-preprocessing-pipeline.md
│   └── ...
└── generated/               ← Output images (gitignored if private)
    └── (PNG/SVG files)
```

## Workflow

### Using Nano Banana Pro

1. Open figure plan (e.g., `figure-plans/fig-repo-01-what-this-repo-does.md`)
2. Read the **Visual Concept** section to understand the layout
3. Copy the **Style Prompt** to set the aesthetic
4. Copy the **Content Prompt** to describe what to show
5. Iterate using **Refinement Notes**
6. Export at 300 DPI as PNG

### Fallback Tools

If Nano Banana Pro is unavailable:

| Type | Tool | Notes |
|------|------|-------|
| Flowcharts | Mermaid | GitHub-native, paste in README |
| Hand-drawn | Excalidraw | Export as PNG |
| Technical | D2 | PlantUML alternative |
| Data-driven | R/ggplot2 | For statistical figures |

## Embedding in Docs

```markdown
<!-- In README.md or docs/ -->
![What this repo does](docs/repo-figures/generated/fig-repo-01-what-this-repo-does.png)

*Caption: This repository evaluates time series foundation models for pupil signal preprocessing.*
```

## Priority Order

**P1 (Create First):**
1. fig-repo-01: Hero image for README
2. fig-repo-02: Pipeline overview
3. fig-repo-07: Figure creation workflow (for contributors)

**P2 (High Impact):**
- fig-repo-03, fig-repo-04, fig-repo-05, fig-repo-11

**P3 (Complete Later):**
- fig-repo-06, fig-repo-08, fig-repo-09, fig-repo-10, fig-repo-12

## Contributing

When adding new figure plans:

1. Use `CONTENT-TEMPLATE.md` as the starting point
2. Follow `STYLE-GUIDE.md` for visual consistency
3. Number sequentially: `fig-repo-13-{name}.md`
4. Add to the catalog table above
5. Specify target persona and priority
