# fig-repo-37: Prefect Orchestration

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repo-37 |
| **Title** | Prefect Orchestration |
| **Complexity Level** | L2 (Technical) |
| **Target Persona** | ML Engineer, Data Engineer |
| **Location** | ARCHITECTURE.md, docs/development/ |
| **Priority** | P2 |
| **Aspect Ratio** | 16:10 |

## Purpose

Explain how Prefect orchestrates the extraction and analysis flows with retry logic and dependency tracking.

## Key Message

"Prefect flows coordinate the two-block pipeline: extraction flow (MLflow → DuckDB) and analysis flow (DuckDB → figures). Built-in retries and observability."

## Visual Concept

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    PREFECT ORCHESTRATION                                        │
│                    Coordinating the Two-Block Pipeline                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  WHAT IS PREFECT?                                                               │
│  ════════════════                                                               │
│                                                                                 │
│  A workflow orchestration framework that:                                       │
│  • Coordinates tasks with dependencies                                          │
│  • Retries failed tasks automatically                                           │
│  • Tracks execution state and logs                                              │
│  • Provides observability (UI dashboard)                                        │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  THE TWO FLOWS                                                                  │
│  ══════════════                                                                 │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                         │   │
│  │   EXTRACTION FLOW (Block 1)                                             │   │
│  │   ═════════════════════════                                             │   │
│  │                                                                         │   │
│  │   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐            │   │
│  │   │ Connect  │ → │ Load     │ → │ Compute  │ → │ Write    │            │   │
│  │   │ MLflow   │   │ Pickles  │   │ STRATOS  │   │ DuckDB   │            │   │
│  │   │          │   │          │   │ Metrics  │   │          │            │   │
│  │   └──────────┘   └──────────┘   └──────────┘   └──────────┘            │   │
│  │        │              │              │              │                   │   │
│  │        └──────────────┴──────────────┴──────────────┘                   │   │
│  │                    retries=3, retry_delay=60s                           │   │
│  │                                                                         │   │
│  │   Output: data/public/foundation_plr_results.db                         │   │
│  │                                                                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                         │   │
│  │   ANALYSIS FLOW (Block 2)                                               │   │
│  │   ═══════════════════════                                               │   │
│  │                                                                         │   │
│  │   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐            │   │
│  │   │ Load     │ → │ Generate │ → │ Export   │ → │ Validate │            │   │
│  │   │ DuckDB   │   │ Figures  │   │ JSON     │   │ Figure   │            │   │
│  │   │          │   │          │   │ Data     │   │ QA       │            │   │
│  │   └──────────┘   └──────────┘   └──────────┘   └──────────┘            │   │
│  │                                                                         │   │
│  │   Can run independently (DuckDB already exists)                         │   │
│  │                                                                         │   │
│  │   Output: figures/generated/*.png + data/*.json                         │   │
│  │                                                                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  CODE STRUCTURE                                                                 │
│  ══════════════                                                                 │
│                                                                                 │
│  src/orchestration/                                                             │
│  ├── flows/                                                                     │
│  │   ├── extraction_flow.py      @flow decorator                               │
│  │   │   └── extract_all_configs()                                             │
│  │   │       ├── connect_mlflow()      @task                                   │
│  │   │       ├── load_pickle()         @task                                   │
│  │   │       ├── compute_stratos()     @task                                   │
│  │   │       └── write_duckdb()        @task                                   │
│  │   │                                                                         │
│  │   └── analysis_flow.py        @flow decorator                               │
│  │       └── generate_figures()                                                │
│  │           ├── load_duckdb()         @task                                   │
│  │           ├── generate_roc()        @task                                   │
│  │           ├── generate_calibration()@task                                   │
│  │           └── validate_figures()    @task                                   │
│  │                                                                             │
│  └── tasks/                      Reusable task definitions                     │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  RUNNING FLOWS                                                                  │
│  ═════════════                                                                  │
│                                                                                 │
│  # Via Makefile (recommended)                                                   │
│  make reproduce              # Both flows                                       │
│  make extract                # Extraction flow only                             │
│  make analyze                # Analysis flow only                               │
│                                                                                 │
│  # Via Python                                                                   │
│  python -m src.orchestration.flows.extraction_flow                              │
│  python -m src.orchestration.flows.analysis_flow                                │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  KEY FEATURES                                                                   │
│  ════════════                                                                   │
│                                                                                 │
│  🔄 Automatic retries         Failed tasks retry 3x before failing flow         │
│  📊 Task dependencies         Prefect tracks which tasks depend on which        │
│  📝 Execution logs            Every task logs to Prefect + loguru               │
│  🎯 Observability            Optional Prefect Cloud dashboard                   │
│  ⚡ Parallel execution        Independent tasks can run concurrently            │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Content Elements

1. **What is Prefect**: Brief explanation
2. **Two flows diagram**: Extraction and Analysis with task boxes
3. **Code structure**: Directory tree with @flow and @task
4. **Running commands**: make and python invocations
5. **Key features**: Retries, dependencies, logs, observability

## Text Content

### Title Text
"Prefect Orchestration: Two Flows, One Pipeline"

### Caption
Prefect coordinates the two-block pipeline: extraction_flow (MLflow → DuckDB with STRATOS metric computation) and analysis_flow (DuckDB → figures with validation). Each flow contains tasks with automatic retry logic. The analysis flow can run independently when DuckDB already exists, enabling fast figure iteration without re-extraction.

## Prompts for Nano Banana Pro

### Style Prompt
Two-flow diagram with task boxes connected by arrows. Code structure showing decorators. Command reference. Feature icons. Clean, workflow-focused aesthetic.

### Content Prompt
Create a Prefect orchestration diagram:

**TOP - What is Prefect**:
- 4 bullet points

**MIDDLE - Two Flows**:
- EXTRACTION: 4 connected task boxes (connect → load → compute → write)
- ANALYSIS: 4 connected task boxes (load → generate → export → validate)
- Note retries and outputs

**BOTTOM LEFT - Code Structure**:
- Directory tree with @flow and @task annotations

**BOTTOM RIGHT - Commands**:
- make commands and python invocations

## Alt Text

Prefect orchestration diagram showing two flows. Extraction flow: connect MLflow → load pickles → compute STRATOS metrics → write DuckDB (with retries=3). Analysis flow: load DuckDB → generate figures → export JSON → validate QA. Code structure in src/orchestration/flows/ with @flow and @task decorators. Commands: make reproduce (both), make extract (extraction only), make analyze (analysis only).

## Status

- [x] Draft created
- [ ] Generated
- [ ] Placed in ARCHITECTURE.md
