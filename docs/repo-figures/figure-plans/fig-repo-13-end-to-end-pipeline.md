# fig-repo-13: End-to-End Pipeline: CSVs → Figures

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repo-13 |
| **Title** | End-to-End Pipeline: CSVs → Publication Figures |
| **Complexity Level** | L2 (Technical overview) |
| **Target Persona** | All (especially Biostatisticians, Research Scientists) |
| **Location** | Root README, ARCHITECTURE.md |
| **Priority** | P0 (Critical) |
| **Aspect Ratio** | 16:9 (landscape, horizontal flow) |

## Purpose

Show the complete data journey from scattered clinical CSV files to publication-ready figures, emphasizing the single-database consolidation and reproducibility architecture.

## Differentiation from fig-repo-01

- **fig-repo-01**: "What does this repo do?" - Hero image, high-level concept
- **fig-repo-13**: Technical data flow with file formats, subject counts, and tool icons

## Key Message

"From 500+ scattered CSV files to 40+ publication figures: consolidated in one database, tracked by MLflow, visualized with ggplot2."

## Visual Concept

**Horizontal timeline with data transformation stages:**

```
┌───────────────┐   ┌───────────────┐   ┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│   RAW DATA    │   │   DATABASE    │   │   PIPELINE    │   │   TRACKING    │   │    FIGURES    │
│               │   │               │   │               │   │               │   │               │
│  📁 500+ CSVs │──▶│ 🗄️ DuckDB    │──▶│ ⚙️ 4 Stages  │──▶│ 📊 MLflow    │──▶│ 📈 ggplot2   │
│  (scattered)  │   │ (single file) │   │ (Prefect)     │   │ (410+ runs)   │   │ (40+ figures) │
│               │   │               │   │               │   │               │   │               │
│  ~1M points   │   │  507 subjects │   │  11×8×5=440  │   │  542 pickles  │   │  PNG + PDF    │
│               │   │  1981 t/subj  │   │  combinations │   │  per-run      │   │  + JSON data  │
└───────────────┘   └───────────────┘   └───────────────┘   └───────────────┘   └───────────────┘
        │                   │                   │                   │                   │
        │                   │                   │                   │                   │
        ▼                   ▼                   ▼                   ▼                   ▼
   "Reproducibility        "Single source      "Error             "Every run         "Regenerate
    nightmare"              of truth"           propagation        logged"            anytime"
                                                tested"
```

## Content Elements

### Required Elements
1. **Five stages** shown left-to-right (data consolidation → figures)
2. **File format icons** (CSV, DuckDB, pickle, PNG/PDF)
3. **Subject counts** (507 preprocessing, 208 classification)
4. **Tool logos/icons** (DuckDB, Prefect, MLflow, ggplot2)
5. **Numbers**: 1M timepoints, 410 runs, 40+ figures

### Optional Elements
1. Small inset showing "before" (scattered files) vs "after" (single DB)
2. Timeline bar showing typical execution time
3. Re-anonymization callout (PLRxxxx → Hxxx/Gxxx)

## Text Content

### Title Text
"From Scattered CSVs to Publication Figures"

### Labels/Annotations
- Stage 1: "500+ clinical CSVs consolidated into single DuckDB file"
- Stage 2: "507 subjects × 1981 timepoints = 1M+ data points"
- Stage 3: "Prefect orchestrates 4-stage preprocessing pipeline"
- Stage 4: "MLflow tracks every experiment combination"
- Stage 5: "ggplot2 generates publication-quality figures with JSON data"

### Caption (for embedding)
The Foundation PLR pipeline transforms 500+ scattered clinical CSV files into 40+ publication figures. Data is consolidated into a single DuckDB database (507 subjects, 1M+ timepoints), processed through a 4-stage Prefect-orchestrated pipeline (outlier detection → imputation → featurization → classification), tracked in MLflow (410+ experiment runs), and visualized with ggplot2. Every step is reproducible: regenerate any figure from the same data.

## Prompts for Nano Banana Pro

### Style Prompt
Scientific data pipeline visualization. Clean horizontal flow diagram with 5 stages. Use matte, professional colors from Economist palette. No glowing effects. Icons for each tool (database, gears, charts). Include numerical callouts in small badges. Light background, subtle shadows for depth. Medical/clinical aesthetic, not tech-startup. Retina/eye iconography welcome if contextually appropriate.

### Content Prompt
Create a horizontal pipeline infographic showing data transformation:

**Stage 1 - RAW DATA**: Folder icon with scattered CSV files, label "500+ CSVs, ~1M timepoints"
**Stage 2 - DATABASE**: DuckDB cylinder icon, label "Single DuckDB file, 507 subjects"
**Stage 3 - PIPELINE**: Gear/process icon with 4 smaller gears inside, label "4-stage preprocessing"
**Stage 4 - TRACKING**: Chart/dashboard icon, label "MLflow: 410+ tracked runs"
**Stage 5 - FIGURES**: Publication figure icon (chart + PDF), label "40+ ggplot2 figures"

Connect with flowing arrows. Below each stage, add a quote in italics:
- "Reproducibility nightmare" → "Single source of truth" → "Error propagation tested" → "Every run logged" → "Regenerate anytime"

### Refinement Notes
- The transition from Stage 1 → Stage 2 should feel like "consolidation/relief"
- Stage 5 should feel like "achievement/completion"
- Include small numerical badges for key stats
- Avoid making it look like a generic tech diagram - this is clinical research

## Alt Text

Horizontal pipeline diagram showing five stages of data transformation: (1) 500+ scattered CSV files, (2) consolidated DuckDB database with 507 subjects, (3) 4-stage Prefect preprocessing pipeline, (4) MLflow experiment tracking with 410+ runs, (5) 40+ ggplot2 publication figures. Arrows show data flow with quotes emphasizing reproducibility at each stage.

## Technical Notes

### Data Validation Points
- Subject count: 507 (verify in ARCHITECTURE.md line 10)
- Timepoints per subject: 1981 (verify in defaults.yaml)
- MLflow runs: 410+ (verify in mlflow_registry)
- Figure count: 40+ (verify in figures/generated/)

### Source Files Referenced
- `SERI_PLR_GLAUCOMA.db`: Input database
- `foundation_plr_results.db`: Extracted results
- `/home/petteri/mlruns/`: MLflow experiment storage
- `figures/generated/ggplot2/`: Output figures

## Status

- [x] Draft created
- [ ] Review passed
- [ ] Generated (16:9 aspect ratio)
- [ ] Placed in README.md, ARCHITECTURE.md
