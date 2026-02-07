# fig-repo-18: Two-Block Architecture: Extraction vs Analysis

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repo-18 |
| **Title** | Two-Block Architecture: Extraction vs Analysis |
| **Complexity Level** | L2 (Technical overview) |
| **Target Persona** | ML Engineer, Research Scientist |
| **Location** | ARCHITECTURE.md, docs/user-guide/ |
| **Priority** | P0 |
| **Aspect Ratio** | 16:9 |

## Purpose

Explain the post-experiment processing architecture: compute metrics ONCE in extraction (Block 1), then READ ONLY in analysis/visualization (Block 2). This is separate from the 6 experiment subflows (see fig-repo-10).

## Relationship to Other Figures

| Figure | Scope | Focus |
|--------|-------|-------|
| **fig-repo-10** | 6 Experiment Subflows | Labor division for running experiments (Data Import → ... → Classification) |
| **fig-repo-18** (THIS) | 2 Post-Experiment Blocks | Extraction vs Analysis for publication artifacts |

## Key Message

"Block 1 computes all metrics from MLflow (once). Block 2 reads from DuckDB—NEVER recomputes. This ensures figures are reproducible."

## Output Formats (Verified from Code)

| Output | Format | Location | Shareable? |
|--------|--------|----------|------------|
| Metrics + predictions | DuckDB | `data/public/foundation_plr_results.db` | ✅ PUBLIC |
| Subject re-anonymization | YAML | `data/private/subject_lookup.yaml` | ❌ PRIVATE |
| Demo PLR traces | Pickle | `data/private/demo_subjects_traces.pkl` | ❌ PRIVATE |
| Figures | PNG/PDF | `figures/generated/` | ✅ PUBLIC |
| Figure data | JSON | `figures/generated/data/` | ⚠️ Check privacy |
| LaTeX tables | .tex | `tables/generated/` | ✅ PUBLIC |

**Note**: Block 1 does output ONE pickle file (demo traces), but this is private/gitignored. The shareable artifact is DuckDB only.

## Visual Concept

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                TWO-BLOCK ARCHITECTURE: EXTRACTION vs ANALYSIS                    │
│                                                                                  │
│  The pipeline separates computation (Block 1) from visualization (Block 2).     │
│  Block 1 extracts MLflow results and computes all STRATOS metrics, storing      │
│  them in DuckDB. Block 2 reads from DuckDB—it NEVER recomputes metrics.         │
│  This ensures figures are reproducible: regenerating a figure always uses       │
│  the same precomputed data.                                                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌──────────────────────────────────┐     ┌──────────────────────────────────┐  │
│  │      BLOCK 1: EXTRACTION         │     │       BLOCK 2: ANALYSIS          │  │
│  │      (Python)                    │     │       (R + Python)               │  │
│  │ ═══════════════════════════════════    │═════════════════════════════════  │  │
│  │                                  │     │                                  │  │
│  │  📊 MLflow                       │     │  🗄️ DuckDB                       │  │
│  │  └── 542 pickle files            │     │  └── foundation_plr_results.db   │  │
│  │      (~20 GB in mlruns/)         │     │      │                           │  │
│  │      │                           │     │      ▼                           │  │
│  │      ▼                           │     │  📖 READ ONLY:                   │  │
│  │  ⚙️ COMPUTE:                     │     │  • Load metrics                  │  │
│  │  • AUROC, Brier, NB              │     │  • Generate figures              │  │
│  │  • Calibration slope/intercept   │     │  • Export JSON data              │  │
│  │  • Bootstrap CIs                 │     │  • LaTeX tables                  │  │
│  │  • Re-anonymization              │     │      │                           │  │
│  │      │                           │     │      ▼                           │  │
│  │      ▼                           │     │  📈 figures/generated/           │  │
│  │  🗄️ DuckDB  ─────────────────────────▶│      └── ggplot2/*.png           │  │
│  │  (Public, shareable)             │     │      └── data/*.json             │  │
│  │                                  │     │                                  │  │
│  └──────────────────────────────────┘     └──────────────────────────────────┘  │
│                                                                                  │
│  ──────────────────────────────────────────────────────────────────────────────  │
│                                                                                  │
│  WHY THIS SEPARATION?                                                            │
│  ════════════════════                                                            │
│                                                                                  │
│  ❌ WITHOUT separation:                  ✅ WITH separation:                     │
│                                                                                  │
│  Compute metrics in viz code             Compute once in extraction              │
│       ↓                                       ↓                                  │
│  Different results each run              Same results always                     │
│  (floating point variance)               (read from database)                    │
│       ↓                                       ↓                                  │
│  "Which code computed this?"             Clear audit trail                       │
│       ↓                                       ↓                                  │
│  Unreproducible figures                  Reproducible figures                    │
│                                                                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│  COMMANDS                                                                        │
│  ════════                                                                        │
│                                                                                  │
│  make extract     →  Run Block 1 only (new MLflow data)                          │
│  make analyze     →  Run Block 2 only (most common)                              │
│  make reproduce   →  Run both blocks (full pipeline)                             │
│                                                                                  │
│  Most users: `make analyze` (figures from existing DB)                           │
│  After experiments: `make extract` then `make analyze`                           │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Content Elements

1. **Two-column block diagram**: Extraction (left) vs Analysis (right)
2. **Data flow arrow**: MLflow → DuckDB → Figures
3. **COMPUTE vs READ ONLY labels**: Emphasize the separation
4. **Why separation matters**: Without vs With comparison
5. **Command reference**: make extract, make analyze, make reproduce

## Text Content

### Title Text
"Two-Block Architecture: Compute Once, Read Forever"

### Caption
The pipeline separates computation (Block 1) from visualization (Block 2). Block 1 extracts MLflow results and computes all STRATOS metrics, storing them in DuckDB. Block 2 reads from DuckDB—it NEVER recomputes metrics. This ensures figures are reproducible: regenerating a figure always uses the same precomputed data.

## Prompts for Nano Banana Pro

### Style Prompt
Architecture diagram with two distinct blocks. Clean swimlane-style layout. Left block in blue tones (computation), right block in green tones (analysis). Clear data flow arrows between blocks. "COMPUTE" and "READ ONLY" badges. Command reference box at bottom. Matte, professional, Economist-style.

### Content Prompt
Create a two-block architecture diagram:

**LEFT BLOCK (Blue) - "EXTRACTION"**:
- MLflow icon at top (note: ~20GB, 542 pickle files)
- Arrow down through "COMPUTE" operations (list metrics)
- Arrow to DuckDB cylinder at bottom

**RIGHT BLOCK (Green) - "ANALYSIS"**:
- DuckDB cylinder at top (arrow from left block)
- "READ ONLY" badge
- Arrow down to figure icons (PNG, PDF)

**MIDDLE - Why Separation**:
- Two columns: "WITHOUT" (problems) vs "WITH" (benefits)

**BOTTOM - Commands**:
- Three commands with descriptions

## Alt Text

Two-block architecture diagram. Left block (Extraction): MLflow → compute AUROC, Brier, calibration metrics → DuckDB. Right block (Analysis): DuckDB → read only → generate figures. Comparison shows without separation causes unreproducible figures; with separation ensures consistent results. Commands: make extract, make analyze, make reproduce.

## Status

- [x] Draft created
- [x] Updated with verified output formats
- [ ] Generated
- [ ] Placed in ARCHITECTURE.md
