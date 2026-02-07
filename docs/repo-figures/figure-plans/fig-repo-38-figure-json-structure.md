# fig-repo-38: Figure JSON Data Structure

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repo-38 |
| **Title** | Figure JSON Data Structure |
| **Complexity Level** | L2 (Technical) |
| **Target Persona** | ML Engineer, Data Scientist |
| **Location** | CONTRIBUTING.md, configs/VISUALIZATION/ |
| **Priority** | P1 |
| **Aspect Ratio** | 16:10 |

## Purpose

Explain the mandatory JSON data files that accompany every figure for reproducibility.

## Key Message

"Every figure has a JSON sidecar with the exact data used to generate it. This ensures reproducibility: same JSON → identical figure."

## Visual Concept

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    FIGURE JSON DATA STRUCTURE                                   │
│                    Reproducibility through data sidecars                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  THE PATTERN                                                                    │
│  ═══════════                                                                    │
│                                                                                 │
│  Every figure has THREE files:                                                  │
│                                                                                 │
│  figures/generated/                                                             │
│  ├── fig_calibration.png         ← Visual output (raster)                       │
│  ├── fig_calibration.pdf         ← Visual output (vector)                       │
│  └── data/                                                                      │
│      └── fig_calibration.json    ← Data sidecar (REPRODUCIBILITY)               │
│                                                                                 │
│  PNG/PDF can be regenerated from JSON at any time.                              │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  JSON STRUCTURE                                                                 │
│  ══════════════                                                                 │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  {                                                                      │   │
│  │    "metadata": {                                                        │   │
│  │      "figure_id": "fig_calibration",                                    │   │
│  │      "generated_at": "2025-01-30T14:32:00Z",                            │   │
│  │      "source_db": "foundation_plr_results.db",                          │   │
│  │      "source_hash": "abc123...",          ← Data provenance             │   │
│  │      "n_subjects": 208,                                                 │   │
│  │      "n_bootstrap": 1000                                                │   │
│  │    },                                                                   │   │
│  │                                                                         │   │
│  │    "combos": ["ground_truth", "best_ensemble", ...],                    │   │
│  │                                                                         │   │
│  │    "stratos_metrics": {                   ← MANDATORY STRATOS section   │   │
│  │      "ground_truth": {                                                  │   │
│  │        "auroc": 0.911,                                                  │   │
│  │        "auroc_ci_lo": 0.851,                                            │   │
│  │        "auroc_ci_hi": 0.955,                                            │   │
│  │        "calibration_slope": 0.98,                                       │   │
│  │        "calibration_intercept": -0.02,                                  │   │
│  │        "brier": 0.131,                                                  │   │
│  │        "net_benefit_15pct": 0.199                                       │   │
│  │      },                                                                 │   │
│  │      "best_ensemble": { ... }                                           │   │
│  │    },                                                                   │   │
│  │                                                                         │   │
│  │    "curves": {                            ← Plot coordinates            │   │
│  │      "ground_truth": {                                                  │   │
│  │        "x": [0.0, 0.1, 0.2, ...],                                       │   │
│  │        "y": [0.0, 0.12, 0.25, ...],                                     │   │
│  │        "ci_lo": [...],                                                  │   │
│  │        "ci_hi": [...]                                                   │   │
│  │      }                                                                  │   │
│  │    }                                                                    │   │
│  │  }                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  MANDATORY SECTIONS                                                             │
│  ═════════════════                                                              │
│                                                                                 │
│  │ Section          │ Required │ Contents                                     │ │
│  │ ────────────────  │ ──────── │ ─────────────────────────────────────────── │ │
│  │ metadata          │ YES      │ Figure ID, timestamp, source, hash          │ │
│  │ combos            │ YES      │ List of pipeline combinations               │ │
│  │ stratos_metrics   │ YES      │ AUROC, calibration, Brier, NB for each      │ │
│  │ curves            │ Depends  │ X/Y coordinates if figure has curves        │ │
│  │ raw_predictions   │ PRIVATE  │ y_true, y_prob arrays (gitignored)          │ │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  PRIVACY LEVELS                                                                 │
│  ══════════════                                                                 │
│                                                                                 │
│  PUBLIC (committed to git):                                                     │
│  • Aggregate metrics (AUROC, Brier, calibration)                                │
│  • Curve coordinates (ROC, calibration, DCA)                                    │
│  • Summary statistics                                                           │
│                                                                                 │
│  PRIVATE (gitignored):                                                          │
│  • Per-subject predictions (y_true, y_prob arrays)                              │
│  • Individual PLR traces                                                        │
│  • Re-identification data                                                       │
│                                                                                 │
│  Check figure_registry.yaml `json_privacy` field for each figure.               │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  WHY JSON SIDECARS?                                                             │
│  ══════════════════                                                             │
│                                                                                 │
│  ✅ Reproducibility: Regenerate figure from data without re-running pipeline    │
│  ✅ Audit trail: Know exactly what data produced this figure                    │
│  ✅ Version control: Git tracks JSON changes, reveals data drift                │
│  ✅ QA testing: Figure tests validate JSON structure and values                 │
│  ✅ Sharing: Reviewers can verify claims from JSON data                         │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Content Elements

1. **File pattern**: PNG/PDF + JSON sidecar
2. **JSON structure example**: Annotated schema
3. **Mandatory sections table**: What's required
4. **Privacy levels**: Public vs private data
5. **Why sidecars**: Benefits list

## Text Content

### Title Text
"Figure JSON Sidecars: Data for Reproducibility"

### Caption
Every generated figure has a JSON sidecar containing the exact data used: metadata (source, timestamp, hash), STRATOS metrics for each pipeline combo, and curve coordinates. This enables reproducibility (regenerate figure from JSON), audit trails (provenance tracking), and QA testing. Private data (per-subject predictions) is gitignored while aggregate data is committed.

## Prompts for Nano Banana Pro

### Style Prompt
File structure showing PNG+PDF+JSON trio. JSON schema with annotations. Mandatory sections table. Privacy levels with icons. Benefits checklist. Clean, data-focused aesthetic.

### Content Prompt
Create a figure JSON structure diagram:

**TOP - File Pattern**:
- Three file icons: .png, .pdf, .json
- Arrow showing JSON → regenerates → PNG/PDF

**MIDDLE - JSON Structure**:
- Annotated JSON example with callouts for metadata, stratos_metrics, curves

**BOTTOM LEFT - Mandatory Sections**:
- Table: Section | Required | Contents

**BOTTOM RIGHT - Privacy + Benefits**:
- Public vs private data lists
- 5 benefits with checkmarks

## Alt Text

Figure JSON data structure diagram. Every figure has three files: PNG (raster), PDF (vector), and JSON (data sidecar). JSON structure includes metadata (figure_id, timestamp, source_hash), combos list, stratos_metrics (AUROC, calibration, Brier, NB per pipeline), and curves (x/y coordinates). Public data committed to git; private per-subject predictions gitignored. Benefits: reproducibility, audit trail, version control, QA testing, sharing.

## Status

- [x] Draft created
- [ ] Generated
- [ ] Placed in CONTRIBUTING.md
