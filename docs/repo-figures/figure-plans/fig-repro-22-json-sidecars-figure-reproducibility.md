# fig-repro-22: JSON Sidecars for Figure Reproducibility

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repro-22 |
| **Title** | JSON Sidecars for Figure Reproducibility |
| **Complexity Level** | L2 (Intermediate) |
| **Target Persona** | ML Engineer, Data Scientist |
| **Location** | CONTRIBUTING.md |
| **Priority** | P1 |
| **Aspect Ratio** | 16:10 |

## Purpose

Explain the mandatory JSON sidecar requirement for every figure to enable data provenance and regeneration.

## Key Message

"Every figure has a JSON file with the exact data that generated it. Same JSON → identical figure. This is how we verify scientific claims."

## Visual Concept

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    JSON SIDECARS FOR FIGURE REPRODUCIBILITY                     │
│                    Every figure has its data                                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  THE PATTERN: FIGURE + DATA                                                     │
│  ═══════════════════════════                                                    │
│                                                                                 │
│  figures/generated/                                                             │
│  ├── fig_calibration.png              ← Visual (raster)                         │
│  ├── fig_calibration.pdf              ← Visual (vector, for print)              │
│  └── data/                                                                      │
│      └── fig_calibration.json         ← DATA (source of truth!)                 │
│                                                                                 │
│  The JSON file contains EVERYTHING needed to regenerate the figure:             │
│  • Raw data points                                                              │
│  • Computed metrics                                                             │
│  • Source database hash                                                         │
│  • Timestamp                                                                    │
│                                                                                 │
│  PNG/PDF can be deleted and regenerated from JSON at any time!                  │
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
│  │      "source_hash": "sha256:abc123...",     ← Data provenance!          │   │
│  │      "git_commit": "e465a96",               ← Code version!             │   │
│  │      "n_subjects": 208                                                  │   │
│  │    },                                                                   │   │
│  │                                                                         │   │
│  │    "combos": ["ground_truth", "best_ensemble", "best_single_fm"],       │   │
│  │                                                                         │   │
│  │    "stratos_metrics": {                     ← STRATOS compliance!       │   │
│  │      "ground_truth": {                                                  │   │
│  │        "auroc": 0.911,                                                  │   │
│  │        "auroc_ci_lo": 0.851,                                            │   │
│  │        "auroc_ci_hi": 0.955,                                            │   │
│  │        "calibration_slope": 0.98,                                       │   │
│  │        "brier": 0.131                                                   │   │
│  │      },                                                                 │   │
│  │      ...                                                                │   │
│  │    },                                                                   │   │
│  │                                                                         │   │
│  │    "curves": {                              ← Plot coordinates          │   │
│  │      "ground_truth": {                                                  │   │
│  │        "predicted_prob": [0.0, 0.1, ...],                               │   │
│  │        "observed_prop": [0.0, 0.08, ...],                               │   │
│  │        "ci_lo": [...],                                                  │   │
│  │        "ci_hi": [...]                                                   │   │
│  │      }                                                                  │   │
│  │    }                                                                    │   │
│  │  }                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  MANDATORY SECTIONS                                                             │
│  ══════════════════                                                             │
│                                                                                 │
│  │ Section         │ Required │ Contents                          │ Privacy │  │
│  │ ─────────────── │ ──────── │ ───────────────────────────────── │ ─────── │  │
│  │ metadata        │ YES      │ ID, timestamp, source hash, git   │ PUBLIC  │  │
│  │ combos          │ YES      │ Pipeline combinations used        │ PUBLIC  │  │
│  │ stratos_metrics │ YES      │ AUROC, calibration, Brier, NB     │ PUBLIC  │  │
│  │ curves          │ If plot  │ X/Y coordinates for all curves    │ PUBLIC  │  │
│  │ raw_predictions │ If avail │ y_true, y_prob per subject        │ PRIVATE │  │
│                                                                                 │
│  PRIVATE sections are gitignored (patient data protection)                      │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  WHY JSON SIDECARS?                                                             │
│  ═══════════════════                                                            │
│                                                                                 │
│  ✅ REPRODUCIBILITY: Delete PNG, regenerate from JSON, get identical figure    │
│  ✅ VERIFICATION: Reviewers can check claims against JSON data                 │
│  ✅ AUDIT TRAIL: source_hash + git_commit = full provenance                    │
│  ✅ QA TESTING: test_figure_qa validates JSON structure                        │
│  ✅ VERSION CONTROL: Git tracks JSON changes (semantic, not pixel diffs)       │
│                                                                                 │
│  Lesson learned: CRITICAL-FAILURE-001 caught synthetic data in figures         │
│  because JSON didn't match expected data source!                                │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Content Elements

1. **File pattern**: PNG + PDF + JSON trio
2. **JSON structure**: Annotated example with sections
3. **Mandatory sections table**: What's required and privacy levels
4. **Benefits checklist**: Five reasons for sidecars
5. **Lesson learned**: CRITICAL-FAILURE-001 reference

## Text Content

### Title Text
"JSON Sidecars: Every Figure Has Its Data"

### Caption
Every Foundation PLR figure has a JSON sidecar containing metadata (source hash, git commit), STRATOS metrics (AUROC, calibration, Brier), and curve coordinates. This enables reproducibility (regenerate from JSON), verification (reviewers check claims), and QA testing (test_figure_qa validates structure). Lesson learned: CRITICAL-FAILURE-001 was caught because JSON data didn't match expected provenance.

## Prompts for Nano Banana Pro

### Style Prompt
File trio icon (PNG, PDF, JSON). JSON code block with annotations. Mandatory sections table. Benefits checklist with icons. Clean, data-documentation style.

### Content Prompt
Create "JSON Sidecars" infographic:

**TOP - File Pattern**:
- Three file icons: .png, .pdf, .json
- Arrow: JSON → regenerates → PNG/PDF

**MIDDLE - JSON Structure**:
- Code block with sections annotated
- Callouts for provenance, metrics, curves

**BOTTOM - Benefits + Table**:
- Mandatory sections table with privacy column
- Five benefits with checkmarks
- CRITICAL-FAILURE-001 reference

## Alt Text

JSON sidecars for figure reproducibility infographic. File pattern: fig_calibration.png, .pdf, and data/fig_calibration.json. JSON structure shows metadata (figure_id, timestamp, source_hash, git_commit), combos list, stratos_metrics (auroc, calibration_slope, brier per combo), and curves (x/y coordinates with CI). Mandatory sections table: metadata, combos, stratos_metrics required and public; curves if plot; raw_predictions private (gitignored). Benefits: reproducibility, verification, audit trail, QA testing, version control. Reference to CRITICAL-FAILURE-001 caught by JSON validation.

## Status

- [x] Draft created
- [ ] Generated
- [ ] Placed in CONTRIBUTING.md

