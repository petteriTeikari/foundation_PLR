# fig-repo-23: Data Privacy: What Gets Shared vs Private

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repo-23 |
| **Title** | Data Privacy: What Gets Shared vs Private |
| **Complexity Level** | L2 (Technical concept) |
| **Target Persona** | All |
| **Location** | docs/user-guide/, CONTRIBUTING.md |
| **Priority** | P1 |
| **Aspect Ratio** | 16:10 |

## Purpose

Clarify what data is shareable (public) vs what stays private, critical for understanding the two-block architecture and SERI institutional data rights.

## Key Message

"Aggregate metrics are PUBLIC and shareable. Individual subject data (PLR traces, per-subject predictions) is PRIVATE and gitignored."

## Visual Concept

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    DATA PRIVACY: WHAT GETS SHARED VS PRIVATE                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────────────────────┐   ┌─────────────────────────────────────┐ │
│  │     🔓 PUBLIC (Shareable)       │   │      🔒 PRIVATE (Gitignored)        │ │
│  │     ═══════════════════════     │   │      ════════════════════════       │ │
│  │                                 │   │                                     │ │
│  │  📊 Aggregate Metrics           │   │  👤 Individual PLR Traces           │ │
│  │     • AUROC (mean, CI)          │   │     • Raw pupil diameter            │ │
│  │     • Brier scores              │   │     • Per-subject time series       │ │
│  │     • Calibration stats         │   │     • Original PLRxxxx codes        │ │
│  │                                 │   │                                     │ │
│  │  📁 DuckDB (public version)     │   │  🔗 Subject Lookup Table            │ │
│  │     • foundation_plr_results.db │   │     • Hxxx/Gxxx → PLRxxxx mapping   │ │
│  │     • Re-anonymized subject IDs │   │     • Re-identification data        │ │
│  │                                 │   │                                     │ │
│  │  📈 Figure JSON (aggregate)     │   │  📈 Figure JSON (individual)        │ │
│  │     • ROC curve coordinates     │   │     • Per-subject predictions       │ │
│  │     • Calibration curves        │   │     • Individual uncertainty        │ │
│  │     • DCA threshold data        │   │     • Demo subject traces           │ │
│  │                                 │   │                                     │ │
│  │  📝 Model Parameters            │   │  📝 Institutional Data              │ │
│  │     • Hyperparameters           │   │     • SERI_PLR_GLAUCOMA.db          │ │
│  │     • Training config           │   │     • Original dataset              │ │
│  │                                 │   │                                     │ │
│  │  ✅ Committed to GitHub         │   │  ❌ Listed in .gitignore             │ │
│  │                                 │   │                                     │ │
│  └─────────────────────────────────┘   └─────────────────────────────────────┘ │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  RE-ANONYMIZATION SCHEME                                                        │
│  ═══════════════════════                                                        │
│                                                                                 │
│  Original (PRIVATE):          Public (SHAREABLE):                               │
│                                                                                 │
│  PLR0042                 →    H001 (Healthy subject 1)                          │
│  PLR0187                 →    H002 (Healthy subject 2)                          │
│  PLR0329                 →    G001 (Glaucoma subject 1)                         │
│  ...                          ...                                               │
│                                                                                 │
│  The mapping table (subject_lookup.yaml) is PRIVATE                             │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  WHY THIS MATTERS                                                               │
│  ════════════════                                                               │
│                                                                                 │
│  🏥 SERI Institutional Data Rights                                              │
│     Original PLR recordings belong to Singapore Eye Research Institute          │
│                                                                                 │
│  🔬 Research Reproducibility                                                    │
│     Public DuckDB allows figure regeneration without private data               │
│                                                                                 │
│  ⚖️ Patient Privacy (PDPA Compliance)                                           │
│     Individual medical data cannot be shared publicly                           │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  .gitignore PATTERNS                                                            │
│  ═══════════════════                                                            │
│                                                                                 │
│  # Private data patterns                                                        │
│  data/private/                                                                  │
│  **/subject_*.json                                                              │
│  **/individual_*.json                                                           │
│  *_lookup.yaml                                                                  │
│  SERI_PLR_GLAUCOMA.db                                                           │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Content Elements

1. **Two-column split**: Public (left, green) vs Private (right, red)
2. **Category lists**: What data falls into each bucket
3. **Re-anonymization diagram**: PLRxxxx → Hxxx/Gxxx mapping
4. **Why this matters**: Institutional rights, reproducibility, PDPA
5. **Gitignore patterns**: Actual patterns from the repository

## Text Content

### Title Text
"Data Privacy: Aggregate is Public, Individual is Private"

### Caption
The repository separates PUBLIC aggregate data (AUROC, calibration stats, DuckDB) from PRIVATE individual data (PLR traces, per-subject predictions, re-identification mappings). This enables research reproducibility while respecting SERI institutional data rights and Singapore PDPA compliance. All figures can be regenerated from the public DuckDB; private data is only needed for subject-level visualizations.

## Prompts for Nano Banana Pro

### Style Prompt
Two-column privacy comparison diagram. Left column green/blue (public, shareable), right column red/orange (private, gitignored). Lock icons for each column. File icons showing databases, JSON files. Arrow showing re-anonymization mapping. Clean, informative, compliance-focused aesthetic.

### Content Prompt
Create a data privacy classification diagram:

**LEFT COLUMN (Green) - "PUBLIC"**:
- Database icon: "foundation_plr_results.db"
- Chart icon: "Aggregate metrics (AUROC, Brier)"
- JSON icon: "Figure data (ROC curves, DCA)"
- Checkmark: "Committed to GitHub"

**RIGHT COLUMN (Red) - "PRIVATE"**:
- Person icon: "Individual PLR traces"
- Key icon: "Subject lookup table"
- Database icon: "SERI_PLR_GLAUCOMA.db"
- X mark: "Listed in .gitignore"

**MIDDLE - Re-anonymization**:
- Arrow: "PLR0042 → H001"
- Note: "Mapping is PRIVATE"

**FOOTER**:
- Three reasons: SERI rights, Reproducibility, PDPA compliance

## Alt Text

Data privacy diagram showing public vs private data classification. Public (shareable): aggregate metrics, public DuckDB, figure JSON for ROC/DCA curves. Private (gitignored): individual PLR traces, subject lookup table mapping PLRxxxx to Hxxx/Gxxx, original SERI database. Re-anonymization scheme protects patient identity while enabling reproducibility.

## Status

- [x] Draft created
- [ ] Generated
- [ ] Placed in docs/user-guide/
