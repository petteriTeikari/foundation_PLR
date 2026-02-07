# fig-trans-18: From PLR to Your Domain: Fork Guide

**Status**: 📋 PLANNED
**Tier**: 4 - Repository Patterns
**Target Persona**: Developers, researchers who want to use PLR as a template

---

## 1. Metadata

| Field | Value |
|-------|-------|
| Figure ID | fig-trans-18 |
| Type | Step-by-step workflow diagram |
| Style | 75% manuscript + 25% Economist |
| Dimensions | 14" × 12" |
| Format | PDF (vector) + PNG (300 DPI) |

---

## 2. Purpose

Provide a concrete roadmap for adapting the PLR repository to a new domain, with estimated effort levels and a checklist of files to modify.

---

## 3. Key Message

> "Fork, don't rewrite. 70% of this codebase is domain-agnostic infrastructure you can keep. Here's exactly what to change to adapt PLR to vibration monitoring, seismic analysis, or any dense time series domain."

---

## 4. Visual Concept

```
┌────────────────────────────────────────────────────────────────────────────┐
│  FROM PLR TO YOUR DOMAIN: Fork Guide                                       │
│  A Step-by-Step Adaptation Roadmap                                         │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  PHASE 0: Setup (1 hour)                                                   │
│  ───────────────────────                                                   │
│                                                                            │
│  □ Fork repository                                                         │
│  □ Run `make setup` to install dependencies                                │
│  □ Run `make test` to verify baseline works                                │
│  □ Read ARCHITECTURE.md and CLAUDE.md                                      │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  PHASE 1: Data Layer (2-4 hours)                                           │
│  ───────────────────────────────                                           │
│                                                                            │
│  ┌────────────────────────────────────────────────────────────────────┐   │
│  │  File                        │ Change                              │   │
│  │  ────────────────────────────┼─────────────────────────────────── │   │
│  │  src/data_io/schema.py       │ Define YOUR signal schema          │   │
│  │  configs/defaults.yaml       │ Update prevalence, class names     │   │
│  │  data/your_database.db       │ Create with YOUR data              │   │
│  └────────────────────────────────────────────────────────────────────┘   │
│                                                                            │
│  Key decisions:                                                            │
│  • Sampling rate of your signal                                            │
│  • How ground truth is defined                                             │
│  • What "outlier" means in your domain                                     │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  PHASE 2: Feature Engineering (4-8 hours) ← MOST EFFORT HERE              │
│  ─────────────────────────────────────────                                 │
│                                                                            │
│  ┌────────────────────────────────────────────────────────────────────┐   │
│  │  PLR Feature              │ Your Domain (example: Vibration)       │   │
│  │  ─────────────────────────┼──────────────────────────────────────  │   │
│  │  amplitude_bins.py        │ fft_magnitude_bins.py                  │   │
│  │  latency_pipr.py          │ bearing_fault_frequencies.py           │   │
│  │  baseline_correction.py   │ high_pass_filter.py                    │   │
│  └────────────────────────────────────────────────────────────────────┘   │
│                                                                            │
│  This is where domain knowledge lives. Options:                            │
│  • Handcraft features (higher accuracy, requires expertise)                │
│  • Use FM embeddings (lower accuracy, zero expertise needed)               │
│  • Hybrid: FM preprocessing + handcrafted features                         │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  PHASE 3: Registry Update (1 hour)                                         │
│  ─────────────────────────────────                                         │
│                                                                            │
│  ┌────────────────────────────────────────────────────────────────────┐   │
│  │  configs/mlflow_registry/parameters/classification.yaml            │   │
│  │  ──────────────────────────────────────────────────────            │   │
│  │                                                                     │   │
│  │  # Update to YOUR method names                                      │   │
│  │  outlier_methods:                                                   │   │
│  │    - manual-gt              # Your ground truth                     │   │
│  │    - MOMENT-gt-finetune     # Keep FM methods                       │   │
│  │    - IsolationForest        # Add domain-specific methods           │   │
│  │    - vibration-threshold    # Your rule-based baseline              │   │
│  │                                                                     │   │
│  │  class_labels:                                                      │   │
│  │    - healthy                                                        │   │
│  │    - bearing_fault                                                  │   │
│  │    - imbalance                                                      │   │
│  └────────────────────────────────────────────────────────────────────┘   │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  PHASE 4: Threshold Calibration (2-4 hours)                                │
│  ──────────────────────────────────────────                                │
│                                                                            │
│  ┌────────────────────────────────────────────────────────────────────┐   │
│  │  src/preprocessing/thresholds.py                                    │   │
│  │  ───────────────────────────────                                    │   │
│  │                                                                     │   │
│  │  # PLR: blink if pupil_size < 1.5mm or > 9.0mm                     │   │
│  │  # Vibration: anomaly if amplitude > 10× baseline RMS              │   │
│  │  # Seismic: event if magnitude > 3σ from running average           │   │
│  │                                                                     │   │
│  │  OUTLIER_THRESHOLD = domain_specific_function()                    │   │
│  └────────────────────────────────────────────────────────────────────┘   │
│                                                                            │
│  This requires domain expertise or labeled validation data.                │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  PHASE 5: Validation (2-4 hours)                                           │
│  ───────────────────────────────                                           │
│                                                                            │
│  □ Run full pipeline on YOUR data                                          │
│  □ Compare foundation models vs traditional on YOUR signal                 │
│  □ Check if handcrafted beats embeddings (like PLR)                        │
│  □ Generate STRATOS-compliant evaluation metrics                           │
│                                                                            │
│  If FM embeddings beat handcrafted → lucky, use them                       │
│  If handcrafted beats embeddings → expected, invest in features            │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  WHAT YOU KEEP (Don't Touch)                                               │
│  ───────────────────────────                                               │
│                                                                            │
│  ✓ src/models/          - MOMENT, SAITS, CSDI wrappers work unchanged     │
│  ✓ src/evaluation/      - STRATOS metrics are domain-agnostic             │
│  ✓ src/viz/             - Plotting infrastructure                         │
│  ✓ src/stats/           - Calibration, DCA, uncertainty                   │
│  ✓ tests/test_figure_qa/- Figure quality assurance                        │
│  ✓ apps/                - React visualization                             │
│                                                                            │
│  Total: ~70% of codebase requires NO changes                               │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Text Content

### Title
"From PLR to Your Domain: Fork Guide"

### Caption
"A step-by-step roadmap for adapting the PLR repository. Phase 0 (setup): fork and verify. Phase 1 (data): define your schema and ground truth. Phase 2 (features): this is where domain knowledge lives—replace amplitude bins with your metrics. Phase 3 (registry): update method names. Phase 4 (thresholds): calibrate outlier detection. Phase 5 (validation): compare FMs vs traditional on your data. Approximately 70% of the codebase (model wrappers, evaluation, visualization) requires no changes."

---

## 6. Prompts for Nano Banana Pro

### Primary Prompt
```
Create a step-by-step fork/adaptation guide diagram.

FIVE PHASES with checkboxes:
Phase 0 (1 hour): Setup - fork, install, test, read docs
Phase 1 (2-4 hours): Data Layer - schema, config, database
Phase 2 (4-8 hours): Features - MOST EFFORT HERE with mapping table
Phase 3 (1 hour): Registry - update method names YAML
Phase 4 (2-4 hours): Thresholds - domain-specific calibration
Phase 5 (2-4 hours): Validation - run pipeline, compare methods

FOOTER - "What you keep":
List of unchanged directories (~70% of code)
src/models/, src/evaluation/, src/viz/, src/stats/, tests/, apps/

Style: Practical checklist format, clear time estimates
```

---

## 7. Alt Text

"Step-by-step fork guide with five phases. Phase 0 (1 hour): setup. Phase 1 (2-4 hours): data layer changes. Phase 2 (4-8 hours): feature engineering marked as most effort, with table mapping PLR features to vibration equivalents. Phase 3 (1 hour): registry update. Phase 4 (2-4 hours): threshold calibration. Phase 5 (2-4 hours): validation. Footer lists unchanged code comprising 70% of repository: model wrappers, evaluation, visualization, statistics."

---

## 8. Status

- [x] Draft created
- [ ] Generated
- [ ] Placed in documentation
