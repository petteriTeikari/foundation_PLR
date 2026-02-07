# fig-trans-15: PLR Code: What's Domain-Specific?

**Status**: 📋 PLANNED
**Tier**: 4 - Repository Patterns
**Target Persona**: Developers who want to fork/adapt the PLR repository

---

## 1. Metadata

| Field | Value |
|-------|-------|
| Figure ID | fig-trans-15 |
| Type | Code architecture diagram |
| Style | 75% manuscript + 25% Economist |
| Dimensions | 14" × 12" |
| Format | PDF (vector) + PNG (300 DPI) |

---

## 2. Purpose

Guide developers who want to adapt the PLR repository to their own domain by clearly separating domain-specific code (features, thresholds, labels) from domain-agnostic code (pipelines, configurations, infrastructure).

---

## 3. Key Message

> "70% of this repository is domain-agnostic infrastructure. If you're building a preprocessing pipeline for vibration, seismic, or any dense signal, you only need to replace the 30% that's PLR-specific."

---

## 4. Visual Concept

```
┌────────────────────────────────────────────────────────────────────────────┐
│  PLR CODE: What's Domain-Specific?                                         │
│  A Guide to Forking This Repository                                        │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  REPOSITORY STRUCTURE                                                      │
│  ───────────────────                                                       │
│                                                                            │
│  ┌─────────────────────────────────────────────────────────────────┐      │
│  │                                                                  │      │
│  │  foundation_PLR/                                                 │      │
│  │  ├── src/                                                        │      │
│  │  │   ├── data_io/           [AGNOSTIC]  Data loading, DuckDB    │      │
│  │  │   ├── preprocessing/     [MIXED]     Outlier/imputation      │      │
│  │  │   │   ├── outlier_base.py   ← Agnostic                       │      │
│  │  │   │   ├── plr_thresholds.py ← DOMAIN-SPECIFIC                │      │
│  │  │   ├── features/          [DOMAIN-SPECIFIC] PLR biomarkers    │      │
│  │  │   │   ├── amplitude_bins.py ← Change for your signal         │      │
│  │  │   │   ├── latency_pipr.py   ← PLR-specific                   │      │
│  │  │   ├── models/            [AGNOSTIC]  MOMENT, SAITS wrappers  │      │
│  │  │   ├── evaluation/        [AGNOSTIC]  STRATOS metrics         │      │
│  │  │   ├── viz/               [AGNOSTIC]  Plotting infrastructure │      │
│  │  │   └── stats/             [AGNOSTIC]  Calibration, DCA        │      │
│  │  ├── configs/               [MIXED]     Registry, parameters    │      │
│  │  │   ├── mlflow_registry/      ← Change method names            │      │
│  │  │   ├── VISUALIZATION/        ← Agnostic                       │      │
│  │  │   └── defaults.yaml         ← Change prevalence, etc.        │      │
│  │  ├── tests/                 [AGNOSTIC]  Test infrastructure     │      │
│  │  └── apps/                  [AGNOSTIC]  React visualization     │      │
│  │                                                                  │      │
│  └─────────────────────────────────────────────────────────────────┘      │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  WHAT TO CHANGE BY LAYER                                                   │
│  ───────────────────────                                                   │
│                                                                            │
│  Layer              │ Effort │ What to Change                              │
│  ───────────────────┼────────┼────────────────────────────────────────────│
│  Feature extraction │ HIGH   │ Define YOUR domain's features              │
│  (30% of code)      │        │ Amplitude bins → your metrics              │
│                     │        │ PIPR → your biomarkers                      │
│  ───────────────────┼────────┼────────────────────────────────────────────│
│  Thresholds/labels  │ MEDIUM │ Outlier detection thresholds               │
│  (10% of code)      │        │ Classification labels                      │
│                     │        │ Prevalence in configs                       │
│  ───────────────────┼────────┼────────────────────────────────────────────│
│  Data loading       │ LOW    │ Database schema (if different)             │
│  (10% of code)      │        │ File formats                               │
│  ───────────────────┼────────┼────────────────────────────────────────────│
│  Everything else    │ NONE   │ Keep as-is                                 │
│  (50% of code)      │        │ Pipeline, evaluation, viz                  │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  EXAMPLE: Adapting to Vibration Monitoring                                 │
│  ─────────────────────────────────────────                                 │
│                                                                            │
│  Replace:                                                                  │
│  ┌─────────────────────────────────────────────────────────────────┐      │
│  │ PLR                          │ Vibration                        │      │
│  │ ────────────────────────────┼────────────────────────────────  │      │
│  │ amplitude_bins.py           │ fft_features.py                  │      │
│  │ latency_pipr.py             │ bearing_frequencies.py           │      │
│  │ plr_thresholds.py           │ vibration_thresholds.py          │      │
│  │ glaucoma_prevalence: 0.035  │ bearing_failure_rate: 0.02       │      │
│  │ class: control/glaucoma     │ class: healthy/degraded          │      │
│  └─────────────────────────────────────────────────────────────────┘      │
│                                                                            │
│  Keep:                                                                     │
│  • All MOMENT/SAITS/CSDI wrappers                                          │
│  • All STRATOS evaluation metrics                                          │
│  • All visualization infrastructure                                        │
│  • All configuration loading                                               │
│  • All pipeline orchestration                                              │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  FILES TO READ FIRST                                                       │
│  ───────────────────                                                       │
│                                                                            │
│  1. ARCHITECTURE.md    - Overall system design                             │
│  2. CLAUDE.md          - Research question and constraints                 │
│  3. src/features/      - Where domain knowledge lives                      │
│  4. configs/defaults.yaml - Where to change parameters                     │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Text Content

### Title
"PLR Code: What's Domain-Specific?"

### Caption
"A guide for developers adapting the PLR repository. Approximately 70% of the codebase is domain-agnostic infrastructure (pipeline orchestration, foundation model wrappers, STRATOS evaluation, visualization). Only 30% requires domain adaptation: feature extraction (amplitude bins → your metrics), classification thresholds, and label definitions. The MOMENT, SAITS, and CSDI wrappers work unchanged for any dense time series."

---

## 6. Prompts for Nano Banana Pro

### Primary Prompt
```
Create a code architecture diagram showing domain-specific vs agnostic parts.

TOP - Directory tree:
Show foundation_PLR/ structure
Color-code: Green = agnostic, Orange = mixed, Red = domain-specific
Mark src/features/ and threshold files as red

MIDDLE - Table by layer:
Feature extraction (HIGH effort, 30%)
Thresholds/labels (MEDIUM, 10%)
Data loading (LOW, 10%)
Everything else (NONE, 50%)

BOTTOM LEFT - Adaptation example:
PLR → Vibration mapping
(amplitude_bins → fft_features, etc.)

BOTTOM RIGHT - Files to read first:
ARCHITECTURE.md, CLAUDE.md, src/features/, configs/

Style: Developer documentation, clean hierarchy
```

---

## 7. Alt Text

"Code architecture diagram showing domain-specific versus agnostic parts of the PLR repository. Directory tree color-codes files: green for agnostic (data_io, models, evaluation), orange for mixed (preprocessing, configs), red for domain-specific (features, thresholds). Table shows effort by layer: 30% high effort for features, 10% medium for thresholds, 50% zero effort for infrastructure. Example maps PLR to vibration: amplitude_bins becomes fft_features. Bottom lists files to read first: ARCHITECTURE.md, CLAUDE.md."

---

## 8. Status

- [x] Draft created
- [ ] Generated
- [ ] Placed in documentation
