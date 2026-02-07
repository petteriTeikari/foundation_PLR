# fig-trans-19: The Data Quality Manifesto

**Status**: 📋 PLANNED
**Tier**: 4 - Repository Patterns
**Target Persona**: All technical professionals working with data

---

## 1. Metadata

| Field | Value |
|-------|-------|
| Figure ID | fig-trans-19 |
| Type | Manifesto / principles diagram |
| Style | 75% manuscript + 25% Economist |
| Dimensions | 12" × 14" |
| Format | PDF (vector) + PNG (300 DPI) |

---

## 2. Purpose

Distill the lessons learned from the PLR project into universal data quality principles that apply across domains—from biosignals to business analytics.

---

## 3. Key Message

> "Garbage in, garbage out. But 'garbage' isn't always obvious. Here are the principles we learned the hard way: fix issues at the source, validate against ground truth, document every assumption."

---

## 4. Context

Synthesizes lessons from multiple meta-learnings:
- CRITICAL-FAILURE-001: Synthetic data in figures
- CRITICAL-FAILURE-002: Mixed featurization in extraction
- CRITICAL-FAILURE-003: Computation in visualization
- Various hardcoding failures

---

## 5. Visual Concept

```
┌────────────────────────────────────────────────────────────────────────────┐
│  THE DATA QUALITY MANIFESTO                                                │
│  Principles Learned the Hard Way                                           │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  PRINCIPLE 1: Fix Issues at the Source                                     │
│  ────────────────────────────────────                                      │
│                                                                            │
│  ┌────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │  MLflow → DuckDB → CSV → R → Figure                                │   │
│  │    ↑         ↑       ↑     ↑      ↑                                │   │
│  │  SOURCE    FIX      NO    NO     NO                                │   │
│  │            HERE     FIX   FIX    FIX                               │   │
│  │                                                                     │   │
│  │  If data is wrong in DuckDB, don't filter in R—fix the extraction.│   │
│  │  Downstream fixes mask problems and break reproducibility.         │   │
│  │                                                                     │   │
│  └────────────────────────────────────────────────────────────────────┘   │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  PRINCIPLE 2: Validate Against Ground Truth                                │
│  ──────────────────────────────────────────                                │
│                                                                            │
│  ┌────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │  Expected values (memorize these):                                  │   │
│  │                                                                     │   │
│  │  Configuration              │ Expected │ If Different              │   │
│  │  ──────────────────────────┼──────────┼─────────────────────────   │   │
│  │  GT + GT + CatBoost        │ 0.911    │ CRITICAL - investigate     │   │
│  │  Ensemble + CSDI + CatBoost│ 0.913    │ CRITICAL - investigate     │   │
│  │  Total configs             │ 316      │ Duplicates or missing      │   │
│  │  Outlier methods           │ 11       │ Registry violation         │   │
│  │                                                                     │   │
│  │  Every analysis should start: "Does GT+GT give 0.911?"             │   │
│  │                                                                     │   │
│  └────────────────────────────────────────────────────────────────────┘   │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  PRINCIPLE 3: Never Use Synthetic Data for Real Figures                    │
│  ──────────────────────────────────────────────────────                    │
│                                                                            │
│  ┌────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │  ❌ BANNED: np.random.seed(42); data = generate_fake_predictions()│   │
│  │                                                                     │   │
│  │  Symptoms of synthetic data in figures:                             │   │
│  │  • All models show identical curves (same random seed)              │   │
│  │  • Calibration is "too perfect"                                     │   │
│  │  • Round numbers everywhere (0.85, 0.90 exactly)                    │   │
│  │                                                                     │   │
│  │  If you're tempted to use synthetic → you haven't loaded real data │   │
│  │                                                                     │   │
│  └────────────────────────────────────────────────────────────────────┘   │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  PRINCIPLE 4: Computation Belongs in Extraction, Not Visualization         │
│  ────────────────────────────────────────────────────────────────────      │
│                                                                            │
│  ┌────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │  [Extraction]           [Visualization]                            │   │
│  │  ─────────────          ───────────────                            │   │
│  │  • Compute metrics      • READ from DuckDB                         │   │
│  │  • Run statistics       • Format for display                       │   │
│  │  • Save to DuckDB       • NEVER compute metrics                    │   │
│  │                                                                     │   │
│  │  Why: Visualization code runs many times during figure iteration.  │   │
│  │  Computation should happen ONCE and be cached.                     │   │
│  │                                                                     │   │
│  └────────────────────────────────────────────────────────────────────┘   │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  PRINCIPLE 5: Document Every Assumption                                    │
│  ──────────────────────────────────────                                    │
│                                                                            │
│  ┌────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │  glaucoma_params:                                                   │   │
│  │    prevalence: 0.0354     # Tham 2014 DOI:10.1016/j.ophtha...      │   │
│  │    target_sensitivity: 0.862  # Najjar 2023 Table 2                │   │
│  │                                                                     │   │
│  │  "Why 0.0354?"  →  Comment has citation                             │   │
│  │  "Why not 0.04?" →  Would need to read Tham 2014                   │   │
│  │                                                                     │   │
│  │  Future you will not remember. Document for future you.             │   │
│  │                                                                     │   │
│  └────────────────────────────────────────────────────────────────────┘   │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  PRINCIPLE 6: Test Your Figures                                            │
│  ──────────────────────────────                                            │
│                                                                            │
│  ┌────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │  pytest tests/test_figure_qa/ -v                                   │   │
│  │                                                                     │   │
│  │  Tests catch:                                                       │   │
│  │  • P0: Synthetic data, identical predictions                        │   │
│  │  • P1: Invalid metrics, visual overlap                              │   │
│  │  • P2: DPI, dimensions                                              │   │
│  │  • P3: Accessibility (colorblind, contrast)                         │   │
│  │                                                                     │   │
│  │  "It looks fine" is not QA. Automated tests are.                   │   │
│  │                                                                     │   │
│  └────────────────────────────────────────────────────────────────────┘   │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  THE SHORT VERSION                                                         │
│  ─────────────────                                                         │
│                                                                            │
│  1. Fix at source, not downstream                                          │
│  2. Know your expected values                                              │
│  3. Real data only for real figures                                        │
│  4. Compute once, visualize many                                           │
│  5. Document why, not just what                                            │
│  6. Test, don't eyeball                                                    │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Text Content

### Title
"The Data Quality Manifesto"

### Caption
"Six principles learned the hard way from the PLR project: (1) Fix issues at the source, not downstream. (2) Validate against known ground truth values. (3) Never use synthetic data for scientific figures. (4) Computation belongs in extraction, not visualization. (5) Document every assumption with citations. (6) Test figures with automated QA, not eyeballing. These principles apply universally—whether you're analyzing biosignals, business metrics, or industrial sensors."

---

## 7. Prompts for Nano Banana Pro

### Primary Prompt
```
Create a manifesto-style principles diagram.

SIX PRINCIPLES in boxes:

1. Fix at Source:
Pipeline diagram showing where to fix (extraction) vs where not to (downstream)

2. Validate Against Ground Truth:
Table of expected values (GT=0.911, configs=316, methods=11)

3. No Synthetic Data:
Warning about identical curves, "too perfect" calibration

4. Computation in Extraction:
Two-column showing extraction (compute) vs visualization (read)

5. Document Assumptions:
YAML example with comments and citations

6. Test Figures:
pytest command and what tests catch (P0-P3 priorities)

FOOTER - Short version:
Six bullet points summarizing principles

Style: Manifesto/declaration format, authoritative but practical
```

---

## 8. Alt Text

"Data quality manifesto with six principles. Principle 1: fix issues at source, with pipeline diagram. Principle 2: validate against ground truth, with expected values table. Principle 3: no synthetic data, warning about identical curves. Principle 4: computation in extraction, not visualization. Principle 5: document assumptions with citations. Principle 6: test figures with pytest. Footer provides six-point summary. Principles derived from PLR project failures and applicable universally."

---

## 9. Status

- [x] Draft created
- [ ] Generated
- [ ] Placed in documentation
