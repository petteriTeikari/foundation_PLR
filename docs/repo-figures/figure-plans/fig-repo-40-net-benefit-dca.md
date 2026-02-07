# fig-repo-40: Net Benefit and Decision Curve Analysis

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repo-40 |
| **Title** | Net Benefit and Decision Curve Analysis |
| **Complexity Level** | L2 (Statistical concept) |
| **Target Persona** | Biostatistician, Clinician |
| **Location** | docs/concepts-for-researchers.md |
| **Priority** | P1 |
| **Aspect Ratio** | 16:10 |

## Purpose

Explain Net Benefit and DCA—the clinical utility metrics that answer "Is this model useful for decisions?"

## Key Message

"Net Benefit accounts for the trade-off between true positives (catching disease) and false positives (unnecessary treatment). A useful model beats both 'treat all' and 'treat none' strategies."

## Visual Concept

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    NET BENEFIT AND DECISION CURVE ANALYSIS                      │
│                    "Is the model useful for clinical decisions?"                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  THE PROBLEM WITH AUROC ALONE                                                   │
│  ════════════════════════════                                                   │
│                                                                                 │
│  High AUROC ≠ Useful model                                                      │
│                                                                                 │
│  A model with AUROC 0.90 could still be:                                        │
│  • Useless at clinical thresholds                                               │
│  • Worse than "treat everybody" strategy                                        │
│  • Worse than "treat nobody" strategy                                           │
│                                                                                 │
│  Net Benefit measures: "Does using this model lead to BETTER DECISIONS?"        │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  THE NET BENEFIT FORMULA                                                        │
│  ═══════════════════════                                                        │
│                                                                                 │
│  At threshold t:                                                                │
│                                                                                 │
│               True Positives     False Positives       t                        │
│  Net Benefit = ────────────── - ─────────────── × ────────                     │
│                     N                 N            (1 - t)                      │
│                                                                                 │
│  Where:                                                                         │
│  • N = total patients                                                           │
│  • t = treatment threshold (e.g., 0.15 = treat if risk > 15%)                   │
│  • t/(1-t) = the "exchange rate" between FP and TP                              │
│                                                                                 │
│  At t=15%: One missed glaucoma case = penalty equivalent to 5.67 false alarms   │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  DECISION CURVE ANALYSIS (DCA)                                                  │
│  ═════════════════════════════                                                  │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                         │   │
│  │  Net       0.20 ┤                                                       │   │
│  │  Benefit        │  ●──●──●──●──●                                        │   │
│  │                 │       ╲           ← MODEL (best!)                     │   │
│  │            0.15 ┤        ╲                                              │   │
│  │                 │         ●──●──●                                       │   │
│  │                 │              ╲                                        │   │
│  │            0.10 ┤               ╲                                       │   │
│  │                 │  ──────────────────────────── ← TREAT ALL             │   │
│  │                 │                      ╲         (declining line)       │   │
│  │            0.05 ┤                       ╲                               │   │
│  │                 │                        ╲                              │   │
│  │                 │                         ╲                             │   │
│  │            0.00 ┼════════════════════════════════════ ← TREAT NONE      │   │
│  │                 │                                     (always 0)        │   │
│  │           -0.05 ┤                                                       │   │
│  │                 └──────────────────────────────────────────────────     │   │
│  │                 0.05   0.10   0.15   0.20   0.25   0.30                 │   │
│  │                              Threshold probability                      │   │
│  │                                                                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  THREE STRATEGIES:                                                              │
│                                                                                 │
│  • TREAT NONE (black line at 0): Never treat. Net benefit = 0 always.           │
│  • TREAT ALL (declining line): Treat everyone. Decreases as threshold rises.    │
│  • MODEL (curve): Use model predictions. Should be ABOVE both baselines.        │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  INTERPRETING THE CURVES                                                        │
│  ═══════════════════════                                                        │
│                                                                                 │
│  │ Model Position       │ Interpretation                                      │ │
│  │ ──────────────────── │ ────────────────────────────────────────────────── │ │
│  │ Above both baselines │ ✅ Model is USEFUL at this threshold                │ │
│  │ Below treat-all      │ ⚠️ Better to just treat everyone                    │ │
│  │ Below treat-none     │ ❌ Model is HARMFUL (worse than no model)           │ │
│  │ Overlapping baselines│ ⚖️ Model provides no benefit                        │ │
│                                                                                 │
│  The vertical distance between model and best baseline = clinical benefit       │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  OUR RESULTS                                                                    │
│  ═══════════                                                                    │
│                                                                                 │
│  At clinical thresholds (5-20%):                                                │
│                                                                                 │
│  │ Threshold │ Net Benefit │ vs Treat-All │ vs Treat-None │                    │
│  │ ───────── │ ─────────── │ ──────────── │ ───────────── │                    │
│  │ 5%        │ 0.231       │ +0.043       │ +0.231        │                    │
│  │ 10%       │ 0.215       │ +0.078       │ +0.215        │                    │
│  │ 15%       │ 0.199       │ +0.089       │ +0.199        │                    │
│  │ 20%       │ 0.184       │ +0.094       │ +0.184        │                    │
│                                                                                 │
│  Model provides clinical benefit across all reasonable thresholds!              │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  WHY THIS MATTERS                                                               │
│  ════════════════                                                               │
│                                                                                 │
│  STRATOS guidelines (Van Calster 2024) recommend:                               │
│  "Clinical utility should be assessed using decision curves"                    │
│                                                                                 │
│  DCA answers the question that matters:                                         │
│  "If we use this model to decide who gets tested/treated,                       │
│   do patients end up better off?"                                               │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Content Elements

1. **Problem with AUROC**: Why utility matters
2. **Net Benefit formula**: Mathematical explanation
3. **DCA plot**: Three strategies with interpretation
4. **Curve interpretation table**: What different positions mean
5. **Our results**: Actual net benefit values at thresholds

## Text Content

### Title Text
"Net Benefit and DCA: Is the Model Clinically Useful?"

### Caption
Net Benefit measures clinical utility by weighing true positives against false positives at a given decision threshold. Decision Curve Analysis (DCA) plots net benefit across thresholds, comparing the model against two baselines: "treat all" (everyone gets intervention) and "treat none" (no intervention). A useful model stays above both baselines. Our model provides clinical benefit at thresholds 5-20%, with net benefit 0.199 at the clinically relevant 15% threshold.

## Prompts for Nano Banana Pro

### Style Prompt
DCA plot with three curves: treat-none (flat at 0), treat-all (declining), model (above both). Net benefit formula. Interpretation table. Results table with values at thresholds. Clean medical decision-making aesthetic.

### Content Prompt
Create a Net Benefit/DCA explanation diagram:

**TOP - Problem**:
- "High AUROC ≠ Useful model"
- What DCA measures

**MIDDLE - DCA Plot**:
- Three curves: treat-none (0 line), treat-all (declining), model (above both)
- Threshold on X-axis (5-30%), Net Benefit on Y-axis
- Labels for each curve

**BOTTOM LEFT - Interpretation Table**:
- Four rows: above both, below treat-all, below treat-none, overlapping

**BOTTOM RIGHT - Results Table**:
- Net benefit at 5%, 10%, 15%, 20% thresholds

## Alt Text

Net Benefit and Decision Curve Analysis explanation. DCA plot shows three strategies: treat-none (horizontal line at 0), treat-all (declining line), and model (curve above both baselines). Model above both baselines indicates clinical utility. Net Benefit formula: TP/N - FP/N × t/(1-t). Results table shows net benefit at thresholds: 0.231 at 5%, 0.215 at 10%, 0.199 at 15%, 0.184 at 20%. Model provides clinical benefit across all reasonable thresholds.

## Status

- [x] Draft created
- [ ] Generated
- [ ] Placed in docs/concepts-for-researchers.md
