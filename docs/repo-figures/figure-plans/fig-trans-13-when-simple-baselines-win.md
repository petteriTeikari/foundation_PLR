# fig-trans-13: When Simple Baselines Win

**Status**: 📋 PLANNED
**Tier**: 3 - Alternative Approaches
**Target Persona**: Data scientists, ML engineers, budget-conscious analysts

---

## 1. Metadata

| Field | Value |
|-------|-------|
| Figure ID | fig-trans-13 |
| Type | Decision framework with scenario guide |
| Style | 75% manuscript + 25% Economist |
| Dimensions | 14" × 10" |
| Format | PDF (vector) + PNG (300 DPI) |

---

## 2. Purpose

Counter the "always use deep learning" narrative by providing a framework for when simple baselines (linear interpolation, moving average, ARIMA) are the right choice—based on data characteristics, constraints, and requirements.

---

## 3. Key Message

> "Foundation models have their place, but simple baselines often win. Know when simple is appropriate: small data, low SNR, interpretability requirements, or real-time constraints."

---

## 4. Literature Sources

| Source | Finding |
|--------|---------|
| Zeng 2023 | "Are Transformers Effective for Time Series Forecasting?" - Linear models competitive |
| Makridakis et al. 2022 | M5 Competition - Simple methods often competitive with deep learning |
| Hyndman & Athanasopoulos | "Forecasting: Principles and Practice" - Always benchmark simple first |

---

## 5. Visual Concept

```
┌────────────────────────────────────────────────────────────────────────────┐
│  WHEN SIMPLE BASELINES WIN                                                 │
│  A Decision Framework                                                      │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  THE BASELINE SPECTRUM                                                     │
│  ────────────────────                                                      │
│                                                                            │
│  SIMPLE                                              COMPLEX               │
│  ════════════════════════════════════════════════════════════════════     │
│                                                                            │
│  Constant/Mean    Linear    Moving    ARIMA    SAITS    MOMENT            │
│  Imputation       Interp    Average                     CSDI              │
│                                                                            │
│  • <1 sec compute │         │         │         │      • Minutes compute  │
│  • Fully interpret│         │         │         │      • Black box        │
│  • No training    │         │         │         │      • GPU required     │
│  • Always stable  │         │         │         │      • Hyperparameters  │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  WHEN TO USE SIMPLE BASELINES                                              │
│  ───────────────────────────                                               │
│                                                                            │
│  ✓ Small data (< 1000 training samples)                                   │
│    → Complex models overfit; linear methods generalize                     │
│                                                                            │
│  ✓ Low signal-to-noise ratio                                              │
│    → Foundation models may learn noise patterns                            │
│                                                                            │
│  ✓ Interpretability required (clinical, regulatory)                        │
│    → "Why did you predict 3.2mm?" needs a simple answer                   │
│                                                                            │
│  ✓ Real-time constraints (< 100ms latency)                                │
│    → Deep models too slow; linear is instant                              │
│                                                                            │
│  ✓ Sparse missing data (< 5% missing)                                     │
│    → Linear interpolation is often optimal for small gaps                  │
│                                                                            │
│  ✓ Debugging/baseline establishment                                       │
│    → Reviewers expect simple baseline comparisons                          │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  WHEN TO USE FOUNDATION MODELS                                             │
│  ─────────────────────────────                                             │
│                                                                            │
│  ✓ Large training corpus available (> 10,000 samples)                      │
│  ✓ Complex patterns (multi-scale periodicity, interactions)                │
│  ✓ Transfer learning needed (pretrain once, apply many)                    │
│  ✓ Long gaps (> 20% missing) with context-dependent reconstruction         │
│  ✓ Zero-shot cross-domain application (no labeled data)                    │
│  ✓ When simple baselines demonstrably fail                                 │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  THE PRACTITIONER'S PROTOCOL                                               │
│  ──────────────────────────                                                │
│                                                                            │
│  1. Always benchmark against simple baselines first                        │
│  2. If simple is within 10% of complex → use simple                        │
│  3. If simple loses by > 20% → consider foundation models                  │
│  4. Document the comparison (reviewers will ask)                           │
│  5. Consider compute/interpretability trade-offs                           │
│                                                                            │
│  "If you can't beat linear by a meaningful margin,                         │
│   you don't need deep learning."                                           │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  COMPARISON DIMENSIONS                                                     │
│  ─────────────────────                                                     │
│                                                                            │
│  ┌──────────────────────────────────────────────────────────────────┐     │
│  │                                                                   │     │
│  │  Dimension        │ Simple         │ Foundation Model            │     │
│  │  ─────────────────┼────────────────┼───────────────────────────  │     │
│  │  Compute time     │ Seconds        │ Minutes to hours            │     │
│  │  Interpretability │ Full           │ Limited (attention maps)    │     │
│  │  Training data    │ None/minimal   │ Thousands of samples        │     │
│  │  Hyperparameters  │ Few/none       │ Many (architecture + HP)    │     │
│  │  Deployment       │ CPU anywhere   │ GPU often required          │     │
│  │  Reproducibility  │ Deterministic  │ Seed-dependent              │     │
│  │                                                                   │     │
│  └──────────────────────────────────────────────────────────────────┘     │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Text Content

### Title
"When Simple Baselines Win"

### Caption
"Simple baselines often outperform complex models when data is limited, noise is high, or interpretability is required. The practitioner's protocol: benchmark against simple methods first, use complex only if they beat simple by a meaningful margin (>10-20%), and always document the comparison. Simple methods offer advantages in compute, interpretability, and deployment that foundation models cannot match."

---

## 7. Prompts for Nano Banana Pro

### Primary Prompt
```
Create a decision framework showing when simple baselines beat foundation models.

TOP - Baseline spectrum:
Simple (constant, linear, moving avg) → Complex (ARIMA, SAITS, MOMENT)
Show trade-offs at each end

MIDDLE LEFT - "When to use simple" checklist:
- Small data
- Low SNR
- Interpretability required
- Real-time constraints
- Sparse missing data
- Baseline establishment

MIDDLE RIGHT - "When to use FMs" checklist:
- Large training corpus
- Complex patterns
- Transfer learning
- Long gaps
- Zero-shot application

BOTTOM - Practitioner's protocol:
5-step process for method selection

TABLE - Comparison dimensions:
Compute, interpretability, training, hyperparams, deployment, reproducibility

Style: Framework-focused, no specific performance numbers
```

---

## 8. Alt Text

"Decision framework for choosing between simple baselines and foundation models. Top shows spectrum from simple (constant, linear, moving average) to complex (ARIMA, SAITS, MOMENT). Left checklist shows scenarios favoring simple: small data, low SNR, interpretability, real-time, sparse gaps. Right checklist shows FM scenarios: large corpus, complex patterns, transfer learning, long gaps, zero-shot. Bottom shows 5-step practitioner's protocol for method selection. Table compares dimensions: compute, interpretability, training, hyperparameters, deployment, reproducibility."

---

## 9. Status

- [x] Draft created
- [x] Revised to focus on decision framework, not specific results
- [ ] Generated
- [ ] Placed in documentation

## Note

Specific performance comparisons from experiments are in the manuscript, not this figure.
