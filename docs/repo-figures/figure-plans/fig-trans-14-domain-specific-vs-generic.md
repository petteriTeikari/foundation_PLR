# fig-trans-14: Domain-Specific vs Generic Models

**Status**: 📋 PLANNED
**Tier**: 3 - Alternative Approaches
**Target Persona**: ML engineers, research scientists, applied data scientists

---

## 1. Metadata

| Field | Value |
|-------|-------|
| Figure ID | fig-trans-14 |
| Type | Trade-off diagram with decision matrix |
| Style | 75% manuscript + 25% Economist |
| Dimensions | 14" × 10" |
| Format | PDF (vector) + PNG (300 DPI) |

---

## 2. Purpose

Explain the trade-off between generic foundation models (MOMENT, TimesFM) and domain-specific models (EchoNet, specialized feature engineering)—when to use each, based on context rather than specific performance numbers.

---

## 3. Key Message

> "Generic foundation models trade domain knowledge for breadth. Choose based on stakes, expertise, and development phase—not just accuracy numbers."

---

## 4. Literature Sources

| Source | Finding |
|--------|---------|
| EchoNet-Dynamic (Ouyang 2020) | Domain-specific architectures for cardiac analysis |
| PhysioNet Challenge | Domain-specific preprocessing essential for clinical signals |
| Grinsztajn 2022 | Tree-based models often beat neural networks on tabular/domain data |
| Transfer learning literature | Domain shift degrades performance |

---

## 5. Visual Concept

```
┌────────────────────────────────────────────────────────────────────────────┐
│  DOMAIN-SPECIFIC vs GENERIC MODELS                                         │
│  The Breadth-Depth Trade-off                                               │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  THE TRADE-OFF SPECTRUM                                                    │
│  ──────────────────────                                                    │
│                                                                            │
│  GENERIC                                              DOMAIN-SPECIFIC      │
│  ════════════════════════════════════════════════════════════════════     │
│                                                                            │
│  MOMENT, TimesFM                                     Handcrafted Features  │
│  UniTS, Chronos                                      EchoNet-Dynamic       │
│  (pretrained on 1B+ points)                          (cardiac-specific)    │
│                                                                            │
│  • Zero-shot capable      ◄──────────────────────►   • Interpretable       │
│  • Works across domains                              • Clinically meaningful│
│  • No domain expertise                               • Sample efficient    │
│  • Rapid prototyping                                 • Smaller models      │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  GENERIC MODELS                       DOMAIN-SPECIFIC MODELS               │
│  ──────────────                       ──────────────────────               │
│                                                                            │
│  ADVANTAGES:                          ADVANTAGES:                          │
│  ✓ Zero-shot capability               ✓ Encodes domain physics/physiology │
│  ✓ Works across domains               ✓ Interpretable features            │
│  ✓ No domain expertise needed         ✓ Often more sample efficient       │
│  ✓ Rapid prototyping                  ✓ Smaller, deployable models        │
│  ✓ Transfer learning possible         ✓ Regulatory-friendly               │
│                                                                            │
│  LIMITATIONS:                         LIMITATIONS:                         │
│  ✗ May miss domain patterns           ✗ Requires domain expertise         │
│  ✗ Opaque predictions                 ✗ Domain-locked (no transfer)       │
│  ✗ Often large models                 ✗ Feature engineering effort        │
│  ✗ Domain shift degrades perf         ✗ May miss cross-domain patterns    │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  DECISION MATRIX                                                           │
│  ───────────────                                                           │
│                                                                            │
│                    │ Have Domain   │ No Domain                             │
│                    │ Expertise     │ Expertise                             │
│  ──────────────────┼───────────────┼───────────────────────────────────    │
│  High Stakes       │ DOMAIN-       │ Consult expert, then                  │
│  (medical, safety) │ SPECIFIC      │ domain-specific                       │
│  ──────────────────┼───────────────┼───────────────────────────────────    │
│  Low Stakes        │ Domain for    │ GENERIC FM                            │
│  (exploratory)     │ best results  │ (good enough)                         │
│  ──────────────────┼───────────────┼───────────────────────────────────    │
│  Rapid prototyping │ Generic first,│ GENERIC FM                            │
│  (POC phase)       │ then domain   │ (speed > accuracy)                    │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  WHY DOMAIN KNOWLEDGE MATTERS                                              │
│  ────────────────────────────                                              │
│                                                                            │
│  1. INDUCTIVE BIAS                                                         │
│     Domain features encode physics/physiology that models must learn       │
│     from scratch. Pre-encoded = fewer samples needed.                      │
│                                                                            │
│  2. SAMPLE EFFICIENCY                                                      │
│     With right representation, fewer examples needed for good performance  │
│                                                                            │
│  3. INTERPRETABILITY                                                       │
│     "PIPR latency is elevated" means something clinically                  │
│     "dimension_472 is elevated" means nothing                              │
│                                                                            │
│  4. DEBUGGING                                                              │
│     Domain features show WHERE the model fails                             │
│     Embedding failures are opaque                                          │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  THE PRACTITIONER'S RULE                                                   │
│  ───────────────────────                                                   │
│                                                                            │
│  • Exploration phase    →  Generic (fast iteration)                        │
│  • Production phase     →  Domain-specific (reliability)                   │
│  • High-stakes domains  →  Always involve domain experts                   │
│  • New domain, no data  →  Generic + careful validation                    │
│                                                                            │
│  "Generic for exploration, domain-specific for production"                 │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Text Content

### Title
"Domain-Specific vs Generic Models: The Breadth-Depth Trade-off"

### Caption
"Generic foundation models (MOMENT, TimesFM) trade domain knowledge for breadth, while domain-specific approaches encode physics and physiology. The choice depends on stakes, expertise, and development phase. High-stakes applications (medical, safety) warrant domain-specific models with interpretable features. Rapid prototyping can use generic FMs for speed. Rule of thumb: generic for exploration, domain-specific for production."

---

## 7. Prompts for Nano Banana Pro

### Primary Prompt
```
Create a trade-off diagram for domain-specific vs generic models.

TOP - Trade-off spectrum:
Left side: Generic (MOMENT, TimesFM)
Right side: Domain-specific (handcrafted, EchoNet)
Show characteristics at each end

MIDDLE - Two-column comparison:
Left: Generic advantages/limitations
Right: Domain-specific advantages/limitations

BOTTOM LEFT - Decision matrix:
Axes: Domain expertise (yes/no) vs Stakes (high/low/prototyping)
Show which approach for each quadrant

BOTTOM RIGHT - Why domain matters:
Inductive bias, sample efficiency, interpretability, debugging

FOOTER - Practitioner's rule

Style: Balanced presentation, framework-focused, no specific accuracy numbers
```

---

## 8. Alt Text

"Trade-off diagram comparing domain-specific versus generic foundation models. Top shows spectrum from generic (MOMENT, TimesFM) to domain-specific (handcrafted features, EchoNet). Middle compares advantages and limitations: generic offers zero-shot and breadth but is opaque; domain-specific offers interpretability and efficiency but requires expertise. Decision matrix shows high-stakes applications favor domain-specific while prototyping can use generic. Bottom explains why domain knowledge matters: inductive bias, sample efficiency, interpretability, debugging."

---

## 9. Status

- [x] Draft created
- [x] Revised to focus on trade-off framework, not specific results
- [ ] Generated
- [ ] Placed in documentation

## Note

Specific performance comparisons from experiments are in the manuscript, not this figure.
