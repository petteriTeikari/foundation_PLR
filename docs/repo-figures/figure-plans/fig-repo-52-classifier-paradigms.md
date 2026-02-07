# Figure Plan: fig-repo-52-classifier-paradigms

**Target**: Repository documentation infographic
**Section**: Conceptual understanding of classifier evolution
**Purpose**: Visual guide to classifier paradigm evolution (Linear → Trees → Foundation Models)
**Version**: 1.0

---

## Title

**Classifier Paradigms: From Linear Models to Foundation Models**

---

## Purpose

Help developers understand:
1. The conceptual evolution of tabular classifiers
2. Why we test multiple paradigms (not just "the best")
3. Trade-offs between paradigms (interpretability, data requirements, compute)
4. Where TabPFN fits in the landscape

---

## Visual Layout (Timeline Evolution)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  FIGURE TITLE: Classifier Paradigms: From Linear Models to Foundation Models │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  TIMELINE: Evolution of Tabular Classification                               │
│                                                                              │
│  1950s              1990s              2010s              2020s              │
│    │                  │                  │                  │                │
│    ▼                  ▼                  ▼                  ▼                │
│  ┌────────────┐   ┌────────────┐   ┌────────────┐   ┌────────────┐         │
│  │   LINEAR   │   │    TREE    │   │  GRADIENT  │   │ FOUNDATION │         │
│  │   MODELS   │   │   MODELS   │   │  BOOSTING  │   │   MODELS   │         │
│  │            │   │            │   │            │   │            │         │
│  │ LogReg     │──▷│ CART, RF   │──▷│ XGBoost    │──▷│ TabPFN     │         │
│  │ LDA        │   │            │   │ CatBoost   │   │ TabM       │         │
│  │            │   │            │   │ LightGBM   │   │            │         │
│  └────────────┘   └────────────┘   └────────────┘   └────────────┘         │
│        │                │                │                │                 │
│        ▼                ▼                ▼                ▼                 │
│  ┌────────────┐   ┌────────────┐   ┌────────────┐   ┌────────────┐         │
│  │PROPERTIES  │   │PROPERTIES  │   │PROPERTIES  │   │PROPERTIES  │         │
│  │            │   │            │   │            │   │            │         │
│  │✓Interpretable│ │✓Non-linear │   │✓State-of-art│  │✓Zero-shot  │         │
│  │✓Fast       │   │✓Robust     │   │✓Handles    │   │✓Pretrained │         │
│  │✓Calibrated │   │✗Data hungry│   │ missing    │   │✗Large N    │         │
│  │✗Linear only│   │            │   │✗Tuning     │   │ unseen     │         │
│  └────────────┘   └────────────┘   └────────────┘   └────────────┘         │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│  DATA REQUIREMENTS                                                           │
│                                                                              │
│  LogReg      ████░░░░░░  N > 10 per feature                                 │
│  Trees       ██████░░░░  N > 100                                            │
│  Boosting    ████████░░  N > 500 (optimal)                                  │
│  TabPFN      ██████████  N ≤ 10,000 (context limit)                         │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│  OUR STUDY: N=208 → All paradigms viable, Boosting expected to dominate     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Content Elements

### Paradigm Comparison Table

| Paradigm | Representative | Strengths | Weaknesses | Our Use |
|----------|---------------|-----------|------------|---------|
| **Linear** | Logistic Regression | Interpretable, calibrated, fast | Linear decision boundary | Baseline |
| **Tree** | Random Forest | Non-linear, robust to outliers | Data hungry, unstable | Not used |
| **Boosting** | CatBoost, XGBoost | State-of-art accuracy, handles missing | Requires tuning | Primary |
| **Foundation** | TabPFN | Zero-shot, no tuning | N ≤ 10,000 limit | Comparison |

### Data Requirement Bars

Visual representation of minimum N for effective use:
- LogReg: 10 per feature (~50-100 total)
- Trees: ~100
- Boosting: ~500 optimal
- TabPFN: Works at any N ≤ 10,000

### Why We Test Multiple Paradigms

1. **Scientific rigor**: Can't claim "X is best" without comparison
2. **Sample size sensitivity**: N=208 may favor different methods than N=20,000
3. **TabPFN novelty**: Test if foundation model paradigm transfers to medical tabular data
4. **Baseline grounding**: LogReg shows improvement over interpretable baseline

---

## Key Messages

1. **Evolution not replacement**: Each paradigm has strengths
2. **Boosting dominates** for mid-size tabular data (N=100-10,000)
3. **TabPFN is zero-shot**: No hyperparameter tuning, instant inference
4. **Our N=208**: All paradigms viable, boosting expected to perform best

---

## Technical Specifications

- **Aspect ratio**: 16:9 (landscape)
- **Resolution**: 300 DPI
- **Background**: #FBF9F3 (Economist off-white)
- **Typography**: Sans-serif, dark grey (#333333)
- **Generation method**: Mermaid or manual design

---

## Anti-Patterns to Avoid

- **NO performance numbers**: This is methodology, not results
- **NO "X is better than Y"**: Present paradigms objectively
- **NO leaderboard**: Not a comparison of accuracy

---

## Related Documentation

- **Config files**: `configs/CLS_HYPERPARAMS/*.yaml`, `configs/CLS_MODELS/*.yaml`
- **Related infographic**: fig-repo-51 (Classifier Config Architecture - technical)

---

## References

- Hastie et al. "Elements of Statistical Learning" - Chapter 4 (Linear Methods)
- Breiman 2001 "Random Forests"
- Chen & Guestrin 2016 "XGBoost"
- Hollmann et al. 2023 "TabPFN"
- Grinsztajn et al. 2022 "Why do tree-based models still outperform deep learning on tabular data?"

---

*Figure plan created: 2026-02-02*
*For: Conceptual understanding of classifier paradigms*
