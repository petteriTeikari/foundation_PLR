# Figure Plan: fig-repo-53-outlier-detection-methods

**Target**: Repository documentation infographic
**Section**: `configs/OUTLIER_MODELS/` and `configs/mlflow_registry/`
**Purpose**: Visual guide to the 11 outlier detection methods in the registry
**Version**: 1.0

---

## Title

**Outlier Detection Methods: The 11 Registry Methods**

---

## Purpose

Help developers understand:
1. The exact 11 outlier methods in the registry (CRITICAL: not 17, not 15, EXACTLY 11)
2. How methods are categorized (Ground Truth, FM, Deep Learning, Traditional, Ensemble)
3. The config files that control each method
4. Why certain methods exist (ground truth baseline, ensemble combinations)

---

## Visual Layout (Categorized Grid)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  FIGURE TITLE: Outlier Detection Methods (11 Registry Methods)               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  GROUND TRUTH (1)                                                        ││
│  │  ┌─────────────────────────────────────────────────────────────────┐    ││
│  │  │  1. pupil-gt                                                     │    ││
│  │  │     Human-annotated blink/artifact masks                         │    ││
│  │  │     Config: N/A (reference standard)                             │    ││
│  │  └─────────────────────────────────────────────────────────────────┘    ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  FOUNDATION MODELS (3)                                                   ││
│  │  ┌───────────────────┐ ┌───────────────────┐ ┌───────────────────┐      ││
│  │  │ 2. MOMENT-gt-     │ │ 3. MOMENT-gt-     │ │ 4. UniTS-gt-      │      ││
│  │  │    finetune       │ │    zeroshot       │ │    finetune       │      ││
│  │  │                   │ │                   │ │                   │      ││
│  │  │ Fine-tuned on PLR │ │ Zero-shot         │ │ Fine-tuned        │      ││
│  │  │ outlier task      │ │ reconstruction    │ │ on PLR            │      ││
│  │  │                   │ │ anomaly scoring   │ │                   │      ││
│  │  │ Config: MOMENT.yaml│ │ Config: MOMENT.yaml│ │ Config: UniTS.yaml│      ││
│  │  └───────────────────┘ └───────────────────┘ └───────────────────┘      ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  DEEP LEARNING (1)                                                       ││
│  │  ┌─────────────────────────────────────────────────────────────────┐    ││
│  │  │  5. TimesNet-gt                                                  │    ││
│  │  │     Supervised deep learning (not foundation model)              │    ││
│  │  │     Config: TimesNet.yaml                                        │    ││
│  │  └─────────────────────────────────────────────────────────────────┘    ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  TRADITIONAL (4)                                                         ││
│  │  ┌───────────────┐ ┌───────────────┐ ┌───────────────┐ ┌───────────────┐││
│  │  │ 6. LOF        │ │ 7. OneClassSVM│ │ 8. PROPHET    │ │ 9. SubPCA     │││
│  │  │               │ │               │ │               │ │               │││
│  │  │ Local Outlier │ │ Support Vector│ │ Time series   │ │ Subspace PCA  │││
│  │  │ Factor        │ │ anomaly       │ │ decomposition │ │ anomaly       │││
│  │  │               │ │               │ │               │ │               │││
│  │  │ Config: lof.yaml│ │ Config: svm.yaml│ │ Config: prophet│ │ Config: subpca│││
│  │  └───────────────┘ └───────────────┘ └───────────────┘ └───────────────┘││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  ENSEMBLE (2)                                                            ││
│  │  ┌─────────────────────────────────────────────────────────────────┐    ││
│  │  │ 10. ensemble-LOF-MOMENT-OneClassSVM-PROPHET-SubPCA-TimesNet-    │    ││
│  │  │     UniTS-gt-finetune                                            │    ││
│  │  │     All methods combined (majority voting)                       │    ││
│  │  ├─────────────────────────────────────────────────────────────────┤    ││
│  │  │ 11. ensembleThresholded-MOMENT-TimesNet-UniTS-gt-finetune       │    ││
│  │  │     FM + DL methods only (thresholded voting)                    │    ││
│  │  └─────────────────────────────────────────────────────────────────┘    ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  REGISTRY SOURCE: configs/mlflow_registry/parameters/classification.yaml ││
│  │  CONFIG LOCATION: configs/OUTLIER_MODELS/*.yaml                          ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Content Elements

### Category Summary Table

| Category | Count | Methods |
|----------|-------|---------|
| Ground Truth | 1 | pupil-gt |
| Foundation Model | 3 | MOMENT-gt-finetune, MOMENT-gt-zeroshot, UniTS-gt-finetune |
| Deep Learning | 1 | TimesNet-gt |
| Traditional | 4 | LOF, OneClassSVM, PROPHET, SubPCA |
| Ensemble | 2 | ensemble-..., ensembleThresholded-... |
| **TOTAL** | **11** | |

### INVALID Methods (NOT in Registry)

These are GARBAGE from orphan MLflow runs - NEVER USE:
- `anomaly`
- `exclude`
- `MOMENT-orig-finetune`
- `UniTS-orig-finetune`
- `UniTS-orig-zeroshot`
- `TimesNet-orig`

---

## Key Messages

1. **EXACTLY 11 methods**: If code shows more, it's BROKEN
2. **Ground truth is the reference**: All other methods compared against pupil-gt
3. **Registry is single source of truth**: `configs/mlflow_registry/parameters/classification.yaml`
4. **Ensemble methods combine others**: Not independent algorithms

---

## Technical Specifications

- **Aspect ratio**: 16:10 (slightly taller for 5 categories)
- **Resolution**: 300 DPI
- **Background**: #FBF9F3 (Economist off-white)
- **Typography**: Sans-serif, dark grey (#333333)
- **Colour coding**: Category headers in semantic colors from style guide

---

## Data Source

- **Registry**: `configs/mlflow_registry/parameters/classification.yaml`
- **Config files**: `configs/OUTLIER_MODELS/*.yaml`

---

## Related Documentation

- **README to expand**: `configs/OUTLIER_MODELS/README.md` (currently 34 lines)
- **Registry README**: `configs/mlflow_registry/README.md`
- **Related infographic**: fig-repo-55 (Registry Pattern)

---

## References

- Breunig et al. 2000 "LOF: Identifying Density-Based Local Outliers"
- Schölkopf et al. 2001 "Estimating the Support of a High-Dimensional Distribution" (One-Class SVM)
- Taylor & Letham 2018 "Forecasting at Scale" (PROPHET)
- Goswami et al. 2024 "MOMENT" (MOMENT)
- Gao et al. 2024 "UniTS"

---

*Figure plan created: 2026-02-02*
*For: configs/OUTLIER_MODELS/ documentation - CRITICAL: exactly 11 methods*
