# Knowledge Synthesis

A structured overview of key concepts, relationships, and decisions in the Foundation PLR project.

## Core Concepts

### Research Question

```
How do preprocessing choices affect downstream classification?
         │
         ├── Outlier Detection (11 methods)
         │         │
         │         ├── Traditional: LOF, OneClassSVM, SubPCA
         │         ├── Foundation Models: MOMENT, UniTS, TimesNet
         │         └── Ensembles
         │
         ├── Imputation (7 methods)
         │         │
         │         ├── Traditional: linear
         │         ├── Deep Learning: SAITS, CSDI, TimesNet
         │         └── Foundation Models: MOMENT
         │
         └── Classification (FIXED: CatBoost)
                   │
                   └── STRATOS Metrics
                             │
                             ├── Discrimination: AUROC
                             ├── Calibration: slope, intercept, O:E
                             ├── Overall: Brier, Scaled Brier
                             └── Clinical: Net Benefit, DCA
```

## Key Relationships

### Error Propagation Chain

```
Outlier Detection Errors
        │
        ▼
   Missed artifacts included in signal
        │
        ▼
   Imputation Errors
        │
        ▼
   Incorrect reconstructed values
        │
        ▼
   Feature Extraction Errors
        │
        ▼
   Distorted amplitude bins / latencies
        │
        ▼
   Classification Degradation
        │
        ▼
   Reduced AUROC, poor calibration
```

### Data Flow

```
SERI_PLR_GLAUCOMA.db (507 subjects)
        │
        ├──▶ 507 subjects for preprocessing evaluation
        │
        └──▶ 208 subjects for classification (labeled)
                    │
                    ├── 152 Control
                    └── 56 Glaucoma
```

## Key Decisions

### Why CatBoost is Fixed

| Decision | Rationale |
|----------|-----------|
| Best performance | AUROC 0.913 with ground truth |
| Stable | Consistent across bootstrap |
| Not the research question | We study preprocessing, not classifiers |

### Why Handcrafted Features

| Feature Type | AUROC | Decision |
|--------------|-------|----------|
| Handcrafted (bins + latency) | 0.830 | **USE THIS** |
| FM Embeddings | 0.740 | Underperforms by 9pp |

### Why STRATOS Metrics

| Old Approach | Problem | STRATOS Approach |
|--------------|---------|------------------|
| AUROC only | Ignores calibration | Full metric set |
| F1 score | Improper scoring | Net Benefit |
| Accuracy | Threshold-dependent | DCA curves |

## Concept Map

```
┌─────────────────────────────────────────────────────────────────┐
│                    FOUNDATION PLR PROJECT                        │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
   ┌─────────┐          ┌─────────┐          ┌─────────┐
   │  Data   │          │ Methods │          │ Metrics │
   └────┬────┘          └────┬────┘          └────┬────┘
        │                    │                    │
   ┌────┴────┐          ┌────┴────┐          ┌────┴────┐
   │ Najjar  │          │ Outlier │          │ STRATOS │
   │ 2023    │          │ 15 types│          │ Van     │
   │ N=507   │          │         │          │ Calster │
   └─────────┘          │ Impute  │          │ 2024    │
                        │ 7 types │          └─────────┘
                        │         │
                        │ Classify│
                        │ CatBoost│
                        └─────────┘
```

## Critical File Relationships

```
configs/VISUALIZATION/plot_hyperparam_combos.yaml
        │
        ├──▶ src/viz/config_loader.py
        │           │
        │           └──▶ src/viz/*.py (figure modules)
        │
        └──▶ .claude/CLAUDE.md (behavior contract)

src/viz/metric_registry.py
        │
        └──▶ All figure modules use this for metric definitions

src/stats/
        │
        ├── calibration_extended.py ──▶ STRATOS calibration
        ├── clinical_utility.py ──▶ Net Benefit, DCA
        ├── scaled_brier.py ──▶ IPA
        └── pminternal_wrapper.py ──▶ R integration (Riley 2023)
```

## Vocabulary

| Term | Meaning |
|------|---------|
| PLR | Pupillary Light Reflex |
| AUROC | Area Under ROC Curve |
| DCA | Decision Curve Analysis |
| O:E Ratio | Observed:Expected ratio (calibration) |
| IPA | Index of Prediction Accuracy (Scaled Brier) |
| pupil-gt | Ground truth (human-annotated) |
| FM | Foundation Model |
| STRATOS | STRengthening Analytical Thinking for Observational Studies |
| pminternal | R package for prediction instability (Riley 2023) |

## Anti-Patterns (What NOT to Do)

| Anti-Pattern | Why Wrong | Correct Approach |
|--------------|-----------|------------------|
| Compare classifiers | Not the research question | Fix CatBoost, vary preprocessing |
| Report only AUROC | Violates STRATOS | Report all metrics |
| Hardcode method names | Breaks validation | Use config files |
| Skip ground truth | Missing baseline | Always include pupil-gt |
| Reimplement R packages | Bug risk | Use rpy2 wrapper |
| Use grep/sed for code | Context-blind | Use AST parsing |
