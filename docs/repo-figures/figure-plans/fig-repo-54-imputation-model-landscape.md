# Figure Plan: fig-repo-54-imputation-model-landscape

**Target**: Repository documentation infographic
**Section**: `configs/MODELS/` (imputation models)
**Purpose**: Visual guide to the 8 imputation methods
**Version**: 1.0

---

## Title

**Imputation Model Landscape: From Linear to Foundation Models**

---

## Purpose

Help developers understand:
1. The 8 imputation methods in the registry
2. Model architectures and their characteristics
3. Which methods are deep learning vs traditional
4. Config file organization

---

## Visual Layout (Architecture Comparison)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  FIGURE TITLE: Imputation Model Landscape                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  COMPLEXITY SPECTRUM                                                         │
│  ←────────────────────────────────────────────────────────────────────────→ │
│  Simple                                                         Complex      │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  GROUND TRUTH (1)                                                        ││
│  │  ┌───────────────────────────────────────────────────────────────┐      ││
│  │  │  1. pupil-gt                                                   │      ││
│  │  │     Human-annotated ground truth signal                        │      ││
│  │  │     [Reference standard - not an algorithm]                    │      ││
│  │  └───────────────────────────────────────────────────────────────┘      ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  TRADITIONAL (1)                                                         ││
│  │  ┌───────────────────────────────────────────────────────────────┐      ││
│  │  │  2. linear                                                     │      ││
│  │  │     Linear interpolation between valid points                  │      ││
│  │  │     [Simple baseline - no learning]                            │      ││
│  │  │                                                                │      ││
│  │  │     •────────•          •────────────•                         │      ││
│  │  │       ╲    ╱              ╲        ╱                           │      ││
│  │  │        ╲  ╱                ╲      ╱                            │      ││
│  │  │         ╲╱                  ╲    ╱                             │      ││
│  │  │     [gap]                  [gap]                               │      ││
│  │  └───────────────────────────────────────────────────────────────┘      ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  DEEP LEARNING (3)                                                       ││
│  │  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐            ││
│  │  │  3. SAITS       │ │  4. CSDI        │ │  5. TimesNet    │            ││
│  │  │                 │ │                 │ │                 │            ││
│  │  │  Self-Attention │ │  Conditional    │ │  Temporal 2D    │            ││
│  │  │  Imputation for │ │  Score-based    │ │  Variation      │            ││
│  │  │  Time Series    │ │  Diffusion      │ │  Modeling       │            ││
│  │  │                 │ │                 │ │                 │            ││
│  │  │  [Transformer]  │ │  [Diffusion]    │ │  [Conv+FFT]     │            ││
│  │  │                 │ │                 │ │                 │            ││
│  │  │  Config:        │ │  Config:        │ │  Config:        │            ││
│  │  │  SAITS.yaml     │ │  CSDI.yaml      │ │  TimesNet.yaml  │            ││
│  │  └─────────────────┘ └─────────────────┘ └─────────────────┘            ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  FOUNDATION MODELS (2)                                                   ││
│  │  ┌─────────────────────────────┐ ┌─────────────────────────────┐        ││
│  │  │  6. MOMENT-finetune         │ │  7. MOMENT-zeroshot         │        ││
│  │  │                             │ │                             │        ││
│  │  │  Foundation model fine-     │ │  Foundation model zero-     │        ││
│  │  │  tuned on PLR imputation    │ │  shot reconstruction        │        ││
│  │  │                             │ │                             │        ││
│  │  │  [Pretrained + Fine-tune]   │ │  [Pretrained only]          │        ││
│  │  │                             │ │                             │        ││
│  │  │  Config: MOMENT.yaml        │ │  Config: MOMENT.yaml        │        ││
│  │  └─────────────────────────────┘ └─────────────────────────────┘        ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  ENSEMBLE (1)                                                            ││
│  │  ┌───────────────────────────────────────────────────────────────┐      ││
│  │  │  8. ensemble-CSDI-MOMENT-SAITS-TimesNet                        │      ││
│  │  │     Weighted average of DL imputation methods                  │      ││
│  │  └───────────────────────────────────────────────────────────────┘      ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  CONFIG LOCATION: configs/MODELS/*.yaml                                  ││
│  │  REGISTRY: configs/mlflow_registry/parameters/classification.yaml       ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Content Elements

### Model Summary Table

| Category | Method | Architecture | Training | Config |
|----------|--------|--------------|----------|--------|
| Ground Truth | pupil-gt | N/A | N/A | N/A |
| Traditional | linear | Interpolation | None | N/A |
| Deep Learning | SAITS | Transformer | Supervised | SAITS.yaml |
| Deep Learning | CSDI | Diffusion | Supervised | CSDI.yaml |
| Deep Learning | TimesNet | Conv + FFT | Supervised | TimesNet.yaml |
| Foundation | MOMENT-finetune | Transformer | Pretrain + FT | MOMENT.yaml |
| Foundation | MOMENT-zeroshot | Transformer | Pretrain only | MOMENT.yaml |
| Ensemble | ensemble-CSDI-... | Weighted avg | N/A | N/A |

### Architecture Comparison

| Model | Backbone | Strengths | Limitations |
|-------|----------|-----------|-------------|
| SAITS | Self-attention | Learns dependencies | Requires training |
| CSDI | Diffusion | Probabilistic output | Slow inference |
| TimesNet | Conv + FFT | Multi-scale patterns | Fixed scales |
| MOMENT | Foundation | Zero-shot possible | May oversmooth |

---

## Key Messages

1. **8 imputation methods total**: Ground truth + linear + 3 DL + 2 FM + 1 ensemble
2. **Complexity spectrum**: Linear (simple) → CSDI (complex)
3. **Zero-shot advantage**: MOMENT-zeroshot requires no training
4. **Task alignment**: Imputation = reconstruction = pretraining objective for FMs

---

## Technical Specifications

- **Aspect ratio**: 16:10 (slightly taller)
- **Resolution**: 300 DPI
- **Background**: #FBF9F3 (Economist off-white)
- **Typography**: Sans-serif, dark grey (#333333)
- **Architecture diagrams**: Simplified schematics (not detailed)

---

## Data Source

- **Registry**: `configs/mlflow_registry/parameters/classification.yaml`
- **Config files**: `configs/MODELS/*.yaml`

---

## Related Documentation

- **README to create**: `configs/MODELS/README.md`
- **Related infographic**: fig-repo-53 (Outlier Detection Methods)

---

## References

- Du et al. 2023 "SAITS: Self-Attention-based Imputation for Time Series"
- Tashiro et al. 2021 "CSDI: Conditional Score-based Diffusion Models"
- Wu et al. 2023 "TimesNet: Temporal 2D-Variation Modeling"
- Goswami et al. 2024 "MOMENT: A Family of Open Time-series Foundation Models"

---

*Figure plan created: 2026-02-02*
*For: configs/MODELS/ documentation*
