# fig-repo-31: Foundation Model Taxonomy

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repo-31 |
| **Title** | Foundation Model Taxonomy |
| **Complexity Level** | L2 (Technical concept) |
| **Target Persona** | ML Engineer, Research Scientist |
| **Location** | docs/concepts-for-researchers.md |
| **Priority** | P1 |
| **Aspect Ratio** | 16:10 |

## Purpose

Explain the foundation models used (MOMENT, UniTS, TimesNet, SAITS) and which tasks they perform.

## Key Message

"Four foundation/deep learning models are tested: MOMENT and UniTS for outlier detection, SAITS and CSDI for imputation, TimesNet for both. Each has unique strengths."

## Visual Concept

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    FOUNDATION MODEL TAXONOMY                                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  WHAT ARE FOUNDATION MODELS?                                                    │
│  ═══════════════════════════                                                    │
│                                                                                 │
│  Large models pretrained on diverse data, then adapted to specific tasks.       │
│  For time series: trained on millions of sequences from weather, finance,       │
│  electricity, traffic—then finetuned or used zero-shot for our PLR data.        │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  MODELS BY TASK                                                                 │
│  ═══════════════                                                                │
│                                                                                 │
│  ┌────────────────────────────────────┬────────────────────────────────────┐   │
│  │      OUTLIER DETECTION             │         IMPUTATION                 │   │
│  │      ═════════════════             │         ══════════                 │   │
│  │                                    │                                    │   │
│  │  ┌──────────────────────────┐      │      ┌──────────────────────────┐ │   │
│  │  │  MOMENT                  │      │      │  SAITS                   │ │   │
│  │  │  ────────                │      │      │  ─────                   │ │   │
│  │  │  • Masked time series FM │      │      │  • Self-Attention        │ │   │
│  │  │  • Pretrained on 1B pts  │      │      │  • Joint imputation +    │ │   │
│  │  │  • Anomaly = high recon  │      │      │    classification        │ │   │
│  │  │    error                 │      │      │  • SOTA on healthcare    │ │   │
│  │  │                          │      │      │    time series           │ │   │
│  │  │  Variants:               │      │      │                          │ │   │
│  │  │  • MOMENT-gt-finetune    │      │      │  SAITS: deep learning    │ │   │
│  │  │  • MOMENT-gt-zeroshot    │      │      │  imputation champion     │ │   │
│  │  └──────────────────────────┘      │      └──────────────────────────┘ │   │
│  │                                    │                                    │   │
│  │  ┌──────────────────────────┐      │      ┌──────────────────────────┐ │   │
│  │  │  UniTS                   │      │      │  CSDI                    │ │   │
│  │  │  ─────                   │      │      │  ────                    │ │   │
│  │  │  • Universal Time Series │      │      │  • Conditional Score-    │ │   │
│  │  │  • Multi-task model      │      │      │    based Diffusion       │ │   │
│  │  │  • Unified architecture  │      │      │  • Probabilistic         │ │   │
│  │  │    for forecast/anomaly  │      │      │    imputation            │ │   │
│  │  │                          │      │      │  • Generates multiple    │ │   │
│  │  │  Variants:               │      │      │    plausible values      │ │   │
│  │  │  • UniTS-gt-finetune     │      │      │                          │ │   │
│  │  └──────────────────────────┘      │      └──────────────────────────┘ │   │
│  │                                    │                                    │   │
│  └────────────────────────────────────┴────────────────────────────────────┘   │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                          TIMESNET (Both Tasks)                          │   │
│  │                          ═════════════════════                          │   │
│  │                                                                         │   │
│  │  • 2D representation via FFT (time → frequency)                         │   │
│  │  • Captures multi-periodic patterns                                     │   │
│  │  • Works for: outlier detection AND imputation                          │   │
│  │                                                                         │   │
│  │  Variants: TimesNet-gt (outlier), TimesNet (imputation)                 │   │
│  │                                                                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  COMPARISON TABLE                                                               │
│  ════════════════                                                               │
│                                                                                 │
│  │ Model    │ Outlier │ Impute │ FM? │ Key Strength                      │     │
│  │ ──────── │ ─────── │ ────── │ ─── │ ──────────────────────────────── │     │
│  │ MOMENT   │   ✅    │   ✅   │ Yes │ Largest pretrained (1B points)    │     │
│  │ UniTS    │   ✅    │        │ Yes │ Multi-task unified architecture   │     │
│  │ TimesNet │   ✅    │   ✅   │ No  │ 2D temporal patterns via FFT      │     │
│  │ SAITS    │         │   ✅   │ No  │ Healthcare imputation SOTA        │     │
│  │ CSDI     │         │   ✅   │ No  │ Probabilistic diffusion           │     │
│                                                                                 │
│  MOMENT imputes via MOMENT-finetune (trained on PLR) / MOMENT-zeroshot          │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  BASELINES                                                                      │
│  ═════════                                                                      │
│                                                                                 │
│  Outlier: LOF, OneClassSVM, SubPCA, PROPHET (traditional ML)                    │
│  Impute: pupil-gt (human ground truth = upper bound)                            │
│                                                                                 │
│  All 5 imputation methods are deep learning. Ground truth provides the          │
│  ceiling performance that automated methods aim to approach.                    │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Content Elements

1. **FM definition**: What foundation models are
2. **Task matrix**: Outlier detection vs imputation columns
3. **Model cards**: MOMENT, UniTS, TimesNet, SAITS, CSDI
4. **Comparison table**: Which models do what
5. **Baselines**: Traditional ML (LOF, SVM, SubPCA, PROPHET) + ground truth (pupil-gt)

## Text Content

### Title Text
"Foundation Model Taxonomy: Which Model Does What"

### Caption
The pipeline tests deep learning models across two tasks: outlier detection (MOMENT, UniTS, TimesNet) and imputation (SAITS, CSDI, TimesNet, MOMENT). MOMENT and UniTS are foundation models pretrained on billions of time series points. TimesNet uses 2D temporal representations via FFT. SAITS and CSDI are specialized imputation models. Traditional outlier baselines (LOF, OneClassSVM, SubPCA, PROPHET) enable FM vs ML comparison. Human ground truth (pupil-gt) provides the ceiling performance for both tasks.

## Prompts for Nano Banana Pro

### Style Prompt
Two-column task matrix with model cards. Outlier detection on left, imputation on right. TimesNet spanning both. Each model as a card with name, key description. Comparison table at bottom. Clean, technical documentation aesthetic.

### Content Prompt
Create a foundation model taxonomy diagram:

**TOP - Definition**:
- "Foundation models: pretrained on diverse data, adapted to specific tasks"

**MIDDLE - Task Matrix**:
- Two columns: Outlier Detection | Imputation
- Cards in each column for relevant models
- TimesNet spanning both columns

**BOTTOM - Comparison Table**:
- Model | Outlier | Impute | FM? | Key Strength
- Checkmarks for supported tasks

**FOOTER**:
- Traditional outlier baselines: LOF, OneClassSVM, SubPCA, PROPHET
- Ground truth baseline: pupil-gt (human annotation)

## Alt Text

Foundation model taxonomy showing models by task. Outlier detection: MOMENT (masked FM, 1B pretrain), UniTS (multi-task unified). Imputation: SAITS (self-attention, healthcare SOTA), CSDI (diffusion-based, probabilistic), MOMENT (finetune/zeroshot). TimesNet does both tasks using 2D FFT representation. Comparison table shows which models are foundation models vs specialized deep learning. Traditional outlier baselines: LOF, OneClassSVM, SubPCA, PROPHET. Human ground truth (pupil-gt) provides ceiling performance.

## Status

- [x] Draft created
- [ ] Generated
- [ ] Placed in docs/concepts-for-researchers.md
