# fig-repo-12: Future Experiments & Extensions

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repo-12 |
| **Title** | What's Next? Future Experiments |
| **Complexity Level** | L2-L3 (Overview + technical) |
| **Target Persona** | All (especially Research Scientists) |
| **Location** | Root README.md, docs/ |
| **Priority** | P3 (Medium) |

## Purpose

Show what experiments and extensions are possible with this codebase, inviting contributions and future research.

## Key Message

"This codebase is a starting point. Add new models, try different embeddings, explore decomposition - the Hydra config system makes it easy."

## Visual Concept

**Roadmap/expansion diagram:**

```
┌─────────────────────────────────────────────────────────────────┐
│                    FUTURE EXPERIMENTS                           │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  CURRENT STATE (v1.0)                                   │   │
│  │  • 11 outlier methods                                   │   │
│  │  • 8 imputation methods                                 │   │
│  │  • 5 classifiers (CatBoost best)                        │   │
│  │  • 96-dim embeddings (quick exploration)                │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           │                                     │
│           ┌───────────────┼───────────────┐                    │
│           ▼               ▼               ▼                    │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐           │
│  │ NEW MODELS   │ │ EMBEDDING    │ │ DECOMPOSITION│           │
│  │              │ │ OPTIMIZATION │ │ ANALYSIS     │           │
│  │ + MIRA       │ │              │ │              │           │
│  │ + TabPFNv2.5 │ │ Test: 8, 16, │ │ Template fit │           │
│  │ + Chronos    │ │ 32, 64, 128, │ │ PCA, GED    │           │
│  │ + Moirai     │ │ 256 dims     │ │ analysis     │           │
│  └──────────────┘ └──────────────┘ └──────────────┘           │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  HOW TO ADD:                                            │   │
│  │  1. Create wrapper in src/                              │   │
│  │  2. Add to configs/mlflow_registry/                     │   │
│  │  3. Run: python run.py --multirun model=new_model       │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Content Elements

### Required Elements
1. Current state summary
2. Three extension directions
3. Specific model/method names
4. "How to add" quick reference

### Optional Elements
1. Research questions for each direction
2. Effort estimates
3. Link to issues/discussions

## Text Content

### Title Text
"Extending the Benchmark"

### Labels/Annotations
- Current: "Established baseline (v1.0)"
- New Models: "MIRA, TabPFNv2.5, Chronos, Moirai"
- Embeddings: "Systematic dim reduction study"
- Decomposition: "Template fitting, PCA, GED exploration"
- How-to: "3 steps to add new methods"

### Caption (for embedding)
The codebase supports easy extension: add new models (MIRA, Chronos), explore embedding dimensions, or investigate signal decomposition. Hydra configs make factorial experiments simple.

## Prompts for Nano Banana Pro

### Style Prompt
Roadmap/expansion diagram. Current state as a solid box, future directions as branching paths. Use growth/expansion visual metaphor. Include specific method names. Professional with a touch of forward-looking optimism.

### Content Prompt
Create an expansion roadmap:
1. TOP: Current state box (established baseline)
2. MIDDLE: Three branches - New Models, Embedding Study, Decomposition
3. BOTTOM: "How to Add" quick reference panel

Include specific model names (MIRA, TabPFNv2.5, Chronos, Moirai) and embedding dimensions to test.

### Refinement Notes
- Should invite contribution, not overwhelm
- Emphasize that the infrastructure is ready
- Include the Hydra multirun command example

## Alt Text

Roadmap showing current baseline expanding into three future directions: new models (MIRA, Chronos), embedding optimization, and decomposition analysis, with quick guide for adding methods.

## Status

- [x] Draft created
- [ ] Review passed
- [ ] Generated
- [ ] Placed in README/docs
