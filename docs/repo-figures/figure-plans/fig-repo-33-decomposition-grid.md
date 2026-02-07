# fig-repo-33: PLR Waveform Decomposition Grid

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repo-33 |
| **Title** | PLR Waveform Decomposition by Preprocessing Method |
| **Complexity Level** | L3 (Technical deep-dive) |
| **Target Persona** | Research Scientist, Domain Expert |
| **Location** | Supplementary Materials |
| **Priority** | P2 |
| **Aspect Ratio** | 14:10 |

## Purpose

Show how different PLR waveform decomposition methods perform across preprocessing categories. This answers: "Do foundation models and deep learning preprocessing preserve the underlying PLR component structure better than traditional methods?"

## Key Message

"Different decomposition methods extract similar physiological components (phasic, sustained, PIPR) when applied to well-preprocessed data, but show varying uncertainty depending on preprocessing quality."

## Visual Concept

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                PLR WAVEFORM DECOMPOSITION BY PREPROCESSING METHOD                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  5×5 FACETED GRID                                                                │
│  ═══════════════                                                                 │
│                                                                                  │
│              │ Ground Truth │ Foundation  │ Deep       │ Traditional │ Ensemble │
│              │              │ Model       │ Learning   │             │          │
│  ────────────┼──────────────┼─────────────┼────────────┼─────────────┼──────────│
│  Template    │     [plot]   │   [plot]    │   [plot]   │   [plot]    │  [plot]  │
│  Fitting     │              │             │            │             │          │
│  ────────────┼──────────────┼─────────────┼────────────┼─────────────┼──────────│
│  Standard    │     [plot]   │   [plot]    │   [plot]   │   [plot]    │  [plot]  │
│  PCA         │              │             │            │             │          │
│  ────────────┼──────────────┼─────────────┼────────────┼─────────────┼──────────│
│  Rotated     │     [plot]   │   [plot]    │   [plot]   │   [plot]    │  [plot]  │
│  PCA         │              │             │            │             │          │
│  ────────────┼──────────────┼─────────────┼────────────┼─────────────┼──────────│
│  Sparse      │     [plot]   │   [plot]    │   [plot]   │   [plot]    │  [plot]  │
│  PCA         │              │             │            │             │          │
│  ────────────┼──────────────┼─────────────┼────────────┼─────────────┼──────────│
│  GED         │     [plot]   │   [plot]    │   [plot]   │   [plot]    │  [plot]  │
│              │              │             │            │             │          │
│                                                                                  │
│  EACH SUBPLOT CONTAINS:                                                          │
│  ───────────────────────                                                         │
│  • Mean PLR waveform (gray dashed line with CI)                                  │
│  • 3 component timecourses (colored lines with CI shading)                       │
│  • Stimulus period markers (blue 15.5-24.5s, red 46.5-55.5s)                     │
│                                                                                  │
│  LEGEND (top-right subplot):                                                     │
│  ─────────────────────────                                                       │
│  • Orange: Phasic/PC1/RC1/SPC1/GED1                                              │
│  • Blue: Sustained/PC2/RC2/SPC2/GED2                                             │
│  • Green: PIPR/PC3/RC3/SPC3/GED3                                                 │
│  • Gray: Mean waveform                                                           │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Content Elements

1. **5×5 grid layout**: 5 decomposition methods × 5 preprocessing categories = 25 subplots
2. **Row labels**: Decomposition method names (Template Fitting, PCA, Rotated PCA, Sparse PCA, GED)
3. **Column labels**: Preprocessing categories (Ground Truth, Foundation Model, Deep Learning, Traditional, Ensemble)
4. **Component traces**: 3 components per subplot with 95% bootstrap CIs
5. **Stimulus markers**: Blue and red shaded regions marking stimulus periods

## Decomposition Methods

| Method | Components | Interpretation |
|--------|------------|----------------|
| Template Fitting | phasic, sustained, pipr | Physiologically-motivated basis functions |
| Standard PCA | PC1, PC2, PC3 | Orthogonal components (variance-maximizing) |
| Rotated PCA (Promax) | RC1, RC2, RC3 | Oblique rotation for interpretability |
| Sparse PCA | SPC1, SPC2, SPC3 | Sparse loadings (feature selection) |
| GED | GED1, GED2, GED3 | Generalized Eigendecomposition (stimulus-contrast) |

## Preprocessing Categories

| Category | Methods Included | Color |
|----------|------------------|-------|
| Ground Truth | pupil-gt | Gold (#FFD700) |
| Foundation Model | MOMENT-gt-finetune, MOMENT-gt-zeroshot, UniTS-gt-finetune | Cyan (#3EBCD2) |
| Deep Learning | TimesNet-gt | Blue (#006BA2) |
| Traditional | LOF, OneClassSVM, PROPHET, SubPCA | Gray (#999999) |
| Ensemble | ensemble-LOF-..., ensembleThresholded-... | Teal (#379A8B) |

## Text Content

### Title Text
"PLR Waveform Decomposition by Preprocessing Method"

### Caption
A 5×5 grid comparing five decomposition methods (rows) across five preprocessing categories (columns). Each subplot shows the mean PLR waveform (gray dashed) and three extracted components (colored) with 95% bootstrap confidence intervals. Stimulus periods are marked (blue: 15.5-24.5s blue light, red: 46.5-55.5s red light). Ground truth preprocessing (leftmost column) serves as the reference. Similar component shapes across columns indicate that the preprocessing method preserves the underlying PLR structure; wider CIs indicate greater uncertainty from preprocessing artifacts.

### Alt Text
5×5 grid of PLR decomposition plots. Rows: Template Fitting, Standard PCA, Rotated PCA (Promax), Sparse PCA, and GED. Columns: Ground Truth, Foundation Model, Deep Learning, Traditional, and Ensemble preprocessing. Each subplot shows a gray dashed mean waveform with three colored component traces (orange, blue, green) representing the first three extracted components with shaded 95% confidence intervals. Blue and red vertical bands mark stimulus periods.

## Code References

| File | Purpose |
|------|---------|
| `src/viz/fig_decomposition_grid.py` | Main figure generation script |
| `src/decomposition/aggregation.py` | Bootstrap aggregation across subjects |
| `src/decomposition/template_fitting.py` | Template fitting method |
| `src/decomposition/pca_methods.py` | PCA, Rotated PCA, Sparse PCA |
| `src/decomposition/ged.py` | Generalized Eigendecomposition |
| `scripts/generate_decomposition_figure.sh` | Runner script |

## Data Dependencies

| Source | Description |
|--------|-------------|
| `data/private/preprocessed_signals_per_subject.db` | Extracted signals per subject/preprocessing |
| `configs/mlflow_registry/category_mapping.yaml` | Outlier method → category mapping |

## Generation

```bash
# Test with synthetic data
uv run python src/viz/fig_decomposition_grid.py --test

# Generate real figure (after extraction completes)
./scripts/generate_decomposition_figure.sh
```

## Status

- [x] Draft created
- [x] Test figure generated (synthetic data)
- [ ] Extraction complete (in progress ~19h)
- [ ] Real figure generated
- [ ] Placed in supplementary materials
