# fig-repo-20: Error Propagation: How Outlier Errors Cascade

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repo-20 |
| **Title** | Error Propagation: How Outlier Errors Cascade |
| **Complexity Level** | L2 (Technical concept) |
| **Target Persona** | Researcher, Biostatistician |
| **Location** | docs/user-guide/, ARCHITECTURE.md |
| **Priority** | P1 |
| **Aspect Ratio** | 16:9 |

## Purpose

Show how errors at Stage 1 (outlier detection) propagate through imputation, feature extraction, and classification—the core research question.

## Differentiation from fig-repo-02

- **fig-repo-02**: Overview of the 4 pipeline stages
- **fig-repo-20**: Focus on ERROR PROPAGATION and AUROC degradation

## Key Message

"A missed blink at Stage 1 becomes corrupted features at Stage 3 and wrong predictions at Stage 4. Ground truth preprocessing achieves 0.913 AUROC; poor preprocessing drops to 0.85."

## Visual Concept

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    ERROR PROPAGATION: HOW OUTLIER ERRORS CASCADE                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  STAGE 1              STAGE 2              STAGE 3              STAGE 4         │
│  Outlier Detection    Imputation           Feature Extraction   Classification  │
│  ════════════════     ══════════           ══════════════════   ══════════════  │
│                                                                                 │
│  ┌─────────────┐     ┌─────────────┐      ┌─────────────┐      ┌─────────────┐ │
│  │  Raw PLR    │     │  Imputed    │      │  Features   │      │  Prediction │ │
│  │  Signal     │────▶│  Signal     │─────▶│  (20 dims)  │─────▶│  Glaucoma?  │ │
│  └─────────────┘     └─────────────┘      └─────────────┘      └─────────────┘ │
│                                                                                 │
│  ──────────────────────────────────────────────────────────────────────────     │
│                                                                                 │
│  ✅ CLEAN PATH (ground truth preprocessing)                                     │
│  ───────────────────────────────────────────                                    │
│                                                                                 │
│  ┌─────────────┐     ┌─────────────┐      ┌─────────────┐      ┌─────────────┐ │
│  │ All blinks  │     │ Perfect     │      │ Correct     │      │ AUROC       │ │
│  │ detected ✓  │────▶│ reconstruction│────▶│ amplitudes  │─────▶│ 0.913 ✓    │ │
│  └─────────────┘     └─────────────┘      └─────────────┘      └─────────────┘ │
│                                                                                 │
│  ❌ CORRUPTED PATH (poor outlier detection)                                     │
│  ──────────────────────────────────────────                                     │
│                                                                                 │
│  ┌─────────────┐     ┌─────────────┐      ┌─────────────┐      ┌─────────────┐ │
│  │ Blink       │     │ Imputation  │      │ Amplitude   │      │ AUROC       │ │
│  │ MISSED! ❌  │────▶│ fits noise  │─────▶│ bins wrong  │─────▶│ 0.850 ❌    │ │
│  └─────────────┘     └─────────────┘      └─────────────┘      └─────────────┘ │
│          │                  │                    │                    │        │
│          ▼                  ▼                    ▼                    ▼        │
│     "Garbage"          "Fits to             "Feature             "Wrong        │
│                         garbage"             corruption"          diagnosis"   │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  AUROC DEGRADATION BY OUTLIER F1                                                │
│  ═══════════════════════════════                                                │
│                                                                                 │
│  Outlier F1: 1.00  │████████████████████████████████████████████│ AUROC: 0.913 │
│  Outlier F1: 0.90  │███████████████████████████████████████░░░░░│ AUROC: 0.895 │
│  Outlier F1: 0.80  │██████████████████████████████████░░░░░░░░░░│ AUROC: 0.870 │
│  Outlier F1: 0.70  │████████████████████████████░░░░░░░░░░░░░░░░│ AUROC: 0.850 │
│                                                                                 │
│  Every 10% drop in outlier detection quality → ~2% drop in AUROC                │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  THIS IS THE RESEARCH QUESTION                                                  │
│  ═════════════════════════════                                                  │
│                                                                                 │
│  "How do preprocessing choices affect downstream classification?"               │
│                                                                                 │
│  Fix classifier (CatBoost) → Vary preprocessing → Measure AUROC impact          │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Content Elements

1. **Four-stage pipeline**: Outlier → Imputation → Features → Classification
2. **Clean vs Corrupted paths**: Side-by-side showing error cascade
3. **Visual corruption markers**: ✓ vs ❌ at each stage
4. **AUROC degradation table**: F1 scores vs resulting AUROC
5. **Research question callout**: "This is what we're studying"

## Text Content

### Title Text
"Error Propagation: How Outlier Errors Cascade"

### Caption
A single missed blink artifact propagates through the pipeline: imputation fits to noise instead of real signal, feature extraction computes wrong amplitudes, and classification makes incorrect predictions. Ground truth preprocessing (all blinks detected) achieves 0.913 AUROC; poor outlier detection (F1=0.70) drops to 0.850 AUROC. This error propagation is the core research question.

## Prompts for Nano Banana Pro

### Style Prompt
Error propagation diagram with clean vs corrupted paths. Four-stage pipeline as horizontal flow. Green checkmarks for clean path, red X marks for corrupted. Signal waveform icons showing corruption at each stage. AUROC degradation bars. Research question callout at bottom. Medical research aesthetic, matte colors.

### Content Prompt
Create an error propagation diagram:

**TOP - Stage Labels**:
- Four boxes: Outlier Detection → Imputation → Features → Classification

**MIDDLE - Two Paths**:
- GREEN PATH (top): All checkmarks, "0.913 AUROC"
- RED PATH (bottom): X marks showing error at each stage, "0.850 AUROC"
- Arrows with labels: "Garbage" → "Fits garbage" → "Wrong features" → "Wrong diagnosis"

**BOTTOM LEFT - AUROC Table**:
- Four rows showing F1 vs AUROC correlation

**BOTTOM RIGHT - Research Question**:
- Highlighted box: "How do preprocessing choices affect classification?"

## Alt Text

Error propagation diagram showing four-stage pipeline. Clean path (top): all blinks detected → perfect reconstruction → correct features → 0.913 AUROC. Corrupted path (bottom): blink missed → imputation fits noise → wrong features → 0.850 AUROC. Table shows correlation between outlier F1 (1.00 to 0.70) and AUROC (0.913 to 0.850). Research question: how preprocessing affects classification.

## Status

- [x] Draft created
- [ ] Generated
- [ ] Placed in docs/user-guide/
