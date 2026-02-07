# fig-repo-32: How to Read Risk-Coverage Curves (AURC)

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repo-32 |
| **Title** | How to Read Risk-Coverage Curves and AURC |
| **Complexity Level** | L2-L3 (Statistical visualization + methodology) |
| **Target Persona** | All |
| **Location** | docs/concepts-for-researchers.md |
| **Priority** | P1 |
| **Aspect Ratio** | 16:10 |

## Purpose

Teach readers how to interpret risk-coverage curves and AURC (Area Under Risk-Coverage curve) for selective classification—a framework where classifiers can ABSTAIN from uncertain predictions.

## Key Message

"Risk-coverage curves show the trade-off between prediction accuracy (risk) and the proportion of samples classified (coverage). AURC measures how well a model's confidence correlates with correctness. AURC ≠ AUROC: AURC evaluates uncertainty quality, AUROC evaluates discrimination."

## Literature Foundation

| Source | Key Contribution |
|--------|------------------|
| Geifman & El-Yaniv 2017 | "Selective Classification for Deep Neural Networks" - foundational framework |
| Galil et al. 2023 | "What Can We Learn From 523 ImageNet Classifiers" - comprehensive benchmarking |
| Cattelan & Silva 2023 | NeurIPS Workshop - selective classification under distribution shift |
| Rabanser 2025 | "Uncertainty-Driven Reliability" - modern synthesis |
| TorchUncertainty | Implementation reference for AURC computation |

## Visual Concept

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│               HOW TO READ RISK-COVERAGE CURVES (AURC)                           │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  WHAT IS SELECTIVE CLASSIFICATION?                                              │
│  ═════════════════════════════════                                              │
│                                                                                 │
│  A framework where classifiers can ABSTAIN (reject) uncertain predictions:      │
│                                                                                 │
│  • Traditional classifier: Must classify ALL inputs                             │
│  • Selective classifier:   Can say "I don't know" for uncertain inputs         │
│                                                                                 │
│  WHY? In high-stakes domains (medical, safety), a confident wrong answer        │
│  is worse than admitting uncertainty.                                           │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                         │   │
│  │   Traditional:  [Input] → [Classifier] → [Always: Class A or B]        │   │
│  │                                                                         │   │
│  │   Selective:    [Input] → [Classifier] → [Class A, B, or ABSTAIN]      │   │
│  │                           + Confidence     ↓                            │   │
│  │                                        If confidence < threshold        │   │
│  │                                        → "Flag for human review"        │   │
│  │                                                                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  KEY CONCEPTS                                                                   │
│  ════════════                                                                   │
│                                                                                 │
│  COVERAGE (x-axis): Proportion of samples the model chooses to classify        │
│  ──────────────────────────────────────────────────────────────────────────    │
│  • Coverage = 1.0: Model classifies ALL samples (no abstention)                 │
│  • Coverage = 0.6: Model classifies 60%, abstains on 40%                        │
│  • Lower coverage = more selective (only most confident predictions)            │
│                                                                                 │
│  RISK (y-axis): Error rate among the samples the model DID classify            │
│  ──────────────────────────────────────────────────────────────────────────    │
│  • Risk at coverage=1.0: Overall error rate (same as 1-accuracy)               │
│  • Risk at coverage=0.6: Error rate on the 60% most confident predictions      │
│  • Lower risk = fewer errors among accepted predictions                         │
│                                                                                 │
│  THE TRADE-OFF: Lower coverage → Lower risk (fewer samples, more confident)    │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  ANATOMY OF A RISK-COVERAGE CURVE                                               │
│  ════════════════════════════════                                               │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                         │   │
│  │  Risk                                                                   │   │
│  │  (Error)                                                                │   │
│  │                                                                         │   │
│  │  0.25 ┤                                                      ●──────●  │   │
│  │       │                                                   ●●●          │   │
│  │  0.20 ┤                                              ●●●●●              │   │
│  │       │                                         ●●●●●                   │   │
│  │  0.15 ┤                                   ●●●●●●       ← At full       │   │
│  │       │                              ●●●●●                coverage:     │   │
│  │  0.10 ┤                        ●●●●●●                     overall      │   │
│  │       │                  ●●●●●●                            error rate   │   │
│  │  0.05 ┤           ●●●●●●                                               │   │
│  │       │      ●●●●●          ← AURC = shaded area                       │   │
│  │  0.00 ┼──●●●───────────────────────────────────────────────────────────│   │
│  │       0.0       0.2       0.4       0.6       0.8       1.0            │   │
│  │                                                                         │   │
│  │                           Coverage                                      │   │
│  │                                                                         │   │
│  │   LOWER curve = BETTER uncertainty estimates (smaller AURC)             │   │
│  │   Higher curve = confidence doesn't correlate with correctness          │   │
│  │                                                                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  AURC: AREA UNDER RISK-COVERAGE CURVE                                           │
│  ═══════════════════════════════════                                            │
│                                                                                 │
│  AURC = ∫ Risk(c) dc  from 0 to 1                                              │
│                                                                                 │
│  • LOWER AURC = BETTER (model's confidence correlates with correctness)        │
│  • Higher AURC = WORSE (confidence doesn't distinguish correct/incorrect)       │
│                                                                                 │
│  AURC measures: "If I reject low-confidence predictions, do I reduce errors?"  │
│                                                                                 │
│  Ideal AURC = 0: Model is 100% confident on all correct predictions,           │
│                  0% confident on all incorrect predictions.                     │
│                                                                                 │
│  Random AURC ≈ overall_error × 0.5 (approximately, for random confidence)      │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  ⚠️  AURC vs AUROC: THEY MEASURE DIFFERENT THINGS!                              │
│  ════════════════════════════════════════════════                               │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                         │   │
│  │  Metric │ Measures                    │ Question Answered               │   │
│  │  ───────┼─────────────────────────────┼────────────────────────────────│   │
│  │  AUROC  │ DISCRIMINATION              │ "Can the model rank patients    │   │
│  │         │ (class separation)          │  by disease probability?"       │   │
│  │         │                             │                                 │   │
│  │  AURC   │ UNCERTAINTY QUALITY         │ "Does the model know when it's  │   │
│  │         │ (confidence calibration)    │  likely to be wrong?"           │   │
│  │                                                                         │   │
│  │  A model can have:                                                      │   │
│  │  • HIGH AUROC + HIGH AURC: Good discrimination, poor uncertainty        │   │
│  │  • HIGH AUROC + LOW AURC:  Good discrimination, good uncertainty (BEST) │   │
│  │  • LOW AUROC + LOW AURC:   Poor discrimination, good uncertainty        │   │
│  │  • LOW AUROC + HIGH AURC:  Poor discrimination, poor uncertainty (WORST)│   │
│  │                                                                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  CLINICAL EXAMPLE:                                                              │
│  • High AUROC: Model correctly ranks most glaucoma patients above controls     │
│  • Low AURC: Model is ALSO uncertain when predictions are borderline            │
│  → Clinician can trust confident predictions, flag uncertain ones for review    │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  READING THE CURVE                                                              │
│  ═════════════════                                                              │
│                                                                                 │
│  1. START AT COVERAGE = 1.0 (right side)                                        │
│     ──────────────────────────────────                                          │
│     • This is overall error rate (same as 1 - accuracy)                         │
│     • All models start at the same point if they have same accuracy             │
│                                                                                 │
│  2. TRACE LEFT (decreasing coverage)                                            │
│     ─────────────────────────────────                                           │
│     • Good model: Risk drops quickly as coverage decreases                      │
│     • Bad model: Risk stays flat or drops slowly                                │
│     • The SHAPE reveals uncertainty quality                                      │
│                                                                                 │
│  3. COMPARE CURVES AT SAME COVERAGE                                             │
│     ─────────────────────────────────                                           │
│     • At coverage = 0.8: Which model has lower risk?                            │
│     • "If both models abstain on 20%, which is more accurate?"                  │
│                                                                                 │
│  4. CHECK THE AREA (AURC)                                                       │
│     ──────────────────────                                                      │
│     • Smaller area under curve = better selective classification                │
│     • Compare AURC values numerically when curves cross                         │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  WHY SELECTIVE CLASSIFICATION MATTERS FOR CLINICAL AI                           │
│  ════════════════════════════════════════════════════                           │
│                                                                                 │
│  In clinical settings, we can SET a coverage threshold:                         │
│  (Illustrative values - actual numbers depend on specific model and dataset)    │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                         │   │
│  │  Scenario A: Full coverage (traditional)                                │   │
│  │  • Coverage = 100%, Risk = 15%                                          │   │
│  │  • Classify everyone, 15% error rate                                    │   │
│  │                                                                         │   │
│  │  Scenario B: Selective (abstain on uncertain 20%)                       │   │
│  │  • Coverage = 80%, Risk = 5%                                            │   │
│  │  • Classify 80%, only 5% error rate on those classified                │   │
│  │  • Remaining 20% flagged for specialist review                          │   │
│  │                                                                         │   │
│  │  CLINICAL BENEFIT: Dramatically reduce errors where model IS used       │   │
│  │  while routing uncertain cases to human experts                         │   │
│  │                                                                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  LARGER DATASETS BENEFIT MORE:                                                  │
│  • Small dataset: Can't afford to abstain on many samples                       │
│  • Large dataset: Abstaining on 20% still leaves many samples for automation   │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  COMMON MISTAKES                                                                │
│  ═══════════════                                                                │
│                                                                                 │
│  ❌ "AURC and AUROC measure the same thing"                                     │
│     → AUROC = discrimination, AURC = uncertainty quality (orthogonal!)         │
│                                                                                 │
│  ❌ "Lower risk at coverage=1.0 means better AURC"                              │
│     → AURC measures the SHAPE, not the endpoint alone                           │
│                                                                                 │
│  ❌ "AURC is only for neural networks"                                          │
│     → Works for ANY model with confidence scores (CatBoost, logistic, etc.)    │
│                                                                                 │
│  ❌ "Comparing AURC across datasets with different base rates"                  │
│     → Normalize by expected random AURC for fair comparison                     │
│                                                                                 │
│  ❌ "Ignoring the coverage-risk trade-off in deployment"                        │
│     → Always specify: at what coverage do you want to operate?                  │
│                                                                                 │
│  ❌ "Using prediction probability as confidence without calibration"            │
│     → Uncalibrated probabilities may not correlate with actual correctness      │
│                                                                                 │
│  ❌ "Assuming lower coverage always means better model"                         │
│     → Coverage is a deployment CHOICE, not a metric. Compare AURC or risk       │
│       at fixed coverage levels.                                                 │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Implementation in This Repository

### Python Implementation
```python
# Using TorchUncertainty
from torch_uncertainty.metrics.classification import AURC

aurc = AURC()
aurc.update(probs, labels)  # probs = model confidence, labels = ground truth
aurc_value = aurc.compute()

# Manual computation
def compute_aurc(confidence, correct, n_bins=100):
    """
    Compute AURC from confidence scores and correctness labels.

    Args:
        confidence: Array of model confidence scores
        correct: Binary array indicating correct predictions
        n_bins: Number of coverage bins

    Returns:
        aurc: Area Under Risk-Coverage curve (lower is better)
    """
    # Sort by confidence (descending)
    sorted_indices = np.argsort(-confidence)
    sorted_correct = correct[sorted_indices]

    # Compute cumulative risk at each coverage level
    cumulative_correct = np.cumsum(sorted_correct)
    cumulative_total = np.arange(1, len(sorted_correct) + 1)
    cumulative_accuracy = cumulative_correct / cumulative_total
    cumulative_risk = 1 - cumulative_accuracy

    # Coverage levels
    coverage = cumulative_total / len(sorted_correct)

    # Integrate using trapezoidal rule
    aurc = np.trapz(cumulative_risk, coverage)

    return aurc
```

### R Implementation
```r
# Compute AURC
compute_aurc <- function(confidence, correct) {
  # Sort by confidence (descending)
  sorted_idx <- order(-confidence)
  sorted_correct <- correct[sorted_idx]

  # Cumulative risk at each coverage level
  cumulative_correct <- cumsum(sorted_correct)
  cumulative_total <- seq_along(sorted_correct)
  cumulative_risk <- 1 - (cumulative_correct / cumulative_total)

  # Coverage levels
  coverage <- cumulative_total / length(sorted_correct)

  # Integrate (trapezoidal rule)
  aurc <- pracma::trapz(coverage, cumulative_risk)

  return(aurc)
}

# Plot risk-coverage curve
plot_risk_coverage <- function(confidence, correct) {
  # ... compute coverage and risk vectors ...

  ggplot(data, aes(x = coverage, y = risk)) +
    geom_line() +
    geom_area(alpha = 0.3) +  # Shaded AURC
    labs(x = "Coverage", y = "Risk (Error Rate)") +
    annotate("text", x = 0.5, y = 0.1,
             label = paste("AURC =", round(aurc, 4)))
}
```

## Content Elements

1. **Selective classification concept**: What it means to abstain
2. **Coverage and Risk definitions**: X and Y axes explained
3. **Risk-coverage curve anatomy**: Labeled diagram
4. **AURC definition**: Formula and interpretation
5. **AURC vs AUROC comparison**: Critical distinction table
6. **Reading guide**: 4 numbered steps
7. **Clinical application**: Why larger datasets benefit
8. **Common mistakes**: What NOT to conclude

## Text Content

### Title Text
"How to Read Risk-Coverage Curves and AURC"

### Caption
Risk-coverage curves (Geifman & El-Yaniv 2017) visualize the trade-off between coverage (proportion of samples classified) and risk (error rate among classified samples). AURC (Area Under Risk-Coverage curve) quantifies selective classification performance—how well confidence correlates with correctness. Lower AURC is better. Critically, AURC measures uncertainty quality, not discrimination (AUROC)—a model can have high AUROC but poor AURC if its confidence doesn't indicate correctness.

## Prompts for Nano Banana Pro

### Style Prompt
Educational diagram explaining risk-coverage curve interpretation. Curve with shaded area for AURC. Comparison table AURC vs AUROC. Clinical deployment scenario box. Clean instructional style.

### Content Prompt
Create a risk-coverage curve reading guide:

**TOP - Selective Classification Concept**:
- Traditional vs selective classifier diagram
- Abstain option for uncertain inputs

**MIDDLE - Anatomy**:
- Risk-coverage curve with labeled axes
- Coverage (x): proportion classified
- Risk (y): error rate among classified
- Shaded area = AURC

**MIDDLE - AURC vs AUROC Table**:
- Side-by-side comparison
- What each measures
- Why they're orthogonal

**BOTTOM - Reading Steps**:
- 4 numbered steps
- 1: Start at coverage=1.0
- 2: Trace left (decreasing coverage)
- 3: Compare at same coverage
- 4: Compare AURC values

## Alt Text

Educational diagram explaining risk-coverage curves and AURC. Shows selective classification concept where models can abstain from uncertain predictions. Risk-coverage curve has coverage on x-axis (proportion classified) and risk on y-axis (error rate). Lower curve = better uncertainty estimates. AURC (Area Under Risk-Coverage curve) is the shaded area—lower is better. Critical comparison table shows AUROC measures discrimination while AURC measures uncertainty quality (orthogonal metrics). Clinical example: at 80% coverage, model achieves 5% error vs 15% at full coverage, flagging 20% for specialist review. Common mistakes include confusing AURC with AUROC.

## Status

- [x] Draft created
- [ ] Generated
- [ ] Placed in docs/concepts-for-researchers.md

## References

- Geifman Y, El-Yaniv R (2017) Selective Classification for Deep Neural Networks. arXiv:1705.08500.
- Galil I, Dabbah M, El-Yaniv R (2023) What Can We Learn From 523 ImageNet Classifiers. arXiv:2302.11874.
- Cattelan LFP, Silva D (2023) On Selective Classification under Distribution Shift. NeurIPS Workshop.
- Rabanser S (2025) Uncertainty-Driven Reliability. arXiv:2508.07556.
- Goren S, Galil I, El-Yaniv R (2024) Hierarchical Selective Classification. arXiv:2405.11533.
- TorchUncertainty documentation: https://torch-uncertainty.github.io/

## Additional Considerations

### Complementary Metrics

| Metric | Definition | Use Case |
|--------|------------|----------|
| **AURC** | Area under risk-coverage | Overall selective classification quality |
| **E-AURC** | Excess AURC (above optimal) | Normalized comparison across datasets |
| **Selective Accuracy** | Accuracy at specific coverage | Deployment decision at fixed coverage |
| **Rejection Rate** | 1 - coverage | How often model abstains |

### Relationship to Calibration

Good calibration (predicted probabilities match actual probabilities) often correlates with good AURC, but they are not identical:

- Calibrated model: P(correct | predicted 80%) ≈ 80%
- Good AURC model: High confidence → correct, low confidence → incorrect

A model can be well-calibrated but have similar confidence for correct/incorrect predictions (poor AURC).

### PLR Application Context

For PLR glaucoma screening:
- High AUROC: Model ranks glaucoma patients above controls well
- Low AURC: Model is uncertain on borderline cases (flagged for specialist)
- Deployment: At 90% coverage, achieve lower error than full coverage
- Larger screening datasets benefit from selective classification (more samples to automate)
