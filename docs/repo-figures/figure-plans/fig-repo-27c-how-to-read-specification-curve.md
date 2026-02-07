# fig-repo-29: How to Read Specification Curves

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repo-29 |
| **Title** | How to Read Specification Curves |
| **Complexity Level** | L2 (Statistical visualization) |
| **Target Persona** | All |
| **Location** | docs/concepts-for-researchers.md |
| **Priority** | P1 |
| **Aspect Ratio** | 16:10 |

## Purpose

Teach readers how to interpret specification curve analysis—a method for assessing robustness of findings across all reasonable analytical choices.

## Key Message

"Specification curves show results for ALL reasonable specifications, not just the 'best' one. If the effect holds across most specifications (sorted by magnitude), the finding is robust. The dashboard chart below shows WHICH choices led to which results."

## Literature Foundation

| Source | Key Contribution |
|--------|------------------|
| Simonsohn et al. 2020 | Nature Human Behaviour - introduced specification curve analysis |
| Steegen et al. 2016 | Multiverse analysis concept |

## Visual Concept

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      HOW TO READ SPECIFICATION CURVES                           │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  WHAT IS A SPECIFICATION CURVE?                                                 │
│  ══════════════════════════════                                                 │
│                                                                                 │
│  A visualization showing results for ALL reasonable analytical specifications   │
│  Instead of: "We chose method X and got result Y"                               │
│  Shows: "Here are results for ALL valid choices—our finding is/isn't robust"    │
│                                                                                 │
│  THREE STEPS (Simonsohn et al. 2020):                                           │
│  1. Identify ALL reasonable specifications                                      │
│  2. Display results graphically (sorted by magnitude)                           │
│  3. Perform joint statistical inference                                         │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  ANATOMY OF A SPECIFICATION CURVE                                               │
│  ════════════════════════════════                                               │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                         │   │
│  │  AUROC                                                                  │   │
│  │  0.92 ┤                                    ●●●●●●●  ← Best specs       │   │
│  │       │                            ●●●●●●●●●                            │   │
│  │  0.90 ┤                     ●●●●●●●                                     │   │
│  │       │              ●●●●●●●                  ↑                         │   │
│  │  0.88 ┤       ●●●●●●●                   Ground truth                    │   │
│  │       │   ●●●●   ↑                      reference line                  │   │
│  │  0.86 ┤ ●●    Error bars                                                │   │
│  │       │ │      (95% CI)                                                 │   │
│  │  0.84 ┼─┴───────────────────────────────────────────────────────────────│   │
│  │       1         100        200        300       328                     │   │
│  │                     Specification (sorted by AUROC)                     │   │
│  │                                                                         │   │
│  │  ═══════════════════════════════════════════════════════════════════   │   │
│  │                                                                         │   │
│  │  DASHBOARD CHART (which choices led to which results)                   │   │
│  │  ────────────────────────────────────────────────────────────────────   │   │
│  │                                                                         │   │
│  │  Outlier:  pupil-gt ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░████████████████████  │   │
│  │            MOMENT   ██████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │   │
│  │            LOF      ██████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │   │
│  │                                                                         │   │
│  │  Impute:   pupil-gt ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░███████████████████  │   │
│  │            SAITS    █████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │   │
│  │            linear   ███░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │   │
│  │                                                                         │   │
│  │            █ = used in this specification                               │   │
│  │                                                                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  READING THE CURVE                                                              │
│  ═════════════════                                                              │
│                                                                                 │
│  1. OVERALL PATTERN                                                             │
│     ───────────────                                                             │
│     • Are most points above a meaningful threshold?                             │
│     • Is the curve steep (high variability) or flat (robust)?                   │
│     • Where does the curve cross key reference lines?                           │
│                                                                                 │
│  2. EXTREME SPECIFICATIONS                                                      │
│     ──────────────────────                                                      │
│     • Left side = worst performing specifications                               │
│     • Right side = best performing specifications                               │
│     • Check dashboard: what's different about extremes?                         │
│                                                                                 │
│  3. REFERENCE LINE (Ground Truth)                                               │
│     ─────────────────────────────                                               │
│     • Horizontal line shows best-case baseline                                  │
│     • How many specs approach or exceed this line?                              │
│     • Ground truth doesn't guarantee best performance!                          │
│                                                                                 │
│  4. DASHBOARD PATTERNS                                                          │
│     ────────────────────                                                        │
│     • Which choices cluster on the right (high performance)?                    │
│     • Which choices cluster on the left (low performance)?                      │
│     • This reveals WHICH decisions matter most                                  │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  EXAMPLE: PLR Preprocessing Pipeline                                            │
│  ═══════════════════════════════════                                            │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                         │   │
│  │  Configuration: 88 total specifications                                 │   │
│  │  • 11 outlier detection methods                                         │   │
│  │  • 8 imputation methods                                                 │   │
│  │  • CatBoost classifier (fixed)                                          │   │
│  │  • 11 × 8 = 88 unique preprocessing pipelines                           │   │
│  │                                                                         │   │
│  │  What to look for in the curve:                                         │   │
│  │  ──────────────────────────────                                         │   │
│  │  (Actual findings presented in manuscript)                              │   │
│  │                                                                         │   │
│  │  • Does AUROC vary widely or remain consistent?                         │   │
│  │  • Do certain methods cluster on the right (high performance)?          │   │
│  │  • Do certain methods cluster on the left (low performance)?            │   │
│  │  • Which choice dimension (outlier vs imputation) shows larger spread?  │   │
│  │                                                                         │   │
│  │  Key question:                                                          │   │
│  │  → "Which preprocessing choices matter most for downstream AUROC?"      │   │
│  │                                                                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  JOINT INFERENCE                                                                │
│  ═══════════════                                                                │
│                                                                                 │
│  Beyond visualization, specification curves enable statistical testing:        │
│                                                                                 │
│  • NULL HYPOTHESIS: All specifications are uninformative                        │
│    (ranking by effect size follows random expected distribution)                │
│                                                                                 │
│  • PERMUTATION TEST: Compares observed curve to null via resampling            │
│    (accounts for non-independence between specifications sharing data)          │
│                                                                                 │
│  • ROBUST FINDING: Observed curve significantly differs from null              │
│    (e.g., majority of specs show effect in same direction)                      │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  COMMON MISTAKES                                                                │
│  ═══════════════                                                                │
│                                                                                 │
│  ❌ "The best specification is the right answer"                                │
│     → ALL reasonable specifications are equally valid a priori                  │
│                                                                                 │
│  ❌ "Cherry-picking only favorable specifications"                              │
│     → Must include ALL reasonable choices, even if some give poor results       │
│                                                                                 │
│  ❌ "Ignoring the dashboard chart"                                              │
│     → Dashboard reveals WHICH choices drive performance differences             │
│                                                                                 │
│  ❌ "Only showing median or mean result"                                        │
│     → The RANGE across specifications is the key insight                        │
│                                                                                 │
│  ❌ "Treating steep curve as a problem"                                         │
│     → Steep curve = specifications matter! This is informative, not bad.        │
│                                                                                 │
│  ❌ "Excluding specifications after seeing results"                             │
│     → Criteria for 'reasonable' must be defined BEFORE analysis                 │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Implementation in This Repository

### R Code Location
`src/r/figures/fig_specification_curve.R`

### Key Implementation Details
```r
# Order specifications by outcome (AUROC)
data <- data %>%
  arrange(auroc_mean) %>%
  mutate(spec_rank = row_number())

# Main curve: points with error bars
ggplot(data, aes(x = spec_rank, y = auroc_mean)) +
  geom_errorbar(aes(ymin = auroc_lo, ymax = auroc_hi),
                width = 0, alpha = 0.3) +
  geom_point(aes(color = uses_ground_truth), size = 1) +

  # Reference line for ground truth performance
  geom_hline(yintercept = gt_auroc, linetype = "dashed") +

  # Annotations
  annotate("text", x = n_specs, y = gt_auroc,
           label = "Ground Truth", hjust = 1)

# Dashboard chart below (separate panel)
# Shows which methods were used in each specification
# Implemented as faceted binary indicator plot
```

### Specification Count
- 88 total configurations (11 × 8)
- 11 outlier detection methods × 8 imputation methods
- CatBoost classifier (fixed)

## Content Elements

1. **Specification curve anatomy**: Main curve + dashboard chart
2. **Reading guide**: 4 numbered steps (pattern, extremes, reference, dashboard)
3. **Example interpretation**: PLR preprocessing findings
4. **Joint inference**: Statistical testing explanation
5. **Common mistakes**: What NOT to conclude

## Text Content

### Title Text
"How to Read Specification Curves"

### Caption
Specification curve analysis (Simonsohn et al. 2020) displays results for ALL reasonable analytical specifications, sorted by effect magnitude. The main curve shows each specification's result (here, AUROC with 95% CI). The dashboard chart below indicates which analytical choices were used in each specification. This visualization reveals whether findings are robust across specifications or depend on specific choices. A horizontal reference line shows the ground truth baseline.

## Prompts for Nano Banana Pro

### Style Prompt
Educational diagram explaining specification curve interpretation. Two-panel figure: main curve with sorted results, dashboard chart showing method indicators. Reference line for baseline. Nature Human Behaviour style per Simonsohn 2020.

### Content Prompt
Create a specification curve reading guide:

**TOP - Anatomy**:
- Main curve: sorted points with error bars
- Reference line for ground truth
- Dashboard chart below showing method indicators

**MIDDLE - Reading Steps**:
- 4 numbered steps with icons
- 1: Overall pattern (steep vs flat)
- 2: Extreme specifications
- 3: Reference line comparison
- 4: Dashboard patterns

**BOTTOM - Example**:
- 88 PLR preprocessing specifications
- Interpretation: outlier detection matters more than imputation

**SIDEBAR - Mistakes**:
- 5 common misinterpretations with X marks

## Alt Text

Educational diagram explaining specification curve analysis interpretation. Shows anatomy with main curve (specifications sorted by AUROC magnitude, points with 95% CI error bars) and dashboard chart below indicating which methods were used in each specification. Reading guide: (1) examine overall pattern, (2) check extreme specifications, (3) compare to reference line, (4) find dashboard patterns. Example shows 88 PLR preprocessing configurations revealing that outlier detection choice matters more than imputation choice. Common mistakes include cherry-picking best specification or ignoring the dashboard.

## Status

- [x] Draft created
- [ ] Generated
- [ ] Placed in docs/concepts-for-researchers.md

## References

- Simonsohn U, Simmons JP, Nelson LD (2020) Specification curve analysis. Nature Human Behaviour 4:1208-1214.
- Steegen S, Tuerlinckx F, Gelman A, Vanpaemel W (2016) Increasing transparency through a multiverse analysis. Perspectives on Psychological Science 11:702-712.
