# fig-repo-28: How to Read Raincloud Plots

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repo-28 |
| **Title** | How to Read Raincloud Plots |
| **Complexity Level** | L2 (Statistical visualization) |
| **Target Persona** | All |
| **Location** | docs/concepts-for-researchers.md |
| **Priority** | P1 |
| **Aspect Ratio** | 16:10 |

## Purpose

Teach readers how to interpret raincloud plots—a modern visualization that combines raw data, summary statistics, and distribution shape in a single figure.

## Key Message

"Raincloud plots show THREE things at once: raw data points (the 'rain'), box plot summaries (medians, quartiles), and distribution shape (the 'cloud'). This prevents hiding outliers or distributional quirks behind summary statistics."

## Literature Foundation

| Source | Key Contribution |
|--------|------------------|
| Allen et al. 2019 | Wellcome Open Research - introduced raincloud plots |
| ggdist R package | Implementation via `stat_halfeye()`, `stat_dots()` |

## Visual Concept

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        HOW TO READ RAINCLOUD PLOTS                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  WHAT IS A RAINCLOUD PLOT?                                                      │
│  ═════════════════════════                                                      │
│                                                                                 │
│  A visualization combining THREE representations of the same data:              │
│  1. Raw data points ("rain")                                                    │
│  2. Box plot (median, quartiles)                                                │
│  3. Split-half violin/density ("cloud")                                         │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                         │   │
│  │         ╭─────────────────╮                                             │   │
│  │        ╱                   ╲     ← "CLOUD": Half-violin density         │   │
│  │       ╱                     ╲      Shows distribution shape             │   │
│  │      │                       │     (skewness, multimodality)            │   │
│  │      │                       │                                          │   │
│  │   ┌──┼───────────────────────┼──┐                                       │   │
│  │   │  │         ┃             │  │  ← BOX PLOT                           │   │
│  │   └──┼─────────┃─────────────┼──┘    • Vertical line = median           │   │
│  │      │         ┃             │       • Box edges = Q1, Q3 (IQR)         │   │
│  │      ●   ●  ●  ●  ●  ● ●  ●  ●       • Whiskers = 1.5×IQR              │   │
│  │      ●  ●   ●   ●   ●    ●   ●  ← "RAIN": Raw data points              │   │
│  │       ●   ●  ● ●  ●  ●  ●           Shows actual values                 │   │
│  │                                      Reveals outliers, clusters         │   │
│  │      ├────────────────────────┤                                         │   │
│  │      0.75    0.80    0.85   0.90    (e.g., AUROC values)                │   │
│  │                                                                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  READING THE PLOT                                                               │
│  ════════════════                                                               │
│                                                                                 │
│  1. START WITH THE CLOUD (density)                                              │
│     ─────────────────────────────                                               │
│     • Where is the "bulk" of the data?                                          │
│     • Is it symmetric or skewed?                                                │
│     • Are there multiple peaks (multimodal)?                                    │
│                                                                                 │
│  2. CHECK THE BOX PLOT (summary)                                                │
│     ───────────────────────────                                                 │
│     • Median line: typical value                                                │
│     • Box width: spread of middle 50%                                           │
│     • Wide box = high variability                                               │
│                                                                                 │
│  3. EXAMINE THE RAIN (raw data)                                                 │
│     ──────────────────────────                                                  │
│     • Look for outliers beyond the cloud                                        │
│     • Check for gaps or clusters                                                │
│     • Count approximate sample size                                             │
│                                                                                 │
│  4. COMPARE GROUPS                                                              │
│     ─────────────                                                               │
│     • Overlap of clouds = similar distributions                                 │
│     • Separated clouds = distinct groups                                        │
│     • Match cloud shape AND summary statistics                                  │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  EXAMPLE: AUROC by Pipeline Type                                                │
│  ═══════════════════════════════                                                │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                         │   │
│  │  Ground Truth    ╭───────╮                                              │   │
│  │                 ╱         ●●●●●●●●  → Tight cluster, high AUROC         │   │
│  │               ─┃──────────┃─                                            │   │
│  │                                                                         │   │
│  │  FM Pipeline   ╭──────────────╮                                         │   │
│  │               ╱     ●●●●●●●●●●●●●●●  → Wider spread, some overlap       │   │
│  │             ─┃────────────┃───                                          │   │
│  │                                                                         │   │
│  │  Traditional    ╭─────────────────╮                                     │   │
│  │                ╱   ●●●●●●●●●●●●●●●●●●  → Widest spread, lower median    │   │
│  │              ─┃──────────────┃────                                      │   │
│  │                                                                         │   │
│  │              0.75    0.80    0.85   0.90    AUROC                       │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  Interpretation:                                                                │
│  • Ground truth has tightest distribution (most consistent)                     │
│  • FM pipelines show intermediate performance with moderate variability         │
│  • Traditional methods have widest spread (more variable results)               │
│  • Some FM configs overlap with ground truth (competitive performance)          │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  WHY RAINCLOUD OVER BOX PLOT ALONE?                                             │
│  ═══════════════════════════════════                                            │
│                                                                                 │
│  Box plots can HIDE important features:                                         │
│                                                                                 │
│  ┌────────────────────────────────────────────────────────────────────────┐    │
│  │                                                                        │    │
│  │   SAME BOX PLOT,              Box Plot:  ─┃────────────┃─              │    │
│  │   DIFFERENT DISTRIBUTIONS                                              │    │
│  │                                                                        │    │
│  │   Scenario A: Uniform         ●●●●●●●●●●●●●●●●●●●●                     │    │
│  │                               (evenly spread)                          │    │
│  │                                                                        │    │
│  │   Scenario B: Bimodal         ●●●●●          ●●●●●                     │    │
│  │                               (two clusters!)                          │    │
│  │                                                                        │    │
│  │   Scenario C: Skewed          ●●●●●●●●●●●●●       ●   ●                │    │
│  │                               (outliers on right)                      │    │
│  │                                                                        │    │
│  └────────────────────────────────────────────────────────────────────────┘    │
│                                                                                 │
│  The cloud and rain REVEAL what the box hides.                                  │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  COMMON MISTAKES                                                                │
│  ═══════════════                                                                │
│                                                                                 │
│  ❌ "The clouds overlap, so the methods are the same"                           │
│     → Check median positions and statistical tests for significance             │
│                                                                                 │
│  ❌ "Ignoring the rain and only reading the cloud"                              │
│     → Raw points reveal sample size and outliers                                │
│                                                                                 │
│  ❌ "Comparing cloud heights as if they show frequency"                         │
│     → Cloud is a DENSITY (area = 1), height shows relative concentration        │
│                                                                                 │
│  ❌ "Assuming symmetric cloud means normal distribution"                        │
│     → Look at the raw points to verify                                          │
│                                                                                 │
│  ❌ "Comparing cloud smoothness across groups with different N"                 │
│     → Smaller samples produce rougher density estimates; compare with caution   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Implementation in This Repository

### R Code Location
`src/r/figures/fig_raincloud_auroc.R`

### Key Implementation Details
```r
# Uses ggdist package for raincloud components
library(ggdist)

# The "cloud" (half-violin density)
stat_halfeye(
  adjust = 1,             # Bandwidth adjustment
  width = 0.6,            # Cloud width
  .width = 0,             # No interval
  justification = -0.2,   # Offset from center
  point_colour = NA,      # Hide default points
  alpha = 0.6             # Transparency
)

# The box plot
geom_boxplot(
  width = 0.15,
  outlier.shape = NA      # Hide outliers (shown in rain)
)

# The "rain" (raw data points)
stat_dots(
  side = "left",
  justification = 1.1,
  binwidth = 0.003,       # Point spacing
  alpha = 0.5             # Transparency
)

# Note: coord_flip() is used to display horizontally
# Falls back to violin+boxplot if ggdist is not installed
```

### Grouping by Pipeline Type
The repository categorizes preprocessing configs into:
- **Ground Truth**: pupil-gt for both outlier and imputation
- **FM Pipeline**: Foundation model for either stage
- **Traditional**: Non-FM methods for both stages

## Content Elements

1. **Raincloud anatomy**: Labeled diagram showing cloud, box, and rain components
2. **Reading guide**: 4 numbered steps to interpret the plot
3. **Example interpretation**: AUROC by pipeline type
4. **Why raincloud**: Comparison showing what box plots hide
5. **Common mistakes**: What NOT to conclude

## Text Content

### Title Text
"How to Read Raincloud Plots"

### Caption
Raincloud plots (Allen et al. 2019) combine three visualization techniques: raw data points ("rain") showing individual values and outliers, box plots showing median and interquartile range, and half-violin densities ("cloud") showing distribution shape. This triple representation prevents the information loss that occurs when using only summary statistics. When comparing groups, examine cloud overlap, median positions, and spread of raw points.

## Prompts for Nano Banana Pro

### Style Prompt
Educational diagram explaining raincloud plot interpretation. Annotated example with callouts for cloud, box, and rain components. Comparison panel showing what box plots hide. Clean, instructional aesthetic. Allen 2019 Wellcome Open Research style.

### Content Prompt
Create a raincloud plot reading guide:

**TOP - Anatomy**:
- Raincloud with labeled parts: half-violin cloud, box plot, raw data rain
- Callouts explaining each component

**MIDDLE - Reading Steps**:
- 4 numbered steps with icons
- 1: Read the cloud (distribution shape)
- 2: Check the box (summary stats)
- 3: Examine the rain (raw data)
- 4: Compare groups

**BOTTOM - Example**:
- AUROC comparison across pipeline types
- Ground Truth, FM Pipeline, Traditional

**SIDEBAR - Why Raincloud**:
- Same box plot, different distributions example
- Shows bimodal vs uniform vs skewed

## Alt Text

Educational diagram explaining raincloud plot interpretation. Shows anatomy with three components: half-violin "cloud" showing distribution shape, box plot showing median and quartiles, and "rain" of raw data points. Reading guide: (1) examine cloud shape, (2) check box plot summaries, (3) look at raw data points, (4) compare groups. Example shows AUROC by pipeline type—ground truth has tight distribution, FM pipelines intermediate, traditional widest spread. Panel demonstrates why rainclouds beat box plots by showing same box with uniform, bimodal, and skewed underlying data.

## Status

- [x] Draft created
- [ ] Generated
- [ ] Placed in docs/concepts-for-researchers.md

## References

- Allen M, Poggiali D, Whitaker K, et al. (2019) Raincloud plots: a multi-platform tool for robust data visualization. Wellcome Open Research 4:63.
- ggdist R package: https://mjskay.github.io/ggdist/
