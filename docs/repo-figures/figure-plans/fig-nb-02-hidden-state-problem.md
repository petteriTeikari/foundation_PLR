# fig-nb-02: Why 96% of Notebooks Fail (The Hidden State Problem)

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-nb-02 |
| **Title** | Why 96% of Notebooks Fail (The Hidden State Problem) |
| **Complexity Level** | L1 (ELI5) |
| **Target Persona** | PhD Student |
| **Location** | notebooks/README.md |
| **Priority** | P1 |
| **Aspect Ratio** | 16:9 |

## Purpose

Visualize the core mechanism behind Jupyter notebook irreproducibility: out-of-order execution creating hidden state that vanishes on restart. Uses a "Spot the Difference" two-panel layout to make the problem viscerally obvious, even to someone who has never thought about kernel state.

## Key Message

"The notebook that 'works' on your screen is a lie -- it depends on invisible state from cells you ran out of order, deleted, or modified after execution."

## Literature Sources

| Source | Finding | DOI/URL |
|--------|---------|---------|
| Pimentel et al. 2021 | 76.88% of notebooks have cells with non-linear execution order | [DOI: 10.1007/s10664-021-09961-9](https://doi.org/10.1007/s10664-021-09961-9) |
| Pimentel et al. 2019 | 4.03% reproducibility rate across 863,878 notebooks | [DOI: 10.1109/MSR.2019.00077](https://doi.org/10.1109/MSR.2019.00077) |

## Visual Concept

```
+---------------------------------------------------------------------------+
|          WHY 96% OF NOTEBOOKS FAIL: THE HIDDEN STATE PROBLEM              |
|          Spot the Difference                                               |
+---------------------------------------------------------------------------+
|                                                                           |
|  DURING DEVELOPMENT                    RE-RUN TOP-TO-BOTTOM               |
|  (What you see)                        (What a collaborator sees)         |
|  ================================      ================================   |
|                                                                           |
|  Execution counters:                   Execution counters:                |
|  [1] [5] [3] [7] [2]                  [1] [2] [3] [4] [5]               |
|                                                                           |
|  +---------------------------+        +---------------------------+       |
|  | [1] import pandas as pd   |        | [1] import pandas as pd   |      |
|  |     data = load("exp.csv")|        |     data = load("exp.csv")|      |
|  +---------------------------+        +---------------------------+       |
|  |                           |        |                           |       |
|  | [5] results = model(clean)|        | [2] clean = normalize(df) |      |
|  |     # "clean" exists from |        |     NameError: 'df'       |      |
|  |     # a DELETED cell!     |        |     is not defined!       |      |
|  |     Works fine here.      |        |     +---------+           |       |
|  |                           |        |     | CRASH!  |           |       |
|  +---------------------------+        |     +---------+           |       |
|  | [3] clean = normalize(df) |        +---------------------------+       |
|  |     # "df" was defined    |        |                           |       |
|  |     # in cell you deleted |        | [3] print(results.head())|       |
|  |     # but kernel remembers|        |     NameError: 'results'  |      |
|  +---------------------------+        |     is not defined!       |       |
|  | [7] print(results.head())|        +---------------------------+       |
|  |     Everything looks      |        |                           |       |
|  |     perfect!              |        | [4] fig = plot(results)   |      |
|  +---------------------------+        |     NameError: 'results'  |      |
|  | [2] fig = plot(results)   |        |     CASCADE FAILURE       |      |
|  |     Beautiful figure.     |        +---------------------------+       |
|  +---------------------------+        |                           |       |
|                                        | [5] save("output.csv")   |      |
|  STATUS: "All good!"                  |     Nothing to save.      |      |
|  (hidden dependency on                 +---------------------------+       |
|   deleted cell's variable)                                                |
|                                        STATUS: 4 of 5 cells FAIL         |
|                                        (the "working" notebook            |
|                                         was never reproducible)           |
|                                                                           |
+---------------------------------------------------------------------------+
|                                                                           |
|  THE INVISIBLE CULPRIT                                                    |
|  =====================                                                    |
|                                                                           |
|  You DELETED this cell:               But the kernel REMEMBERS:           |
|  +---------------------------+        +---------------------------+       |
|  | df = pd.read_csv(...)     |        | Memory: df = <DataFrame>  |      |
|  | # Ran as cell [4], then   |        | Still alive until you     |      |
|  | # deleted from notebook   |        | restart the kernel!       |      |
|  +---------------------------+        +---------------------------+       |
|                                                                           |
|  76.88% of notebooks have non-linear execution order                     |
|  Source: Pimentel et al. 2021                                             |
|                                                                           |
+---------------------------------------------------------------------------+
|                                                                           |
|  HOW QUARTO AND MARIMO PREVENT THIS                                       |
|  ====================================                                     |
|                                                                           |
|  QUARTO: Always renders top-to-bottom    MARIMO: DAG enforcement          |
|  No kernel state between renders.        Cells form a dependency graph.   |
|  Delete a cell = variable gone.          Out-of-order execution is        |
|  quarto render = fresh execution.        structurally impossible.         |
|                                                                           |
+---------------------------------------------------------------------------+
```

## Semantic Tags Used

| Element | Semantic Tag | Description |
|---------|-------------|-------------|
| Left panel header | `healthy_normal` | Green -- appears to work |
| Right panel header | `abnormal_warning` | Red -- reveals failures |
| Execution counters (disordered) | `melanopsin` | Gold -- draws attention to disorder |
| Execution counters (sequential) | `primary_pathway` | Blue -- correct order |
| CRASH/NameError callouts | `abnormal_warning` | Red warning boxes |
| "Invisible Culprit" section | `melanopsin` | Gold -- key insight |
| Quarto/Marimo solutions | `healthy_normal` | Green -- solutions |
| 76.88% statistic | `abnormal_warning` | Red emphasis |

## Content Elements

1. **Two-panel "Spot the Difference" layout**: Left (during development) vs Right (re-run from top)
2. **Execution counter visualization**: [1][5][3][7][2] vs [1][2][3][4][5] showing order mismatch
3. **Cell-by-cell walkthrough**: 5 cells showing which work and which crash
4. **Deleted cell ghost**: The invisible variable that persists in kernel memory
5. **NameError cascade**: One missing variable causes 4 of 5 cells to fail
6. **"Invisible Culprit" callout**: Deleted cell's variable still in kernel memory
7. **76.88% statistic**: Non-linear execution order prevalence
8. **Solution strip**: How Quarto (top-to-bottom) and Marimo (DAG) prevent this

## Anti-Hallucination Rules

- The 76.88% figure is from Pimentel et al. 2021 (EMSE journal). Do NOT attribute it to the 2019 MSR paper.
- The 4.03% reproducibility rate is from 2019 (863,878 notebooks). The 96% figure in the title rounds from this.
- Kernel state is the IPython/Jupyter kernel's in-memory Python namespace. Do NOT confuse with OS-level process state.
- Quarto does NOT use a persistent kernel between renders. Each `quarto render` starts fresh. Do NOT claim it uses kernel restart tricks.
- Marimo uses a static DAG derived from AST analysis of cell references. Do NOT claim it uses runtime dependency tracking.
- This figure focuses on EXECUTION ORDER as the hidden state mechanism. It complements fig-repro-02a/02b which focus on DEPENDENCY and IMPORT failures.

## Text Content

### Title Text
"Why 96% of Notebooks Fail: The Hidden State Problem"

### Caption
The core mechanism behind notebook irreproducibility: out-of-order cell execution creates hidden state. Left panel shows a notebook during development -- cells executed as [1][5][3][7][2] with a deleted cell's variable (df) still alive in kernel memory. Everything appears to work. Right panel shows what happens when the same notebook is re-run top-to-bottom: 4 of 5 cells crash with NameError because the hidden dependencies are gone. 76.88% of notebooks exhibit non-linear execution order (Pimentel et al. 2021). Quarto prevents this by always rendering top-to-bottom with a fresh process. Marimo prevents this by enforcing a dependency DAG that makes out-of-order execution structurally impossible.

## Prompts for Nano Banana Pro

### Style Prompt
Split-panel "Spot the Difference" infographic on off-white background. Left panel with subtle green tint (appears to work), right panel with subtle red tint (everything crashes). Terminal-style code cells with execution counters. Clean sans-serif typography. Ghost/transparency effect for the deleted cell. Warm, educational tone.

### Content Prompt
Create a "Hidden State Problem" two-panel infographic:

**LEFT PANEL - "During Development"**:
- Five code cells with out-of-order execution counters [1][5][3][7][2]
- All cells appear to run successfully
- Annotation showing a DELETED cell's variable still in memory
- Green status indicator: "All good!"

**RIGHT PANEL - "Re-run Top-to-Bottom"**:
- Same five cells with sequential counters [1][2][3][4][5]
- Cell 2 crashes with NameError (depends on deleted cell)
- Cascade: cells 3, 4, 5 also fail
- Red status indicator: "4 of 5 cells FAIL"

**BOTTOM STRIP**:
- "The Invisible Culprit": deleted cell still in kernel memory
- 76.88% statistic
- Quarto and Marimo prevention mechanisms

## Alt Text

Two-panel Spot the Difference infographic showing the Jupyter hidden state problem. Left panel: during development, five cells with execution counters [1][5][3][7][2] all run successfully because a deleted cell's variable persists in kernel memory. Right panel: when re-run top-to-bottom with counters [1][2][3][4][5], cell 2 crashes with NameError because the deleted cell's variable no longer exists, causing 4 of 5 cells to fail in cascade. Bottom section explains 76.88% of notebooks have non-linear execution order and shows how Quarto (top-to-bottom rendering) and Marimo (DAG enforcement) prevent this problem.

## Related Figures

- **fig-repro-02a**: Why 96.8% of notebooks fail (ELI5 with recipe analogy) -- focuses on dependency/import failures
- **fig-repro-02b**: Expert version of notebook failure modes
- **fig-nb-01**: Notebook landscape (the alternatives available)
- **fig-nb-03**: Our Quarto architecture (how we solved this)
- **fig-nb-04**: Quarto freeze (the CI component of the solution)

## Cross-References

Reader flow: **fig-repro-02a** (notebooks fail -- overview) --> **THIS FIGURE** (the specific hidden state mechanism) --> **fig-nb-01** (what are the alternatives?) --> **fig-nb-03** (our solution)

Complementary to fig-repro-02a/02b: those figures cover the BREADTH of failure modes (dependencies, imports, paths, randomness). This figure goes DEEP on one specific mechanism (hidden state from out-of-order execution).

## References

1. Pimentel JF, Murta L, Braganholo V, Freire J. "Understanding and Improving the Quality and Reproducibility of Jupyter Notebooks." Empirical Software Engineering 26, 65 (2021). DOI: 10.1007/s10664-021-09961-9
2. Pimentel JF, Murta L, Braganholo V, Freire J. "A Large-Scale Study About Quality and Reproducibility of Jupyter Notebooks." MSR 2019. DOI: 10.1109/MSR.2019.00077

## Status

- [x] Draft created
- [ ] Generated
- [ ] Placed in notebooks/README.md
