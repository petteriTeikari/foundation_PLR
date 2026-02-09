# fig-nb-01: The Notebook Landscape (Decision Matrix)

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-nb-01 |
| **Title** | The Notebook Landscape (Decision Matrix) |
| **Complexity Level** | L2 (Technical) |
| **Target Persona** | Data Scientist |
| **Location** | notebooks/README.md |
| **Priority** | P1 |
| **Aspect Ratio** | 16:9 |

## Purpose

Provide a side-by-side comparison of the three major computational notebook ecosystems (Jupyter, Quarto, Marimo), highlighting their trade-offs across reproducibility, git-friendliness, multi-language support, and execution model. A decision flowchart at the bottom guides readers to the right tool for their use case.

## Key Message

"Only 4.03% of Jupyter notebooks reproduce out of the box -- but the fix is choosing the right notebook tool for your constraints, not abandoning notebooks entirely."

## Literature Sources

| Source | Finding | DOI/URL |
|--------|---------|---------|
| Pimentel et al. 2019 | 4.03% of 863,878 Jupyter notebooks from GitHub could be re-executed with same results | [DOI: 10.1109/MSR.2019.00077](https://doi.org/10.1109/MSR.2019.00077) |
| Marimo documentation | Reactive DAG execution model eliminates hidden state | [marimo.io](https://marimo.io) |
| Quarto documentation | Multi-language literate programming with freeze support | [quarto.org](https://quarto.org) |

## Visual Concept

```
+---------------------------------------------------------------------------+
|               THE NOTEBOOK LANDSCAPE: WHICH TOOL WHEN?                     |
|               Choosing the right computational notebook                    |
+---------------------+---------------------+---------------------+---------+
|                     |                     |                     |         |
|      JUPYTER        |      QUARTO         |      MARIMO         |         |
|   [notebook icon]   |   [document icon]   |   [reactive icon]   |         |
|                     |                     |                     |         |
+---------------------+---------------------+---------------------+---------+
|  FEATURE MATRIX                                                           |
|  =========================================================================|
|                     |  Jupyter   |  Quarto    |  Marimo    |              |
|  -------------------|------------|------------|------------|              |
|  File format        |  .ipynb    |  .qmd      |  .py       |              |
|                     |  (JSON)    |  (markdown) |  (script)  |              |
|  Git diffs          |  Painful   |  Clean     |  Clean     |              |
|                     |  (binary)  |  (text)    |  (text)    |              |
|  Reproducibility    |  4.03%     |  High      |  High      |              |
|  rate               |            |  (freeze)  |  (DAG)     |              |
|  Multi-language     |  Kernels   |  Native    |  Python    |              |
|                     |  (1 per nb)|  (R+Py+JS) |  only      |              |
|  Publishing         |  nbconvert |  Built-in  |  Export    |              |
|                     |            |  (PDF/HTML) |  (HTML)   |              |
|  Interactivity      |  Widgets   |  Shiny/OJS |  Reactive  |              |
|                     |            |            |  (native)  |              |
|  Execution model    |  Kernel    |  Top-to-   |  DAG       |              |
|                     |  (mutable) |  bottom    |  (auto)    |              |
|  Ecosystem size     |  Massive   |  Growing   |  Emerging  |              |
|                     |  (10M+ nbs)|            |            |              |
+---------------------+---------------------+---------------------+---------+
|                                                                           |
|  DECISION FLOWCHART                                                       |
|  ====================                                                     |
|                                                                           |
|                   +-------------------+                                   |
|                   | What do you need? |                                   |
|                   +--------+----------+                                   |
|                            |                                              |
|              +-------------+-------------+                                |
|              |             |             |                                |
|        Need R+Python? Need reactive? Need ecosystem?                      |
|              |             |             |                                |
|         +----v----+   +----v----+   +----v----+                           |
|         | QUARTO  |   | MARIMO  |   | JUPYTER |                           |
|         | .qmd    |   | .py     |   | .ipynb  |                           |
|         | freeze  |   | DAG     |   | + guards|                           |
|         +---------+   +---------+   +---------+                           |
|                                      nbstripout                           |
|                                      pre-commit                           |
|                                      jupytext                             |
|                                                                           |
|  NOTE: Marimo + Quarto are complementary                                  |
|  (official Quarto extension: quarto-marimo)                               |
|                                                                           |
+---------------------------------------------------------------------------+
|  KEY STAT: Only 4.03% of 863,878 Jupyter notebooks reproduce             |
|  Source: Pimentel et al. 2019, MSR                                        |
+---------------------------------------------------------------------------+
```

## Semantic Tags Used

| Element | Semantic Tag | Description |
|---------|-------------|-------------|
| Jupyter column header | `abnormal_warning` | Warning red -- lowest reproducibility |
| Quarto column header | `primary_pathway` | Deep blue -- primary recommendation |
| Marimo column header | `melanopsin` | Gold -- emerging alternative |
| Feature matrix text | `secondary_pathway` | Neutral gray body text |
| Decision flowchart arrows | `primary_pathway` | Deep blue pathway lines |
| "4.03%" statistic | `abnormal_warning` | Warning red for emphasis |
| "complementary" note | `healthy_normal` | Green for positive synergy |

## Content Elements

1. **Three-column header**: Jupyter, Quarto, Marimo with distinct icons
2. **Feature matrix**: 7 rows comparing file format, git diffs, reproducibility rate, multi-language, publishing, interactivity, execution model
3. **Ecosystem size row**: Jupyter's massive advantage acknowledged
4. **Decision flowchart**: Three-branch tree from "What do you need?"
5. **Guard rails callout**: nbstripout, pre-commit, jupytext for Jupyter mitigation
6. **Complementary note**: Marimo + Quarto interop via official extension
7. **Key statistic strip**: 4.03% figure with Pimentel citation

## Anti-Hallucination Rules

- The 4.03% figure comes from Pimentel et al. 2019 (863,878 notebooks from GitHub). Do NOT conflate with the 2021/2023 papers which use different samples.
- Marimo is Python-only as of 2025. Do NOT claim it supports R or Julia.
- Quarto natively supports R, Python, Julia, and Observable JS. Do NOT omit Julia/OJS.
- The quarto-marimo extension is a real project. Do NOT invent features it does not have.
- Jupyter's reproducibility problem is execution order + hidden state, not the format itself. Do NOT claim .ipynb is inherently broken.

## Text Content

### Title Text
"The Notebook Landscape: Which Tool When?"

### Caption
Comparison of three computational notebook ecosystems across seven dimensions. Jupyter dominates in ecosystem size but suffers from a 4.03% reproducibility rate due to hidden state and mutable kernel execution. Quarto provides native multi-language support with top-to-bottom execution and freeze-based CI. Marimo enforces reproducibility through DAG-based reactive execution but is Python-only. The decision flowchart guides users: multi-language projects benefit from Quarto, reactive workflows from Marimo, and ecosystem-dependent projects from Jupyter with guard rails (nbstripout, pre-commit, jupytext). Marimo and Quarto are complementary via the official quarto-marimo extension.

## Prompts for Nano Banana Pro

### Style Prompt
Clean editorial three-column comparison layout on off-white background. Feature matrix with subtle alternating row shading. Decision flowchart with organic flowing arrows. Muted color accents for each tool column. Professional data visualization quality. Sans-serif typography.

### Content Prompt
Create a "Notebook Landscape Decision Matrix" infographic:

**TOP**: Three columns with tool names and icons (Jupyter orange, Quarto deep blue, Marimo gold)

**MIDDLE**: Feature comparison matrix with 7 rows -- file format, git diffs, reproducibility rate, multi-language, publishing, interactivity, execution model. Use subtle color coding for good/neutral/poor ratings.

**BOTTOM**: Decision flowchart -- three branches from central question. Each branch leads to recommended tool with key rationale. Note that Marimo+Quarto are complementary.

**FOOTER**: Key stat strip with 4.03% figure and citation.

## Alt Text

Three-column comparison infographic of Jupyter, Quarto, and Marimo notebook ecosystems. Feature matrix compares file format (JSON vs markdown vs Python script), git diff quality (painful vs clean vs clean), reproducibility rate (4.03% vs high vs high), multi-language support (kernels vs native R+Python vs Python only), publishing (nbconvert vs built-in vs export), interactivity (widgets vs Shiny/OJS vs reactive native), and execution model (mutable kernel vs top-to-bottom vs DAG). Decision flowchart recommends Quarto for multi-language, Marimo for reactive workflows, and Jupyter with guard rails for ecosystem needs. Notes that Marimo and Quarto are complementary via official extension.

## Related Figures

- **fig-repro-02a**: Why 96.8% of notebooks fail (ELI5) -- the problem this figure contextualizes
- **fig-repro-02b**: Expert version of notebook failure modes
- **fig-nb-02**: Hidden state problem (deep dive into WHY Jupyter fails)
- **fig-nb-03**: Our Quarto architecture (project-specific solution)

## Cross-References

Reader flow: **fig-repro-02a** (notebooks fail) --> **THIS FIGURE** (what are the alternatives?) --> **fig-nb-02** (why Jupyter specifically fails) --> **fig-nb-03** (our solution)

## References

1. Pimentel JF, Murta L, Braganholo V, Freire J. "A Large-Scale Study About Quality and Reproducibility of Jupyter Notebooks." MSR 2019. DOI: 10.1109/MSR.2019.00077
2. Marimo documentation. https://marimo.io
3. Quarto documentation. https://quarto.org
4. quarto-marimo extension. https://github.com/marimo-team/quarto-marimo

## Status

- [x] Draft created
- [ ] Generated
- [ ] Placed in notebooks/README.md
