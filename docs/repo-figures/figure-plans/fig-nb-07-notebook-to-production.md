# fig-nb-07: From Notebook to Production (The Thin Notebook Pattern)

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-nb-07 |
| **Title** | From Notebook to Production (The Thin Notebook Pattern) |
| **Complexity Level** | L2 (Intermediate) |
| **Target Persona** | Data Scientist |
| **Location** | docs/notebooks-guide.md |
| **Priority** | P1 |
| **Aspect Ratio** | 16:10 |

## Purpose

Present four notebook-to-production patterns as a gradient from best practice to anti-pattern, showing where this project sits (thin notebook) and when alternative patterns are appropriate.

## Key Message

"Keep notebooks thin: import logic from tested src/ modules, read data from DuckDB, and let the notebook be a narrative layer, not the computation engine."

## Literature Sources

| Source | Finding | DOI/URL |
|--------|---------|---------|
| Rougier et al. 2014 | Ten Simple Rules for Better Figures, Rule #1: Know Your Audience | [PLOS Comp Biol](https://doi.org/10.1371/journal.pcbi.1003833) |
| Quaranta et al. 2022 | Smooth transition from explorative to production phase | [IEEE TSE](https://doi.org/10.1109/TSE.2021.3135585) |
| Ploomber | Modular notebook pipelines | [ploomber.io](https://ploomber.io/) |
| Databricks | Notebook-as-orchestrator pattern | [docs.databricks.com](https://docs.databricks.com/) |
| Netflix | Notebook infrastructure at scale | [netflixtechblog.com](https://netflixtechblog.com/) |

## Visual Concept

```
+---------------------------------------------------------------------------------+
|                    FROM NOTEBOOK TO PRODUCTION                                   |
|                    Four Patterns: Good to Bad                                    |
+---------------------------------------------------------------------------------+
|                                                                                 |
|  PATTERN GRADIENT                                                               |
|  ================                                                               |
|                                                                                 |
|  RECOMMENDED                                                ANTI-PATTERN        |
|  <-------------------------------------------------------------->               |
|  Green                          Gold                          Red               |
|                                                                                 |
|  +-------------------+  +-------------------+  +-------------------+            |
|  | 1. THIN NOTEBOOK  |  | 2. nbdev LITERATE |  | 3. NB-AS-ORCH.   |            |
|  | (RECOMMENDED)     |  | (FOR LIBRARIES)   |  | (PLATFORM-TIED)   |            |
|  |                   |  |                   |  |                   |            |
|  | notebook.ipynb    |  | notebook.ipynb    |  | master.ipynb      |            |
|  |   |               |  |   |               |  |   |               |            |
|  |   | import        |  |   | #|export      |  |   | invokes       |            |
|  |   v               |  |   v               |  |   v               |            |
|  | src/module.py     |  | lib/module.py     |  | worker_1.ipynb    |            |
|  |   |               |  |   |               |  | worker_2.ipynb    |            |
|  |   | tested by     |  |   | + docs site   |  | worker_3.ipynb    |            |
|  |   v               |  |   | + tests       |  |   |               |            |
|  | pytest (2042)     |  |   v               |  |   v               |            |
|  |                   |  | pip installable   |  | DAG execution     |            |
|  | PRODUCTION-READY  |  | LIBRARY-READY     |  | PLATFORM-READY    |            |
|  +-------------------+  +-------------------+  +-------------------+            |
|                                                                                 |
|  +----------------------------------------------------------------------+       |
|  | 4. FAT NOTEBOOK (ANTI-PATTERN)                                        |       |
|  |                                                                       |       |
|  | notebook.ipynb (800 lines, 60 cells)                                  |       |
|  |   - All logic inline                                                  |       |
|  |   - No imports from external modules                                  |       |
|  |   - Hardcoded paths and constants                                     |       |
|  |   - Not importable by other code                                      |       |
|  |   - No tests                                                          |       |
|  |   - "Works on my machine"                                             |       |
|  |                                                                       |       |
|  | RESULT: Cannot test, cannot reuse, cannot reproduce                    |       |
|  +----------------------------------------------------------------------+       |
|                                                                                 |
+---------------------------------------------------------------------------------+
|  OUR IMPLEMENTATION: THIN NOTEBOOK                                              |
|  =====================================                                          |
|                                                                                 |
|  | Metric              | Our Notebooks     | Fat Notebook (typical) |           |
|  |---------------------|-------------------|------------------------|           |
|  | Cells per notebook  | 10-21             | 50-100+                |           |
|  | Lines per cell      | <15               | 30-100+                |           |
|  | Data source         | DuckDB read-only  | Inline pd.read_csv()   |           |
|  | Colors              | COLORS dict       | Hardcoded hex          |           |
|  | Logic location      | src/ modules      | Inline                 |           |
|  | Test coverage       | pytest on src/    | None                   |           |
|                                                                                 |
+---------------------------------------------------------------------------------+
|  DECISION MATRIX                                                                |
|  ===============                                                                |
|                                                                                 |
|  Building a reusable LIBRARY?  --> nbdev (Python) or fusen (R)                  |
|  Building an APPLICATION?      --> Thin Notebook + tested src/                   |
|  Exploring data interactively? --> Start fat, REFACTOR to thin before commit     |
|                                                                                 |
|  Rougier Rule #1: "Know Your Audience"                                          |
|    Notebooks = for researchers (narrative)                                       |
|    src/ = for developers (logic)                                                 |
|                                                                                 |
|  Quaranta et al. 2022: "smooth transition from explorative to production"       |
|                                                                                 |
+---------------------------------------------------------------------------------+
```

## Semantic Tags Used

| Element | Semantic Tag | Description |
|---------|-------------|-------------|
| Thin notebook (pattern 1) | `healthy_normal` | Green - recommended |
| nbdev literate (pattern 2) | `highlight_accent` | Gold - good for libraries |
| Notebook-as-orchestrator (3) | `highlight_accent` | Gold - platform-dependent |
| Fat notebook (pattern 4) | `abnormal_warning` | Red - anti-pattern |
| Pattern gradient arrow | `healthy_normal` to `abnormal_warning` | Green-to-red gradient |
| Implementation metrics | `primary_pathway` | Deep blue emphasis |
| Decision matrix | `primary_pathway` | Deep blue structure |

## Content Elements

1. **Pattern gradient**: Four patterns arranged from recommended to anti-pattern
2. **Pattern flow diagrams**: Each pattern showing notebook-to-output data flow
3. **Fat notebook callout**: Highlighted anti-pattern with failure modes
4. **Implementation comparison table**: Our notebooks vs typical fat notebooks
5. **Decision matrix**: When to use each pattern
6. **Rougier quote**: "Know Your Audience" applied to notebook architecture
7. **Quaranta quote**: "Smooth transition from explorative to production phase"

## Anti-Hallucination Rules

- DO NOT invent notebook cell counts. Our notebooks have 10-21 cells with under 15 lines each (verified from actual .ipynb files).
- DO NOT claim nbdev is deprecated or unmaintained. nbdev v2 switched to Quarto in 2022 and is actively developed by fast.ai.
- DO NOT show AUROC or model performance. This is about code architecture patterns.
- The 2042 test count is from actual pytest output.
- Databricks notebook-as-orchestrator and Netflix notebook infrastructure are real engineering patterns, not invented examples.
- Ploomber is a real tool for modular notebook pipelines, not a hypothetical.

## Text Content

### Title Text
"From Notebook to Production: The Thin Notebook Pattern"

### Caption
Four notebook-to-production patterns on a spectrum from best practice to anti-pattern. The Thin Notebook (recommended): notebooks import from tested src/ modules, read DuckDB data, contain 10-21 cells with under 15 lines each. nbdev Literate: notebooks export to installable libraries (good for package development). Notebook-as-Orchestrator: master notebooks invoke worker notebooks in a DAG (Databricks/Netflix pattern). Fat Notebook (anti-pattern): all logic inline, no tests, not reproducible. Foundation PLR uses the Thin Notebook pattern with 2042 pytest tests covering src/ modules.

## Prompts for Nano Banana Pro

### Style Prompt
Four-pattern comparison with green-to-red gradient indicating quality. Each pattern shown as a mini architecture diagram. Comparison table below. Decision matrix at bottom. Professional editorial layout on warm off-white background. Clean flow arrows within each pattern.

### Content Prompt
Create "Notebook to Production" infographic:

**TOP - Pattern Gradient**:
- Four boxes left to right: Thin Notebook (green), nbdev (gold), Orchestrator (gold), Fat (red)
- Each with mini flow diagram showing data/code paths
- Arrow from "RECOMMENDED" to "ANTI-PATTERN"

**MIDDLE - Implementation Comparison**:
- Table: Our thin notebooks vs fat notebooks on 6 metrics
- Cells, lines, data source, colors, logic location, tests

**BOTTOM - Decision Matrix**:
- Building library -> nbdev/fusen
- Building application -> thin notebook
- Exploring data -> start fat, refactor to thin

## Alt Text

Four notebook-to-production patterns arranged from recommended to anti-pattern. Pattern 1 Thin Notebook (green, recommended): notebooks import from src/ modules, tested by 2042 pytest tests. Pattern 2 nbdev Literate (gold): notebooks export to installable library with docs and tests. Pattern 3 Notebook-as-Orchestrator (gold): master notebook invokes workers in DAG execution. Pattern 4 Fat Notebook (red, anti-pattern): all logic inline, no tests, not reproducible. Comparison table shows Foundation PLR notebooks: 10-21 cells, under 15 lines each, DuckDB read-only, COLORS dict, logic in src/. Decision matrix: libraries use nbdev/fusen, applications use thin notebook.

## References

- Rougier et al. 2014: Ten Simple Rules for Better Figures ([PLOS Comp Biol](https://doi.org/10.1371/journal.pcbi.1003833))
- Quaranta et al. 2022: Notebook quality in practice ([IEEE TSE](https://doi.org/10.1109/TSE.2021.3135585))
- Ploomber: Modular notebook pipelines ([ploomber.io](https://ploomber.io/))
- Databricks: Notebook best practices ([docs.databricks.com](https://docs.databricks.com/))
- Netflix Tech Blog: Notebook infrastructure ([netflixtechblog.com](https://netflixtechblog.com/))
- nbdev: fast.ai notebook development ([nbdev.fast.ai](https://nbdev.fast.ai/))

## Related Figures

- **fig-nb-05**: Notebook Testing Landscape (how we test the thin pattern)
- **fig-nb-08**: nbdev and fusen deep dive (pattern 2 details)
- **fig-repo-18**: Two-Block Architecture (the src/ architecture notebooks import from)

## Cross-References

Reader flow: **fig-repro-02a/02b** (why notebooks fail) -> **THIS FIGURE** (how to structure them) -> **fig-nb-08** (library-building variant) -> **fig-nb-05** (how to test them)

## Status

- [x] Draft created
- [ ] Generated
- [ ] Placed in docs/notebooks-guide.md
