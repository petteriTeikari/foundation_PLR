# fig-nb-08: Building Libraries from Notebooks (nbdev and fusen)

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-nb-08 |
| **Title** | Building Libraries from Notebooks (nbdev and fusen) |
| **Complexity Level** | L2 (Intermediate) |
| **Target Persona** | Data Scientist |
| **Location** | docs/notebooks-guide.md |
| **Priority** | P3 |
| **Aspect Ratio** | 16:10 |

## Purpose

Compare the two major notebook-to-library frameworks (nbdev for Python, fusen for R) side by side, showing their workflows, outputs, and when each is the right choice versus the thin notebook pattern.

## Key Message

"nbdev and fusen let you build tested, documented, installable libraries directly from notebooks -- the notebook IS the source code, not a wrapper around it."

## Literature Sources

| Source | Finding | DOI/URL |
|--------|---------|---------|
| Howard & Gugger 2022 | nbdev v2 with Quarto for docs generation | [fast.ai blog](https://www.fast.ai/posts/2022-07-28-nbdev2.html) |
| nbdev documentation | Notebook-driven development framework | [nbdev.fast.ai](https://nbdev.fast.ai/) |
| fusen (CRAN) | Inflate R packages from Rmd/Qmd files | [CRAN](https://cran.r-project.org/package=fusen) |
| GitHub Blog | nbdev: Productive and Collaborative Development | [github.blog](https://github.blog/developer-skills/github/nbdev-a-system-for-exploratory-programming/) |
| Rochette 2021 | fusen: Build a package from Rmd | [thinkr.fr](https://thinkr-open.github.io/fusen/) |

## Visual Concept

```
+---------------------------------------------------------------------------------+
|                    BUILDING LIBRARIES FROM NOTEBOOKS                             |
|                    nbdev (Python) and fusen (R)                                  |
+---------------------------------------------------------------------------------+
|                                                                                 |
|  TWO-COLUMN COMPARISON                                                          |
|  ======================                                                         |
|                                                                                 |
|  PYTHON: nbdev                        |  R: fusen                               |
|  ─────────────────────                |  ──────────────                         |
|                                       |                                         |
|  notebook.ipynb                       |  dev_history.Rmd                        |
|    |                                  |    |                                    |
|    | Directives:                      |    | Chunks:                            |
|    | #|export                         |    | ```{r function-name}              |
|    | #|hide                           |    | ```{r examples-name}              |
|    | #|test                           |    | ```{r tests-name}                 |
|    |                                  |    |                                    |
|    v                                  |    v                                    |
|  nb_export()                          |  fusen::inflate()                       |
|    |                                  |    |                                    |
|    +---> lib/module.py                |    +---> R/function.R                   |
|    |       (source code)              |    |       (source code)                |
|    |                                  |    |                                    |
|    +---> docs/ (Quarto site)          |    +---> vignettes/                     |
|    |       (documentation)            |    |       (documentation)              |
|    |                                  |    |                                    |
|    +---> tests/                       |    +---> tests/testthat/                |
|            (every cell = test)        |            (test chunks)                |
|                                       |                                         |
|  pip install -e .                     |  devtools::install()                    |
|                                       |                                         |
+---------------------------------------------------------------------------------+
|  KEY INNOVATIONS                                                                |
|  ===============                                                                |
|                                                                                 |
|  nbdev                                |  fusen                                  |
|  ─────                                |  ─────                                  |
|                                       |                                         |
|  Custom git merge driver              |  "If you know how to write              |
|  solves .ipynb JSON conflicts         |   RMarkdown, you know how               |
|  (the #1 pain point with              |   to build a package"                   |
|  notebook version control)            |                                         |
|                                       |  Docs-first development                 |
|  Switched to Quarto in v2 (2022)      |  produces better-documented             |
|  for documentation generation         |  packages (motivation)                  |
|                                       |                                         |
|  Success stories:                     |  On CRAN since 2021                     |
|  - fastai v2 (deep learning)          |  Used by ThinkR team                    |
|  - fastcore (Python foundations)      |  for client packages                    |
|  - ghapi (GitHub API client)          |                                         |
|                                       |                                         |
+---------------------------------------------------------------------------------+
|  WHEN TO USE WHAT                                                               |
|  ================                                                               |
|                                                                                 |
|  +---------------------------+  +---------------------------+                   |
|  | BUILDING A LIBRARY?       |  | BUILDING AN APPLICATION?  |                   |
|  |                           |  |                           |                   |
|  | Python: nbdev             |  | Use the THIN NOTEBOOK     |                   |
|  | R: fusen                  |  | pattern (fig-nb-07)       |                   |
|  |                           |  |                           |                   |
|  | Notebook IS the source    |  | Notebook IMPORTS from     |                   |
|  | Code exports to package   |  | tested src/ modules       |                   |
|  | Tests embedded in cells   |  | Tests are separate pytest |                   |
|  | Docs auto-generated       |  | Docs are README + MkDocs  |                   |
|  |                           |  |                           |                   |
|  | Examples:                 |  | Examples:                 |                   |
|  | fastai, research tools    |  | Foundation PLR, apps      |                   |
|  +---------------------------+  +---------------------------+                   |
|                                                                                 |
|  Foundation PLR uses THIN NOTEBOOK because we build an analysis                 |
|  application, not a reusable library. Our logic lives in src/.                  |
|                                                                                 |
+---------------------------------------------------------------------------------+
```

## Semantic Tags Used

| Element | Semantic Tag | Description |
|---------|-------------|-------------|
| nbdev column (Python) | `primary_pathway` | Deep blue - primary language |
| fusen column (R) | `highlight_accent` | Gold - secondary language |
| Export arrows | `healthy_normal` | Green - generation outputs |
| "When to use" boxes | `primary_pathway` / `healthy_normal` | Blue (library) / Green (app) |
| Divider lines | `secondary_pathway` | Gray separators |
| Our choice highlight | `healthy_normal` | Green - Foundation PLR uses thin |

## Content Elements

1. **Two-column workflow**: nbdev (Python) vs fusen (R) side by side
2. **Directive/chunk syntax**: #|export for nbdev, chunk labels for fusen
3. **Output triple**: Source code + documentation + tests from each
4. **Key innovations box**: Merge driver (nbdev), docs-first philosophy (fusen)
5. **Success stories**: fastai v2, fastcore, ghapi for nbdev
6. **When-to-use decision boxes**: Library vs application distinction
7. **Foundation PLR callout**: Why we chose thin notebook over nbdev

## Anti-Hallucination Rules

- DO NOT claim Foundation PLR uses nbdev or fusen. We use the thin notebook pattern.
- DO NOT invent success stories. fastai v2, fastcore, and ghapi are real nbdev-built projects.
- fusen has been on CRAN since 2021 (verify at cran.r-project.org/package=fusen).
- nbdev v2 switched to Quarto for docs in 2022 (fast.ai blog post dated 2022-07-28).
- The custom git merge driver is a real nbdev feature that resolves .ipynb JSON conflicts.
- DO NOT show AUROC or model performance. This is about development methodology.
- DO NOT claim nbdev is the only way to develop packages from notebooks. It is one approach.
- fusen::inflate() is the actual function name, not a placeholder.

## Text Content

### Title Text
"Building Libraries from Notebooks: nbdev and fusen"

### Caption
Two frameworks for building installable libraries directly from notebooks. Python's nbdev uses #|export directives to extract source code, generates Quarto documentation, and treats every cell as a test. Its custom git merge driver solves the notorious .ipynb JSON conflict problem. R's fusen uses labeled Rmd/Qmd chunks to inflate R packages with functions, vignettes, and tests. Both follow the "notebook IS the source" philosophy. Foundation PLR uses the thin notebook pattern instead because we build an analysis application, not a reusable library.

## Prompts for Nano Banana Pro

### Style Prompt
Two-column comparison layout on warm off-white background. Left column for Python/nbdev in deep blue, right column for R/fusen in gold. Flow arrows showing notebook-to-output transformations. Decision boxes at bottom with clear recommendation. Professional editorial design with clean typography.

### Content Prompt
Create "nbdev and fusen" two-column comparison:

**TOP - Parallel Workflows**:
- Left (Python): notebook.ipynb -> #|export -> lib/module.py + docs + tests
- Right (R): dev_history.Rmd -> fusen::inflate() -> R/ + vignettes/ + tests/

**MIDDLE - Key Innovations**:
- nbdev: Custom merge driver for JSON conflicts, Quarto docs
- fusen: Docs-first, "write RMarkdown = build a package"
- nbdev success stories: fastai, fastcore, ghapi

**BOTTOM - Decision Matrix**:
- Library -> nbdev/fusen (notebook IS source)
- Application -> thin notebook (notebook IMPORTS source)

## Alt Text

Two-column comparison of notebook-to-library frameworks. Left column Python nbdev: notebook.ipynb with #|export directives generates lib/module.py source code, Quarto documentation site, and tests from every cell. Key innovation: custom git merge driver solves .ipynb JSON conflicts. Success stories: fastai v2, fastcore, ghapi. Right column R fusen: dev_history.Rmd with labeled chunks generates R/ source, vignettes/, and tests/testthat/ via fusen::inflate(). Philosophy: if you can write RMarkdown, you can build a package. On CRAN since 2021. Decision matrix: building a library use nbdev or fusen; building an application use thin notebook pattern. Foundation PLR uses thin notebook.

## References

- nbdev documentation: [nbdev.fast.ai](https://nbdev.fast.ai/)
- Howard & Gugger 2022: nbdev v2 announcement ([fast.ai blog](https://www.fast.ai/posts/2022-07-28-nbdev2.html))
- fusen: CRAN package ([cran.r-project.org/package=fusen](https://cran.r-project.org/package=fusen))
- fusen documentation: Rochette ([thinkr-open.github.io/fusen](https://thinkr-open.github.io/fusen/))
- GitHub Blog: nbdev overview ([github.blog](https://github.blog/developer-skills/github/nbdev-a-system-for-exploratory-programming/))
- fastai v2: Built with nbdev ([github.com/fastai/fastai](https://github.com/fastai/fastai))

## Related Figures

- **fig-nb-07**: From Notebook to Production (the thin notebook pattern we chose)
- **fig-repro-09**: R Package Ecosystem (fusen fits within CRAN ecosystem)
- **fig-repo-30**: Python-R Interop (how our repo bridges both languages)

## Cross-References

Reader flow: **fig-nb-07** (pattern overview, "when to use nbdev") -> **THIS FIGURE** (nbdev/fusen deep dive) -> **fig-nb-05** (how to test whichever pattern you chose)

## Status

- [x] Draft created
- [ ] Generated
- [ ] Placed in docs/notebooks-guide.md
