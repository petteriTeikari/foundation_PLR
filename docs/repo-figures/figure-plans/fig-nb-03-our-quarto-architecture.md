# fig-nb-03: Our Quarto Architecture

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-nb-03 |
| **Title** | Our Quarto Architecture |
| **Complexity Level** | L2 (Technical) |
| **Target Persona** | Data Scientist |
| **Location** | notebooks/README.md |
| **Priority** | P1 |
| **Aspect Ratio** | 16:9 |

## Purpose

Document Foundation PLR's specific Quarto notebook setup: project configuration, notebook inventory, enforcement layers, and data flow. This is the project-specific "how we did it" complement to the general notebook landscape (fig-nb-01) and hidden state problem (fig-nb-02).

## Key Message

"Our Quarto setup enforces reproducibility at three layers -- pre-commit AST checks, CI rendering, and freeze snapshots -- while keeping notebooks read-only consumers of DuckDB data."

## Visual Concept

```
+---------------------------------------------------------------------------+
|                     OUR QUARTO ARCHITECTURE                                |
|                     Foundation PLR Notebook Setup                          |
+---------------------------------------------------------------------------+
|                                                                           |
|  PROJECT CONFIG                                                           |
|  ==============                                                           |
|                                                                           |
|  _quarto.yml                                                              |
|  +---------------------------------------------------------------+       |
|  | project:                                                       |       |
|  |   type: website                                                |       |
|  | execute:                                                       |       |
|  |   freeze: auto     <-- pre-computed results in git             |       |
|  |   error: false      <-- any error = build fails                |       |
|  | format:                                                        |       |
|  |   html:                                                        |       |
|  |     theme: cosmo                                               |       |
|  |     code-fold: true                                            |       |
|  +---------------------------------------------------------------+       |
|                                                                           |
+---------------------------------------------------------------------------+
|                                                                           |
|  NOTEBOOK INVENTORY                                                       |
|  ==================                                                       |
|                                                                           |
|  notebooks/                                                               |
|  +-------------------------------+  +-------------------------------+     |
|  | 01-pipeline-walkthrough.qmd   |  | 02-reproduce-and-extend.qmd   |    |
|  | ============================  |  | ============================  |    |
|  | 10 code cells                 |  | 21 code cells                 |    |
|  | 4 Mermaid diagrams            |  | Contribution workflow         |    |
|  |                               |  |                               |    |
|  | Purpose:                      |  | Purpose:                      |    |
|  | Walk through the full         |  | Guide contributors through    |    |
|  | preprocessing pipeline        |  | reproducing and extending     |    |
|  | from raw PLR to metrics       |  | the analysis                  |    |
|  +-------------------------------+  +-------------------------------+     |
|                                                                           |
|  extensions/                                                              |
|  +-------------------------------+                                        |
|  | _extension.yml                |  Quarto extension template            |
|  | README.md                     |  for project-specific filters         |
|  +-------------------------------+                                        |
|                                                                           |
+---------------------------------------------------------------------------+
|                                                                           |
|  THREE ENFORCEMENT LAYERS                                                 |
|  ========================                                                 |
|                                                                           |
|  LAYER 1: PRE-COMMIT         LAYER 2: CI               LAYER 3: FREEZE   |
|  (before git commit)         (on push)                  (in git)          |
|  +-------------------+       +-------------------+      +--------------+  |
|  | AST check:        |       | quarto render     |      | _freeze/     |  |
|  | - No hardcoded    |       | --to html         |      |  01-pipe../  |  |
|  |   paths           |  -->  | - Fresh kernel    |  --> |   execute-   |  |
|  | - Imports from    |       | - error:false     |      |   results/   |  |
|  |   src.viz only    |       |   enforced        |      |   *.json     |  |
|  | - No plt.savefig  |       | - All cells must  |      |  02-repr../  |  |
|  +-------------------+       |   pass            |      |   ...        |  |
|                              +-------------------+      +--------------+  |
|                                                         Committed to git  |
|  Catches: style             Catches: runtime            Caches: outputs   |
|  violations early           errors, broken code         for fast CI       |
|                                                                           |
+---------------------------------------------------------------------------+
|                                                                           |
|  DATA FLOW (READ-ONLY)                                                    |
|  ======================                                                   |
|                                                                           |
|  +---------------+       +------------------+       +----------------+    |
|  | DuckDB        |  ---> | notebooks/       |  ---> | plt.show()     |   |
|  | 406 configs   |  SQL  | .qmd files       |  fig  | (inline only)  |   |
|  | (read-only)   |       | (read-only)      |       | NO savefig()   |   |
|  +---------------+       +------------------+       +----------------+    |
|                                                                           |
|  Import pattern:                                                          |
|  from src.viz.plot_config import COLORS, setup_style, FIXED_CLASSIFIER    |
|  from src.viz.plot_config import save_figure  # for standalone scripts    |
|                                                                           |
|  Notebooks call setup_style() then use COLORS dict -- no hex codes.       |
|  Notebooks SELECT from DuckDB -- no sklearn, no metric computation.       |
|                                                                           |
+---------------------------------------------------------------------------+
```

## Semantic Tags Used

| Element | Semantic Tag | Description |
|---------|-------------|-------------|
| _quarto.yml config box | `primary_pathway` | Deep blue -- central config |
| Notebook cards | `melanopsin` | Gold -- content containers |
| Layer 1 (pre-commit) | `melanopsin` | Gold -- first gate |
| Layer 2 (CI) | `primary_pathway` | Deep blue -- second gate |
| Layer 3 (freeze) | `healthy_normal` | Green -- cached outputs |
| DuckDB box | `primary_pathway` | Deep blue -- data source |
| plt.show() box | `healthy_normal` | Green -- correct output |
| "error: false" annotation | `abnormal_warning` | Red -- strict enforcement |
| Import pattern text | `secondary_pathway` | Gray -- code reference |

## Content Elements

1. **_quarto.yml config block**: Actual project configuration with annotations
2. **Notebook inventory**: Two cards showing 01-pipeline-walkthrough (10 cells, 4 Mermaid) and 02-reproduce-and-extend (21 cells)
3. **Extensions directory**: Template and README for project-specific Quarto filters
4. **Three enforcement layers**: Pre-commit (AST), CI (quarto render), Freeze (_freeze/ in git)
5. **Data flow diagram**: DuckDB --> notebooks --> plt.show() (read-only chain)
6. **Import pattern**: Exact import lines from src.viz.plot_config
7. **Anti-pattern callouts**: No plt.savefig(), no sklearn imports, no hex codes

## Anti-Hallucination Rules

- The _quarto.yml settings shown MUST match the actual file in the repository. Verify before generation.
- Cell counts (10 cells, 4 Mermaid for notebook 01; 21 cells for notebook 02) MUST be verified against actual .qmd files.
- The import pattern uses `src.viz.plot_config` not `src.visualization.config` or any other path.
- DuckDB contains 406 configs (11 outlier x 8 imputation x 5 classifier - invalid combos). Do NOT say 440 or any other number without checking.
- `freeze: auto` means freeze is updated when source changes. Do NOT confuse with `freeze: true` (never re-execute).
- Notebooks are READ-ONLY consumers of DuckDB. They NEVER compute metrics. This is the computation decoupling rule (CRITICAL-FAILURE-003).
- The extensions/ directory is for Quarto extensions, not Python extensions.

## Text Content

### Title Text
"Our Quarto Architecture: Foundation PLR Notebook Setup"

### Caption
Foundation PLR's Quarto architecture enforces reproducibility at three layers. Project configuration (_quarto.yml) sets freeze:auto for cached execution and error:false for strict failure. Two notebooks cover the pipeline walkthrough (10 cells, 4 Mermaid diagrams) and contribution workflow (21 cells). Pre-commit hooks run AST checks for style violations, CI runs quarto render with fresh kernels, and _freeze/ stores pre-computed results in git for fast rendering. Data flows read-only from DuckDB (406 configs) through notebooks to inline plots -- no savefig(), no metric computation, no hardcoded values.

## Prompts for Nano Banana Pro

### Style Prompt
Architecture diagram on off-white background. YAML config block with monospace font and syntax highlighting. Card layout for notebooks. Three-column enforcement layer comparison. Clean flowing arrows for data flow. Professional documentation quality. Matte finish, no glowing effects.

### Content Prompt
Create an "Our Quarto Architecture" infographic:

**TOP - Project Config**: _quarto.yml shown as a styled config block with key settings annotated (freeze:auto, error:false, theme:cosmo)

**MIDDLE - Notebook Inventory**: Two notebook cards side by side showing name, cell count, and purpose. Extensions directory below.

**MIDDLE - Three Enforcement Layers**: Three columns showing pre-commit AST checks, CI quarto render, and _freeze/ directory. Arrows showing the progression from early catch to cached output.

**BOTTOM - Data Flow**: Linear flow from DuckDB (406 configs) through .qmd files to plt.show() inline output. Import pattern shown in monospace.

## Alt Text

Architecture diagram of Foundation PLR's Quarto notebook setup. Top section shows _quarto.yml configuration with freeze:auto and error:false settings. Middle section displays two notebook cards: 01-pipeline-walkthrough (10 code cells, 4 Mermaid diagrams) and 02-reproduce-and-extend (21 code cells, contribution workflow), plus extensions directory. Three enforcement layers shown as columns: pre-commit AST checks catch style violations, CI runs quarto render with fresh kernels, and _freeze/ directory stores pre-computed results in git. Bottom section shows read-only data flow from DuckDB (406 configurations) through notebooks to inline plt.show() output, with import pattern from src.viz.plot_config.

## Related Figures

- **fig-nb-01**: Notebook landscape (why we chose Quarto)
- **fig-nb-02**: Hidden state problem (what we are preventing)
- **fig-nb-04**: Quarto freeze deep dive (Layer 3 in detail)
- **fig-repo-18**: Two-block architecture (the extraction/analysis split this enforces)
- **fig-repo-74**: Computation decoupling enforcement (the rule notebooks follow)

## Cross-References

Reader flow: **fig-nb-01** (why Quarto?) --> **fig-nb-02** (what problem?) --> **THIS FIGURE** (our solution) --> **fig-nb-04** (freeze deep dive)

This figure is the project-specific implementation of concepts introduced in fig-nb-01 and fig-nb-02.

## References

1. Quarto project documentation. https://quarto.org/docs/projects/quarto-projects.html
2. Quarto execution freeze. https://quarto.org/docs/projects/code-execution.html
3. Foundation PLR CLAUDE.md -- computation decoupling rules (CRITICAL-FAILURE-003)

## Status

- [x] Draft created
- [ ] Generated
- [ ] Placed in notebooks/README.md
